# src/agent.py
import os
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .schemas import ChatRequest, ChatResponse

# Lazy LLM imports (optional; safe if not installed)
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    genai = None
    _HAS_GENAI = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    openai = None
    _HAS_OPENAI = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


@dataclass
class OrderItem:
    name: str
    quantity: int = 1
    tags: Optional[List[str]] = None
    available: bool = True
    prep_time_min: Optional[int] = None
    menu_id: Optional[str] = None


class AgentClient:
    def __init__(self):
        # backend selection
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.backend = "mock"
        if self.gemini_key and _HAS_GENAI:
            self.backend = "gemini"
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception:
                pass
        elif self.openai_key and _HAS_OPENAI:
            self.backend = "openai"
        logger.info("Agent backend: %s", self.backend)

        # synonyms to help matching
        self.synonyms = {
            "fries": "French Fries",
            "chips": "French Fries",
            "lava cake": "Chocolate Lava Cake",
            "chicken sandwich": "Grilled Chicken Sandwich",
            "caesar salad": "Veg Caesar Salad",
            "salad": "Veg Caesar Salad",
        }

        # load menu (data/menu.json or data/data.json fallback)
        self.menu = self._load_menu()

        # in-memory session store: session_id -> context dict
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Menu loader
    # -------------------------
    def _load_menu(self) -> List[Dict[str, Any]]:
        base = Path(__file__).resolve().parents[1]
        menu_path1 = base / "data" / "menu.json"
        menu_path2 = base / "data" / "data.json"
        chosen = None
        if menu_path1.exists():
            chosen = menu_path1
        elif menu_path2.exists():
            chosen = menu_path2

        if chosen:
            try:
                with open(chosen, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("Loaded menu from %s (%d items)", chosen, len(data))
                    return data
            except Exception as e:
                logger.error("Failed to load menu file: %s", e)
        logger.warning("No menu file found; starting with empty menu.")
        return []

    # -------------------------
    # Helpers: normalize & synonyms
    # -------------------------
    def _normalize(self, s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower())

    # -------------------------
    # Matching items by name/tag/synonym
    # -------------------------
    def match_items(self, text: str) -> List[OrderItem]:
        """Return list of OrderItem matched from the menu."""
        if not self.menu:
            return []

        normalized = self._normalize(text)

        # apply synonyms
        for k, v in self.synonyms.items():
            if k in normalized:
                normalized = normalized.replace(k, self._normalize(v))

        found: List[OrderItem] = []
        for item in self.menu:
            name = item.get("name", "")
            tags = item.get("tags", []) or []
            id_ = item.get("id") or item.get("_id") or name

            # exact name contained
            if self._normalize(name) in normalized:
                found.append(OrderItem(name=name, tags=tags, available=bool(item.get("available", True)),
                                       prep_time_min=item.get("prep_time_min"), menu_id=str(id_)))
                continue

            # token match in name
            for tok in normalized.split():
                if tok and tok in self._normalize(name).split():
                    found.append(OrderItem(name=name, tags=tags, available=bool(item.get("available", True)),
                                           prep_time_min=item.get("prep_time_min"), menu_id=str(id_)))
                    break

            # tag match
            for tag in tags:
                if tag and tag in normalized:
                    found.append(OrderItem(name=name, tags=tags, available=bool(item.get("available", True)),
                                           prep_time_min=item.get("prep_time_min"), menu_id=str(id_)))
                    break

        # dedupe by menu_id
        uniq: Dict[str, OrderItem] = {}
        for o in found:
            uniq[o.menu_id or o.name] = o
        return list(uniq.values())

    # -------------------------
    # Simple parser: intents & quantity detection
    # -------------------------
    def parse(self, text: str) -> Dict[str, Any]:
        t = text.strip().lower()

        # greeting
        if re.search(r"\b(hi|hello|hey|good morning|good evening)\b", t):
            return {"intent": "greeting", "items": [], "dietary": []}

        # cancel
        if re.search(r"\b(cancel|never mind|stop|don't)\b", t):
            return {"intent": "cancel", "items": [], "dietary": []}

        # confirm (short positive)
        if re.fullmatch(r"\b(yes|yep|confirm|please|sure|ok)\b", t):
            return {"intent": "confirm", "items": [], "dietary": []}

        # set preferences
        if any(k in t for k in ("vegetarian", "vegan", "no onion", "no dairy", "dairy-free", "nut allergy")):
            dietary = []
            if "vegetarian" in t or "vegan" in t:
                dietary.append("vegetarian")
            if "no onion" in t:
                dietary.append("no_onion")
            if "no dairy" in t or "dairy-free" in t:
                dietary.append("no_dairy")
            if "nut" in t:
                dietary.append("no_nuts")
            return {"intent": "set_preference", "items": [], "dietary": dietary}

        # quantity pattern e.g., "2 fries" or "2 x fries"
        qty_match = re.search(r"(\d+)\s*(?:x|pcs|pieces)?\s*(.+)", t)
        if qty_match:
            qty = int(qty_match.group(1))
            item_text = qty_match.group(2)
            matches = self.match_items(item_text)
            for m in matches:
                m.quantity = qty
            if matches:
                return {"intent": "order_food", "items": matches, "dietary": []}

        # order verbs or short utterance (1-3 words)
        if re.search(r"\b(order|i want|i'd like|get me|bring me|please get|i need)\b", t) or len(t.split()) <= 3:
            matches = self.match_items(text)
            if matches:
                return {"intent": "order_food", "items": matches, "dietary": []}
            else:
                # ambiguous short query -> clarify
                return {"intent": "clarify", "items": [], "dietary": []}

        # fallback unknown
        return {"intent": "unknown", "items": [], "dietary": []}

    # -------------------------
    # Session management (simple in-memory)
    # -------------------------
    def _load_session(self, session_id: Optional[str]) -> (str, Dict[str, Any]):
        if not session_id:
            session_id = f"guest-{int(time.time())}"
        if session_id not in self.sessions:
            self.sessions[session_id] = {"preferences": {}, "pending": None, "history": []}
        return session_id, self.sessions[session_id]

    def _save_session(self, session_id: str, ctx: Dict[str, Any]):
        self.sessions[session_id] = ctx

    # -------------------------
    # Core orchestrator
    # -------------------------
    def run(self, request: ChatRequest) -> ChatResponse:
        text = request.message
        session_id, ctx = self._load_session(request.session_id)

        parsed = self.parse(text)

        # greeting
        if parsed["intent"] == "greeting":
            return ChatResponse(session_id=session_id, reply="Hello! How can I help with room service today?", intent="greeting")

        # set preferences
        if parsed["intent"] == "set_preference":
            for p in parsed.get("dietary", []):
                ctx["preferences"][p] = True
            self._save_session(session_id, ctx)
            return ChatResponse(session_id=session_id, reply=f"Saved preferences: {parsed.get('dietary')}", intent="set_preference", context={"preferences": ctx["preferences"]})

        # cancel
        if parsed["intent"] == "cancel":
            ctx["pending"] = None
            self._save_session(session_id, ctx)
            return ChatResponse(session_id=session_id, reply="Cancelled your pending request.", intent="cancel")

        # confirm -> if pending order exists, place it (in-memory)
        if parsed["intent"] == "confirm":
            if ctx.get("pending"):
                order = ctx["pending"]
                ctx["history"].append(order)
                ctx["pending"] = None
                self._save_session(session_id, ctx)
                return ChatResponse(session_id=session_id, reply="Order placed. Thank you!", intent="confirm", context={"history": ctx["history"]})
            else:
                return ChatResponse(session_id=session_id, reply="Nothing to confirm.", intent="confirm")

        # order flow
        if parsed["intent"] == "order_food":
            items: List[OrderItem] = parsed.get("items", [])
            if not items:
                return ChatResponse(session_id=session_id, reply="I couldn't find that item. Can you name it differently?", intent="clarify", suggested_actions=["provide_item_name"])

            # check preferences and availability
            prefs = ctx.get("preferences", {})
            conflicts = []
            unavailable = []
            for o in items:
                tags = o.tags or []
                if prefs.get("vegetarian") and any(t in ("non-veg", "chicken", "beef", "pork", "fish", "egg") for t in tags):
                    conflicts.append(o)
                if not o.available:
                    unavailable.append(o)

            if conflicts:
                names = ", ".join([c.name for c in conflicts])
                ctx["pending"] = {"items": [asdict(i) for i in items]}
                self._save_session(session_id, ctx)
                return ChatResponse(
                    session_id=session_id,
                    reply=f"These items conflict with your dietary preferences: {names}. Replace or remove?",
                    intent="confirm",
                    suggested_actions=["replace_item", "remove_item"],
                    context={"conflicts": [asdict(c) for c in conflicts]},
                )

            if unavailable:
                names = ", ".join([u.name for u in unavailable])
                ctx["pending"] = {"items": [asdict(i) for i in items]}
                self._save_session(session_id, ctx)
                return ChatResponse(
                    session_id=session_id,
                    reply=f"Sorry, these are currently unavailable: {names}. Would you like alternatives?",
                    intent="clarify",
                    suggested_actions=["offer_alternatives"],
                    context={"unavailable": [asdict(u) for u in unavailable]},
                )

            # else prepare confirmation with ETA
            eta = sum([o.prep_time_min or 0 for o in items])
            ctx["pending"] = {"items": [asdict(i) for i in items], "eta_min": eta}
            self._save_session(session_id, ctx)
            item_names = ", ".join([f"{i.quantity} x {i.name}" for i in items])
            return ChatResponse(
                session_id=session_id,
                reply=f"Confirming: {item_names}. ETA ~{eta} minutes. Shall I place the order?",
                intent="confirm_request",
                suggested_actions=["confirm", "modify", "cancel"],
                context={"pending": ctx["pending"]},
            )

        # clarify or unknown
        if parsed["intent"] in ("clarify", "unknown"):
            return ChatResponse(
                session_id=session_id,
                reply="I didn't understand — can you specify the dish name (e.g., 'Grilled Chicken Sandwich')?",
                intent="clarify",
                suggested_actions=["provide_item_name"],
            )

        # fallback
        return ChatResponse(session_id=session_id, reply="Sorry, I couldn't process that.", intent="error")


# single client instance & helper
client = AgentClient()

def run_agent(request: ChatRequest) -> ChatResponse:
    return client.run(request)