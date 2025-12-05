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

# try LLM libs lazily (we won't require them for mock mode)
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
        # backend selection: gemini / openai / mock
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

        # load local menu
        self.menu = self._load_menu()
        # simple in-memory sessions: session_id -> context dict
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _load_menu(self) -> List[Dict[str, Any]]:
        base = Path(__file__).resolve().parents[1]
        menu_path = base / "data" / "menu.json"
        if menu_path.exists():
            try:
                with open(menu_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("Loaded menu.json with %d items", len(data))
                    return data
            except Exception as e:
                logger.error("Failed to load data/menu.json: %s", e)
        logger.warning("No menu.json found; starting with empty menu")
        return []

    def _normalize(self, s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower())

    def match_items(self, text: str) -> List[OrderItem]:
        """Match menu items by name or tag using simple rules."""
        normalized = self._normalize(text)
        found: List[OrderItem] = []
        for item in self.menu:
            name = item.get("name", "")
            tags = item.get("tags", []) or []
            if self._normalize(name) in normalized:
                found.append(OrderItem(name=name, tags=tags, available=item.get("available", True),
                                       prep_time_min=item.get("prep_time_min"), menu_id=item.get("id") or item.get("id")))
                continue
            # token matching
            for tok in normalized.split():
                if tok and tok in self._normalize(name).split():
                    found.append(OrderItem(name=name, tags=tags, available=item.get("available", True),
                                           prep_time_min=item.get("prep_time_min"), menu_id=item.get("id") or item.get("id")))
                    break
            # tag match
            for tag in tags:
                if tag and tag in normalized:
                    found.append(OrderItem(name=name, tags=tags, available=item.get("available", True),
                                           prep_time_min=item.get("prep_time_min"), menu_id=item.get("id") or item.get("id")))
                    break
        # dedupe by name
        uniq = {}
        for o in found:
            uniq[o.name] = o
        return list(uniq.values())

    def parse(self, text: str) -> Dict[str, Any]:
        """Very small rule-based parser: greeting, cancel, confirm, set pref, order."""
        t = text.strip().lower()

        if re.search(r"\b(hi|hello|hey)\b", t):
            return {"intent": "greeting", "items": [], "dietary": []}
        if re.search(r"\b(cancel|never mind|stop)\b", t):
            return {"intent": "cancel", "items": [], "dietary": []}
        if re.fullmatch(r"\b(yes|confirm|please)\b", t):
            return {"intent": "confirm", "items": [], "dietary": []}
        # preference
        if "vegetarian" in t or "vegan" in t or "no onion" in t or "no dairy" in t:
            dietary = []
            if "vegetarian" in t or "vegan" in t:
                dietary.append("vegetarian")
            if "no onion" in t:
                dietary.append("no_onion")
            if "no dairy" in t or "dairy-free" in t:
                dietary.append("no_dairy")
            return {"intent": "set_preference", "items": [], "dietary": dietary}
        # order keywords
        if re.search(r"\b(order|i want|i'd like|get me|bring me|please get|i need)\b", t) or len(t.split()) <= 3:
            items = self.match_items(text)
            if items:
                return {"intent": "order_food", "items": items, "dietary": []}
            # fallback: if nothing matched, return unknown but candidate for clarification
            return {"intent": "clarify", "items": [], "dietary": []}
        return {"intent": "unknown", "items": [], "dietary": []}

    def _save_session(self, session_id: str, ctx: Dict[str, Any]):
        self.sessions[session_id] = ctx

    def _load_session(self, session_id: str) -> Dict[str, Any]:
        if not session_id:
            session_id = f"guest-{int(time.time())}"
        if session_id not in self.sessions:
            self.sessions[session_id] = {"preferences": {}, "pending": None, "history": []}
        return self.sessions[session_id]

    def run(self, request: ChatRequest) -> ChatResponse:
        text = request.message
        session_id = request.session_id or f"guest-{request.guest_id or int(time.time())}"
        ctx = self._load_session(session_id)

        parsed = self.parse(text)

        # greeting
        if parsed["intent"] == "greeting":
            return ChatResponse(session_id=session_id, reply="Hello! How can I help with room service?", intent="greeting")

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

        # confirm (place pending order)
        if parsed["intent"] == "confirm":
            if ctx.get("pending"):
                order = ctx["pending"]
                ctx["history"].append(order)
                ctx["pending"] = None
                self._save_session(session_id, ctx)
                return ChatResponse(session_id=session_id, reply="Order placed. Thank you!", intent="confirm", context={"history": ctx["history"]})
            else:
                return ChatResponse(session_id=session_id, reply="Nothing to confirm — no pending order.", intent="confirm")

        # placing an order
        if parsed["intent"] == "order_food":
            items = parsed.get("items", [])
            # check dietary conflicts using preferences
            prefs = ctx.get("preferences", {})
            conflicts = []
            unavailable = []
            for o in items:
                # conflict: vegetarian preference but item non-veg tag
                tags = (o.tags or [])
                if prefs.get("vegetarian") and any(t in ("non-veg", "chicken", "beef", "pork", "fish", "egg") for t in tags):
                    conflicts.append(o)
                if not o.available:
                    unavailable.append(o)

            if conflicts:
                names = ", ".join([c.name for c in conflicts])
                ctx["pending"] = {"items": [asdict(i) for i in items]}
                self._save_session(session_id, ctx)
                return ChatResponse(session_id=session_id, reply=f"These items conflict with your dietary preferences: {names}. Replace or remove?", intent="confirm", suggested_actions=["replace_item", "remove_item"], context={"conflicts":[asdict(c) for c in conflicts]})

            if unavailable:
                names = ", ".join([u.name for u in unavailable])
                ctx["pending"] = {"items": [asdict(i) for i in items]}
                self._save_session(session_id, ctx)
                return ChatResponse(session_id=session_id, reply=f"Sorry, these are unavailable: {names}. Would you like alternatives?", intent="clarify", suggested_actions=["offer_alternatives"], context={"unavailable":[asdict(u) for u in unavailable]})

            # OK: prepare confirmation
            eta = sum([o.prep_time_min or 0 for o in items])
            ctx["pending"] = {"items": [asdict(i) for i in items], "eta_min": eta}
            self._save_session(session_id, ctx)
            item_names = ", ".join([f"{i.quantity} x {i.name}" for i in items])
            return ChatResponse(session_id=session_id, reply=f"Confirm: {item_names}. ETA ~{eta} min. Shall I place the order?", intent="confirm_request", suggested_actions=["confirm", "modify", "cancel"], context={"pending": ctx["pending"]})

        # clarify / unknown
        if parsed["intent"] in ("clarify", "unknown"):
            return ChatResponse(session_id=session_id, reply="I didn't catch that item. Could you specify the dish name (e.g., 'Grilled Chicken Sandwich')?", intent="clarify", suggested_actions=["provide_item_name"])

        # fallback
        return ChatResponse(session_id=session_id, reply="Sorry, I couldn't process that.", intent="error")

# single client instance and helper function
client = AgentClient()

def run_agent(request: ChatRequest) -> ChatResponse:
    return client.run(request)