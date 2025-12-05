# src/agent.py
import os
import logging
from typing import Optional

from .schemas import ChatRequest, ChatResponse

# Try import SDKs lazily to avoid hard dependency errors
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
    # simple console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


class AgentClient:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

        self.backend = "mock"
        if self.gemini_key and _HAS_GENAI:
            self.backend = "gemini"
            genai.configure(api_key=self.gemini_key)
            # pick model you have access to
            self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            logger.info("AgentClient: using Gemini backend")
        elif self.openai_key and _HAS_OPENAI:
            self.backend = "openai"
            openai.api_key = self.openai_key
            # set a sensible default or override with env OPENAI_MODEL
            self.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"AgentClient: using OpenAI backend (model={self.openai_model})")
        else:
            self.backend = "mock"
            logger.info("AgentClient: no API key found, using mock backend (echo)")

    def _call_gemini(self, prompt: str) -> str:
        """
        Calls google.generativeai GenerativeModel. Adjust extraction if SDK differs.
        """
        model = genai.GenerativeModel(self.gemini_model)
        resp = model.generate_content(prompt)
        # Many SDK versions return a `.text` or `.result` or nested structure.
        text = getattr(resp, "text", None)
        if text is None:
            # try known fallback shapes
            try:
                # sometimes response.output[0].content[0].text (SDK variants)
                out = getattr(resp, "output", None)
                if out and len(out) > 0:
                    first = out[0]
                    # attempt a few common keys
                    for key in ("content", "text", "message", "body"):
                        if isinstance(first, dict) and key in first:
                            return str(first[key])
            except Exception:
                pass
            # last-resort
            text = str(resp)
        return text

    def _call_openai(self, prompt: str) -> str:
        """
        Uses legacy ChatCompletion or Completion depending on availability.
        This may need small changes depending on which openai package version you installed.
        """
        # prefer ChatCompletion if available
        try:
            if hasattr(openai, "ChatCompletion"):
                completion = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                # standard path
                choice = completion.choices[0]
                # new SDK uses choice.message.content or choice.text
                msg = None
                if hasattr(choice, "message") and isinstance(choice.message, dict):
                    msg = choice.message.get("content")
                elif hasattr(choice, "message") and hasattr(choice.message, "get"):
                    msg = choice.message.get("content")
                else:
                    msg = getattr(choice, "text", None)
                if msg is None:
                    return str(completion)
                return msg
            else:
                # fallback to Completion API
                completion = openai.Completion.create(
                    model=self.openai_model,
                    prompt=prompt,
                    max_tokens=512,
                )
                return completion.choices[0].text
        except Exception as e:
            logger.exception("OpenAI call failed")
            raise

    def run(self, request: ChatRequest) -> ChatResponse:
        user_prompt = request.message

        try:
            if self.backend == "gemini":
                reply_text = self._call_gemini(user_prompt)
            elif self.backend == "openai":
                reply_text = self._call_openai(user_prompt)
            else:
                # Mock / local fallback: echo + small transformation
                reply_text = f"[MOCK] Echo: {user_prompt}"

            return ChatResponse(
                session_id=request.session_id,
                reply=reply_text,
                intent=None,
                suggested_actions=None,
                context={"backend": self.backend},
            )

        except Exception as e:
            logger.exception("Agent call error")
            return ChatResponse(
                session_id=request.session_id,
                reply=f"Agent error: {e}",
            )


# single client instance for reuse
client = AgentClient()


def run_agent(request: ChatRequest) -> ChatResponse:
    return client.run(request)