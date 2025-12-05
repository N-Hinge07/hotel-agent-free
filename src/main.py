import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .agent import run_agent, client

APP_TITLE = "Hotel Agent - Simple"
APP_VERSION = "0.1.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app - backend=%s", getattr(client, "backend", "mock"))
    yield
    logger.info("Stopping app")

app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local/demo; narrow in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "app": APP_TITLE, "version": APP_VERSION, "backend": getattr(client, "backend", "mock")}

@app.get("/health")
def health():
    return {"status": "ok", "backend": getattr(client, "backend", "mock")}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        response = run_agent(request)
        return response
    except Exception:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")