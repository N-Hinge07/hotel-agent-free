# src/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse
from .agent import run_agent

APP_TITLE = "Hotel Agent - Gemini/OpenAI Fallback"
APP_VERSION = "0.1.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup code
    print("Starting app - lifespan startup")
    # optionally: warm clients or log backend choice
    yield
    # shutdown code
    print("Stopping app - lifespan shutdown")

app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

# Optional: allow CORS for local dev / UI apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # narrow in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "app": APP_TITLE, "version": APP_VERSION}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        response = run_agent(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))