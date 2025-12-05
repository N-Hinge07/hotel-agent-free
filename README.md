# Hotel Agent - Room Service (Simple)

## Overview
Small FastAPI app that implements a hotel room service agent:
- Parse natural language orders (simple rules + synonyms)
- Match menu items from `data/menu.json`
- Handle dietary preferences, availability, and confirmations
- In-memory session handling (multi-turn)
- Works without any API key (mock mode). Optional Gemini/OpenAI support if keys provided.

## Run locally
1. Create a virtual env and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
