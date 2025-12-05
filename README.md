# Hotel Agent Free

## Setup
1. Create & activate venv:
   python -m venv venv
   source venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. (Optional) set keys:
   export GEMINI_API_KEY="..."
   export OPENAI_API_KEY="..."

4. Run server:
   uvicorn src.main:app --reload --port 8000

5. Open Swagger UI:
   http://127.0.0.1:8000/docs

If no keys are set, the /chat endpoint will return a mock reply for testing.
