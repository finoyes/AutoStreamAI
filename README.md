# AutoStream AI Agent

An intelligent conversational sales agent built with **LangGraph** and **Google Gemini** for AutoStream — an AI-powered video editing platform for content creators.

## What It Does

| Capability | Description |
|---|---|
| **RAG-Powered Q&A** | Answers pricing, policy, and feature questions from a structured knowledge base |
| **Intent Classification** | Detects greetings, info requests, and sign-up intent in real time |
| **Multi-Turn Lead Capture** | Collects Name → Email → Platform one at a time before firing the CRM tool |
| **Tool Calling** | Triggers `mock_lead_capture` only when all three fields are present |
| **Persistent Memory** | Maintains conversation state across turns within a session |

## Architecture

```
User Message
     │
     ▼
┌────────────────┐
│ classify_intent │  ← LLM classifies: greeting / info / signup
└───────┬────────┘
        │
   ┌────┴─────┐
   ▼          ▼
respond    collect_lead ◄──┐
 _info        │             │ (loops via user replies)
   │     ┌────┴─────┐      │
   │     │ all fields?│─No─►┘
   │     └────┬─────┘
   │          │ Yes
   │          ▼
   │     call_tool  →  mock_lead_capture
   │          │
   ▼          ▼
  END        END
```

## Project Structure

```
autostream-agent/
├── data/
│   └── knowledge_base.json      # Pricing, policies, FAQ (RAG source)
├── src/
│   ├── __init__.py
│   ├── agent.py                 # LangGraph state machine & LLM logic
│   ├── tools.py                 # mock_lead_capture tool
│   ├── rag_engine.py            # JSON knowledge base loader & retriever
│   └── state.py                 # AgentState TypedDict schema
├── .env                         # API keys (GEMINI_API_KEY)
├── main.py                      # CLI entry point
├── gui.py                       # Desktop GUI entry point
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone Repository

```bash
cd AutoStreamAI
```

### 2. Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your API Key

Edit `.env` and paste your Google Gemini API key:

```
GEMINI_API_KEY=your_actual_key_here
```

> Get a key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 5. Run (CLI)

```bash
python main.py
```

### 6. Run (Desktop GUI)

```bash
python gui.py
```

### 7. Try These Conversations

| Scenario | Example Messages |
|---|---|
| **Pricing inquiry** | "What plans do you offer?" / "How much is Pro?" |
| **Policy question** | "What's your refund policy?" / "Can I cancel anytime?" |
| **Sign-up flow** | "I want to sign up" → provide name → email → platform |
| **FAQ** | "What is AutoStream?" / "What video formats do you support?" |

## WhatsApp Integration (Future)

To deploy this agent to WhatsApp:

1. **Twilio / Meta Cloud API** — Set up a WhatsApp Business account and get a Phone Number ID.
2. **Webhook** — Create a FastAPI or Flask endpoint (`/webhook`).
3. **Connectivity** — When a user messages your WhatsApp number, Meta sends a POST request to your webhook.
4. **Processing** — Pass the message to the LangGraph agent, receive the response, and use the Twilio/Meta API to reply.
5. **Session ID** — Use the user's WhatsApp phone number as the `thread_id` in LangGraph for persistent per-user memory.

```python
# Example FastAPI webhook skeleton
from fastapi import FastAPI, Request
from src.agent import build_agent_graph

app = FastAPI()
agent = build_agent_graph()
sessions = {}  # phone_number -> state

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    phone = data["from"]
    message = data["text"]
    state = sessions.get(phone, {...default state...})
    state["messages"].append(HumanMessage(content=message))
    result = agent.invoke(state)
    sessions[phone] = result
    return {"reply": result["messages"][-1].content}
```

## Tech Stack

- **LangGraph** — State machine with cycles for multi-turn collection
- **LangChain Core** — Message primitives, tool decorators
- **Google Gemini 2.0 Flash** — Fast, capable LLM for classification + generation
- **Python 3.10+**

## License

MIT
