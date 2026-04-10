# AutoStream AI Agent

An intelligent conversational sales agent built with **LangGraph** and **Google Gemini** for AutoStream — an AI-powered video editing platform for content creators.

## What It Does

| Capability | Description |
|---|---|
| **RAG-Powered Q&A** | Answers pricing, policy, and feature questions from a structured knowledge base |
| **Intent Classification** | Detects the 3 required intents: Casual greeting, Product/Pricing inquiry, High-intent lead |
| **Multi-Turn Lead Capture** | Collects Name → Email → Platform one at a time before firing the CRM tool |
| **Tool Calling** | Triggers `mock_lead_capture` only when all three fields are present |
| **Persistent Memory** | Maintains conversation state across turns within a session |

## Architecture Explanation (~200 words)

This project uses **LangGraph** because the required flow is stateful and step-based, not a single prompt/response. LangGraph gives explicit control over conversation routing: one node classifies intent, another handles RAG responses, another collects lead fields, and a final node calls the tool only when all required values are present. That graph structure makes the behavior predictable, easy to debug, and aligned with the evaluation criteria around tool safety and reasoning.

For state management, the shared `AgentState` stores chat history plus `user_name`, `user_email`, `user_platform`, `intent`, and `lead_captured`. Because this state is carried forward each turn, the agent can remember partial lead details across 5-6 messages and continue collection without asking for already provided information. The lead flow is "sticky": once signup intent is detected, the graph remains in lead-collection mode until all three fields are collected or the conversation changes.

RAG uses a local JSON knowledge base in `data/knowledge_base.json` (as required). The retriever is intentionally lightweight because the KB is small and structured. This keeps responses grounded in known pricing and policy facts while avoiding over-engineering. Overall, this design balances correctness, maintainability, and real-world extensibility.

## Architecture Diagram

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

## WhatsApp Deployment via Webhooks

To integrate this agent with WhatsApp in production, use inbound and outbound webhooks:

1. **Provider setup**: Configure WhatsApp Business via Meta Cloud API directly, or through Twilio WhatsApp.
2. **Inbound webhook**: Expose a backend endpoint (FastAPI/Flask) that receives WhatsApp message events.
3. **Identity to session mapping**: Use the sender phone number as the conversation key so each user keeps independent memory/state.
4. **Agent invocation**: Convert inbound text into a `HumanMessage`, invoke the LangGraph app with stored state, and persist updated state.
5. **Outbound reply**: Send the latest AI response back using the provider send-message endpoint.
6. **Security and reliability**: Validate webhook signatures, add retry-safe idempotency keys, and log tool calls for auditing.

This creates a full request/response loop where every WhatsApp user has persistent multi-turn memory and safe lead capture.

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
- **Google Gemini 1.5 Flash** (default) with optional fallbacks via environment configuration
- **Python 3.10+**

## Demo Video Checklist (2-3 minutes)

1. Ask a pricing question and show grounded RAG answer from local KB.
2. Send a high-intent message (for example, "I want to try the Pro plan for my YouTube channel").
3. Show lead qualification prompts collecting missing Name and Email.
4. Provide all details and show successful `mock_lead_capture` execution.

## License

MIT
