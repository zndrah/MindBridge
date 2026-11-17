# MindBridge 2.0 — Grounded Mental Health First-Response Companion

> **Safety-first, AI-powered first-response chatbot for emotional support.**  
> Not a replacement for therapy — a grounded first-responder companion.

---

## What is MindBridge?

MindBridge 2.0 is a mental health support chatbot that:

- Listens to users experiencing emotional distress, anxiety, or stress
- Detects crisis signals using a multi-layer AI safety pipeline
- Responds with **verified, citation-backed** information — not hallucinated advice
- Displays **real emergency hotlines** (India, US, UK, Canada, Australia) when needed
- **Never** diagnoses, prescribes, or replaces professional care

---

## Architecture

```
User (Browser)
     │
     ▼
┌──────────────────────────────────────┐
│   Frontend  │  React 18 + Vite 6    │  :5173
│             │  Tailwind CSS          │
└──────────────────┬───────────────────┘
                   │ REST API
                   ▼
┌──────────────────────────────────────┐
│   Backend   │  Python + FastAPI      │  :8000
│             │  LangGraph Pipeline    │
└──────┬───────────────┬───────────────┘
       │               │
  RoBERTa          BGE Embeddings
  Crisis Detector  + FAISS Index
  (CPU, local)     (Knowledge Base)
```

---

## AI Pipeline (7 Steps)

```
User Message
  ↓
1. INPUT NODE         — PII sanitization, raw text discarded immediately
  ↓
2. DETERMINISTIC      — Regex patterns + negation detection for emergency keywords
   SAFETY NODE
  ↓
3. ROBERTA CLASSIFIER — Fine-tuned RoBERTa-base; crisis probability threshold 0.5098
  ↓
4. SAFETY ROUTER      — Routes to: emergency_bypass / follow_up / deescalation / standard
  ↓
5. PLANNING NODE      — Chooses dialog goal: grounding / reframing / screening / general
  ↓
6. RAG RETRIEVAL      — BGE embeddings + FAISS search; cosine similarity ≥ 0.65
  ↓
7. RESPONSE GEN       — LLM generation → Verification Gate → Final response
  ↓
Verified Response sent to User
```

---

## Tech Stack

### Backend
| Library | Purpose |
|---|---|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **LangGraph** | AI state machine orchestration |
| **PyTorch (CPU)** | Local neural network inference |
| **HuggingFace Transformers** | RoBERTa & BGE model loading |
| **FAISS** | Vector similarity search |
| **Pydantic v2** | Input validation & schemas |
| **Google Gemini / Groq** | Optional LLM providers |

### Frontend
| Library | Purpose |
|---|---|
| **React 18** | UI components |
| **Vite 6** | Build tool & dev server |
| **Tailwind CSS** | Styling |
| **Lucide React** | Icons |

---

## AI Models

| Model | Purpose | Size |
|---|---|---|
| **RoBERTa-base** (fine-tuned) | Crisis / Non-Crisis binary classifier | ~500 MB |
| **BAAI/bge-small-en-v1.5** | Sentence embeddings for RAG retrieval | ~130 MB |

> **Note:** Model weights are not included in this repository (too large for GitHub).  
> Download and place them in `models/crisis_detector/roberta/` and `models/embeddings/bge-small-en-v1.5/`.

---

## Safety States

| State | Description |
|---|---|
| `emergency_bypass` | Immediate danger — bypasses AI, shows hotlines only |
| `pending_safety_follow_up` | Multi-turn PHQ-9 risk assessment in progress |
| `supportive_deescalation` | Passive ideation, no immediate danger |
| `standard_conversation` | Normal informational conversation |
| `elevated_classifier_signal` | RoBERTa flagged, no explicit keywords |
| `ml_risk_unassessed` | Model failed — safe deterministic fallback active |

---

## Project Structure

```
MindBridge/
├── backend/
│   ├── api/
│   │   ├── app.py              # FastAPI application factory
│   │   ├── routes.py           # API endpoints
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── middleware.py       # Rate limiting, body size, logging
│   │   └── dependencies.py    # Singleton dependency injection
│   ├── orchestrator/
│   │   ├── graph.py            # LangGraph state machine
│   │   ├── state.py            # Shared state schema (TypedDict)
│   │   ├── input_node.py       # PII sanitization + regex patterns
│   │   ├── safety_node.py      # Crisis detection & routing logic
│   │   ├── planning_node.py    # Dialog goal planning
│   │   ├── response_node.py    # LLM response generation
│   │   └── verification_node.py # Safety verification gate
│   ├── retrieval/
│   │   ├── retriever.py        # RAG engine
│   │   ├── embeddings.py       # BGE embedding model
│   │   ├── vector_store.py     # FAISS vector database
│   │   ├── ingestion.py        # Document ingestion
│   │   └── governance.py      # Document eligibility rules
│   ├── safety/
│   │   ├── crisis_detector.py  # RoBERTa inference wrapper
│   │   └── hotlines.json       # Emergency contacts (IN/US/UK/CA/AU)
│   └── config.py               # Settings from .env
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Root component
│   │   ├── pages/              # LandingPage, ChatPage, SafetyProtocolsPage
│   │   ├── components/         # Chat, layout, grounding, safety UI
│   │   ├── context/            # ChatContext (React state)
│   │   └── services/api.js     # Fetch calls to backend
│   └── vite.config.js          # Vite + API proxy config
├── data/
│   └── knowledge_base/         # 31 verified markdown knowledge documents
├── models/                     # (not in repo — download separately)
│   ├── crisis_detector/roberta/
│   └── embeddings/bge-small-en-v1.5/
├── scripts/
│   ├── start_backend.bat       # Windows: start backend
│   ├── start_frontend.bat      # Windows: start frontend
│   ├── launch_backend.ps1      # PowerShell launcher (handles path spaces)
│   └── launch_frontend.ps1     # PowerShell launcher
├── .env.example                # Configuration template
├── requirements-windows.txt    # Python dependencies (Windows CPU)
├── run.bat                     # Windows: start both services
├── stop.bat                    # Windows: stop all services
└── FIRST_TIME_SETUP.bat        # Windows: first-time install
```

---

## Running Locally (Windows)

### Prerequisites
- Python 3.12
- Node.js 18+
- npm

### First-Time Setup
```bat
FIRST_TIME_SETUP.bat
```

### Start the App
```bat
run.bat
```
Then open **http://localhost:5173** in your browser.

### Stop the App
```bat
stop.bat
```

---

## Manual Terminal Setup

**Terminal 1 — Backend:**
```bash
cd "MindBridge"
.\.venv\Scripts\python.exe -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000
```
*Wait ~30 seconds for models to load.*

**Terminal 2 — Frontend:**
```bash
cd "MindBridge\frontend"
npm run dev
```

Open **http://localhost:5173**

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (models loaded?) |
| `/chat` | POST | Send message, receive AI response |
| `/safety/resources` | GET | Emergency hotlines by jurisdiction |
| `/docs` | GET | Swagger interactive API docs |

### Chat Request
```json
{
  "message": "I have been feeling very anxious lately",
  "session_id": "optional-uuid",
  "jurisdiction": "IN"
}
```

### Chat Response
```json
{
  "response": "Here are some grounding techniques...",
  "safety_route": "standard_conversation",
  "verification_status": "verified",
  "grounding_status": "grounded",
  "citations": [{ "title": "...", "document_id": "..." }],
  "emergency_resources": []
}
```

---

## Configuration

Copy `.env.example` to `.env` and set:

```env
# Optional: Gemini API for better responses
GEMINI_API_KEY=your_key_here

# Optional: Groq API (fallback LLM)
GROQ_API_KEY=your_key_here

# LLM provider: auto | gemini | groq | local
LLM_PROVIDER=auto

# Default jurisdiction for hotlines
DEFAULT_JURISDICTION=IN
```

Without API keys, the system works fully offline using local deterministic fallbacks.

---

## Privacy & Safety Design

- **No data storage** — sessions are entirely in-memory (ephemeral)
- **No user accounts** — fully anonymous
- **PII sanitization** — raw user messages discarded at input boundary
- **Rate limiting** — 60 requests/minute per IP
- **Verification gate** — every AI response checked before delivery
- **Hardcoded hotlines** — emergency contacts never AI-generated
- **No hallucination** — responses grounded in verified knowledge base only

---

## Jurisdiction Support

Emergency hotlines are available for: **India (IN), United States (US), United Kingdom (UK), Canada (CA), Australia (AU), Global (GL)**

---

## License

This project is for educational and research purposes.  
MindBridge is **not a licensed medical device or clinical tool.**  
Always refer users in crisis to qualified mental health professionals.

---

*MindBridge 2.0 — v0.2.0 | Python 3.12 | FastAPI | React 18 | PyTorch CPU*

