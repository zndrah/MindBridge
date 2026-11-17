# MindBridge 2.0 — Production-Quality Frontend

A safety-first, grounded mental health first-response companion and conversational workspace built with React, Vite, and Tailwind CSS. Seamlessly integrated with the MindBridge 2.0 FastAPI backend, RoBERTa crisis classifier, and FAISS dense knowledge retriever.

---

## 🌿 Overview & Product Experience

MindBridge 2.0 unifies six Stitch UI design specifications into ONE cohesive, responsive application flow:

$$\text{Landing Page} \longrightarrow \text{Chat Workspace} \longrightarrow \text{Contextual Safety Response}$$

### Visual Identity (MindBridge Sanctuary)
- **Palette**: Deep Sage (`#305343`), Resilient Sea-Foam (`#346761`), Grounded Earth/Terracotta (`#7e5d3c` / `#634627`), and Warm Ivory Canvas (`#f5fbf6`).
- **Typography**: Newsreader (editorial serif headings) paired with Plus Jakarta Sans (body, controls, and conversational dialogue).
- **Aesthetic**: Low cognitive load, non-clinical, calming, and organic. Rejects cold clinical dashboards and generic ChatGPT clones.

---

## 🏗️ Architecture & Component Hierarchy

```
frontend/
├── src/
│   ├── app/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Navbar.jsx               # Top bar with jurisdiction selector and crisis trigger
│   │   │   ├── Footer.jsx               # Scope disclaimer and boundary notices
│   │   │   └── EmergencyHeaderBanner.jsx# Non-clinical first-response reminder
│   │   ├── chat/
│   │   │   ├── ChatWorkspace.jsx        # 12-column responsive layout
│   │   │   ├── ChatWindow.jsx           # Scrollable message stream & empty state
│   │   │   ├── Sidebar.jsx              # Session controls, tone picker & breathing widget
│   │   │   ├── MessageBubble.jsx        # User and assistant bubbles with citations
│   │   │   ├── ChatInput.jsx            # Textarea with 1000 char counter & Enter-to-send
│   │   │   ├── SuggestedPrompts.jsx     # Quick emotional entry points
│   │   │   └── TypingIndicator.jsx      # Pulsing sage dots
│   │   ├── safety/
│   │   │   ├── SafetyResponseView.jsx   # Safe Harbor crisis triage view
│   │   │   ├── EmergencyResourceCard.jsx# Verified hotlines with direct tel/sms
│   │   │   └── SafetyBoundaryNotice.jsx # Mandatory boundary statements
│   │   ├── citations/
│   │   │   └── CitationCard.jsx         # Expandable knowledge provenance pill
│   │   ├── grounding/
│   │   │   ├── BreathingOrb.jsx         # SVG animated breathing pacer (4-7-8 & Box)
│   │   │   ├── SensoryGroundingModal.jsx# Interactive 5-4-3-2-1 somatic grounding tool
│   │   │   └── AudioSanctuaryModal.jsx  # Procedural Web Audio API soundscape generator
│   │   └── common/
│   │       └── MarkdownView.jsx         # Safe inline formatting & alert blockquotes
│   ├── pages/
│   │   ├── LandingPage.jsx              # Public landing page with ethics pledge
│   │   ├── ChatPage.jsx                 # Conversational portal + Safe Harbor view
│   │   └── SafetyProtocolsPage.jsx      # Standalone protocols & hotline directory
│   ├── services/
│   │   └── api.js                       # Exact FastAPI contract client
│   ├── hooks/
│   │   ├── useBreathing.js              # 4-7-8 & Box breathing state machine
│   │   └── useAmbientAudio.js           # Procedural Web Audio API sound generator
│   ├── context/
│   │   └── ChatContext.jsx              # In-memory ephemeral session state
│   ├── styles/
│   │   └── index.css                    # Tailwind directives & Sanctuary tokens
│   ├── App.jsx
│   └── main.jsx
├── index.html
├── tailwind.config.js
├── vite.config.js
├── package.json
└── .env.example
```

---

## ⚡ Integration with FastAPI Backend

The frontend interfaces directly with the MindBridge 2.0 FastAPI backend:

1. **`GET /health`**: Liveness probe.
2. **`GET /ready`**: Readiness probe checking that the RoBERTa model, BGE embeddings, FAISS vector store, and LangGraph workflow are loaded.
3. **`POST /chat`**: Strict payload boundary sending:
   ```json
   {
     "message": "User query string (1-10000 chars)",
     "session_id": "alphanumeric-session-id",
     "jurisdiction": "US | IN | UK | CA | AU | GL",
     "conversation_history": [
       { "role": "user", "content": "..." },
       { "role": "assistant", "content": "..." }
     ]
   }
   ```
   *Strict Boundary*: Extra fields are prohibited (`extra="forbid"`), and sessions are renewed cleanly upon clicking "+ New Reflection".

---

## 🔒 Safety, Privacy & Clinical Boundaries

- **Zero Persistence in LocalStorage**: User utterances, emotional disclosures, and mental health messages are held strictly in ephemeral volatile memory (React state). Clearing history or reloading discards all traces.
- **Backend as Source of Truth**: Clinical risk and crisis routing are never evaluated in the frontend; all safety triage decisions originate strictly from the backend deterministic rules and fine-tuned RoBERTa classifier.
- **Verified Hotlines Only**: Hotlines (988 in US/CA, 112/14416 in India, 999/111 in UK) are compiled from authoritative statutory directories. No hotline numbers, URLs, or dispatch capabilities are fabricated.
- **Genuine Functionality**: Grounding tools feature an interactive 5-4-3-2-1 sensory reset and a procedural Web Audio API ambient synthesizer generating real rain and wave noise (no fake play buttons).

---

## 🚀 Running Locally

### 1. Start Backend
```bash
# In project root
PYTHONPATH=. uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start Frontend
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Visit `http://localhost:5173` in your browser.

### 3. Run Tests
```bash
# Frontend Unit Tests
cd frontend
npm test

# Backend Test Suite
PYTHONPATH=. uv run pytest tests/test_api.py
```
