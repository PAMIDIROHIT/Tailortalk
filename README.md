# 🚢 Titanic Dataset Chat Agent

<div align="center">

**Author: PAMIDI ROHIT**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.133-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036)

*A full-stack AI chatbot that answers natural-language questions about the Titanic passenger dataset and generates data visualisations on demand.*

</div>

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Architecture Diagram](#system-architecture-diagram)
- [Sequence Diagram](#sequence-diagram)
- [Agent Processing Flowchart](#agent-processing-flowchart)
- [UML Class Diagram](#uml-class-diagram)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Example Questions](#example-questions)
- [Deployment](#deployment)

---

## Architecture Overview

The system follows a **3-tier architecture**:

| Tier | Component | Technology |
|------|-----------|------------|
| **Presentation** | Chat UI with sidebar, suggestion buttons, chart rendering | Streamlit 1.49 |
| **Application** | REST API, LLM orchestration, code execution | FastAPI + LangChain + Groq |
| **Data** | Titanic CSV (891 rows × 12 cols), Generated PNG charts | pandas + matplotlib |

---

## System Architecture Diagram

```mermaid
graph TB
    subgraph Client["🖥️ Client Browser"]
        UI[Streamlit Chat UI<br/>localhost:8501]
    end

    subgraph Frontend["📱 Frontend Layer (Streamlit)"]
        APP[app.py<br/>Chat Interface]
        CSS[style.css<br/>Premium Theme]
        SB[Sidebar<br/>Suggestions & Status]
    end

    subgraph Backend["⚙️ Backend Layer (FastAPI)"]
        API[POST /api/chat<br/>endpoints.py]
        AGENT[Agent Service<br/>agent.py]
        STATIC[StaticFiles<br/>GET /static/*.png]
        CFG[Config<br/>config.py]
        SCHEMA[Schemas<br/>ChatRequest / ChatResponse]
    end

    subgraph LLM["🤖 LLM Layer (Groq Cloud)"]
        GROQ[Groq API<br/>LLaMA 3.3 70B]
        M2[llama3-70b-8192<br/>fallback 1]
        M3[mixtral-8x7b-32768<br/>fallback 2]
        M4[llama3-8b-8192<br/>fallback 3]
    end

    subgraph Data["📊 Data Layer"]
        CSV[(titanic.csv<br/>891 rows)]
        PNG[(backend/static/<br/>*.png charts)]
    end

    UI --> APP
    APP --> SB
    APP -->|HTTP POST /api/chat| API
    APP -->|HTTP GET /static/plot.png| STATIC
    CSS --> APP
    API --> SCHEMA
    API --> AGENT
    AGENT --> CFG
    AGENT -->|invoke LLM| GROQ
    GROQ -->|quota exceeded| M2
    M2 -->|quota exceeded| M3
    M3 -->|quota exceeded| M4
    AGENT -->|read| CSV
    AGENT -->|exec + savefig| PNG
    STATIC --> PNG

    style Client fill:#EFF6FF,stroke:#BFDBFE
    style Frontend fill:#F0FDF4,stroke:#BBF7D0
    style Backend fill:#FFF7ED,stroke:#FED7AA
    style LLM fill:#FDF4FF,stroke:#E9D5FF
    style Data fill:#F8FAFC,stroke:#E2E8F0
```

---

## Sequence Diagram

Shows the complete lifecycle of a user query — from input to rendered response.

```mermaid
sequenceDiagram
    actor User
    participant ST as Streamlit UI<br/>(app.py)
    participant API as FastAPI<br/>(endpoints.py)
    participant AGENT as Agent Service<br/>(agent.py)
    participant GROQ as Groq API<br/>(LLaMA 3.3 70B)
    participant FS as File System<br/>(static/*.png)

    User->>ST: Types question / clicks suggestion
    ST->>ST: Append user message to session_state
    ST->>ST: Render user chat bubble
    ST->>ST: Show spinner "Analysing…"

    ST->>+API: POST /api/chat<br/>{"message": "Show histogram of ages"}
    API->>API: Validate request (empty check)
    API->>+AGENT: process_query(query)

    AGENT->>AGENT: Reserve plot_<uuid>.png path
    AGENT->>AGENT: Build system prompt<br/>(injects PLOT_PATH, df schema)

    AGENT->>+GROQ: invoke([SystemMsg, HumanMsg])
    Note over GROQ: LLaMA 3.3 70B generates<br/>Python code for the query

    alt Quota exhausted
        GROQ-->>AGENT: 429 RESOURCE_EXHAUSTED
        AGENT->>GROQ: Retry with fallback model<br/>(llama3-70b-8192)
    end

    GROQ-->>-AGENT: Python code string

    AGENT->>AGENT: clean_code - strip markdown fences
    AGENT->>AGENT: exec code in namespace df pd np plt sns PLOT_PATH

    alt Visualisation query
        AGENT->>FS: plt.savefig(PLOT_PATH)
        FS-->>AGENT: PNG written (33–80 KB)
        AGENT->>AGENT: image_url = "/static/plot_<uuid>.png"
    end

    alt exec fails
        AGENT->>+GROQ: Retry with error context
        GROQ-->>-AGENT: Corrected Python code
        AGENT->>AGENT: exec corrected code in namespace
    end

    AGENT->>AGENT: Capture stdout as text response
    AGENT-->>-API: {"response": "...", "image_url": "/static/..."}
    API-->>-ST: HTTP 200 ChatResponse JSON

    ST->>ST: Render assistant chat bubble<br/>(markdown text)

    alt image_url present
        ST->>+API: GET /static/plot_<uuid>.png
        API->>FS: Read PNG bytes
        FS-->>API: PNG bytes
        API-->>-ST: image/png (HTTP 200)
        ST->>ST: st.image() — render chart
    end

    ST->>ST: Append assistant message to session_state
    ST-->>User: Final response visible
```

---

## Agent Processing Flowchart

Internal logic of `agent.py` — how a raw query becomes a structured response.

```mermaid
flowchart TD
    START([▶ process_query called]) --> GETLLM[Get Groq LLM singleton<br/>_get_llm]
    GETLLM --> APICHECK{GROQ_API_KEY<br/>configured?}
    APICHECK -- No --> ERRKEY[Return error message]
    APICHECK -- Yes --> PROMPT[Build system prompt<br/>Inject df schema + PLOT_PATH]

    PROMPT --> INVOKE[Invoke LLM<br/>SystemMsg + HumanMsg]

    INVOKE --> QUOTA{429 Quota<br/>Error?}
    QUOTA -- Yes --> CASCADE[Try next model<br/>in cascade]
    CASCADE --> ALLEXHAUSTED{All 4 models<br/>exhausted?}
    ALLEXHAUSTED -- Yes --> ERRQUOTA[Return quota error message]
    ALLEXHAUSTED -- No --> INVOKE

    QUOTA -- No --> APIERR{Other API<br/>Error?}
    APIERR -- Yes --> ERRAPI[Return API error message]
    APIERR -- No --> CLEAN[_clean_code<br/>Strip markdown fences]

    CLEAN --> EXEC["exec(compiled_code, namespace)<br/>df, pd, np, plt, sns, PLOT_PATH"]

    EXEC --> EXECFAIL{exec<br/>raised exception?}

    EXECFAIL -- Yes --> RETRY[Build retry prompt<br/>with error + offending code]
    RETRY --> INVOKE2[Re-invoke LLM<br/>with error context]
    INVOKE2 --> EXEC2["exec(corrected_code, namespace)"]
    EXEC2 --> EXECFAIL2{Still<br/>failed?}
    EXECFAIL2 -- Yes --> ERREXEC[Return exec error message]
    EXECFAIL2 -- No --> STDOUT2[Capture stdout]

    EXECFAIL -- No --> STDOUT[Capture stdout as text]

    STDOUT --> PNGCHECK{plot_<uuid>.png<br/>exists & > 0 bytes?}
    STDOUT2 --> PNGCHECK

    PNGCHECK -- Yes --> IMGURL[image_url = /static/plot_uuid.png]
    PNGCHECK -- No --> NOIMG[image_url = None]

    IMGURL --> BUILD[Build response dict]
    NOIMG --> BUILD

    BUILD --> TEXTCHECK{text and<br/>image_url?}
    TEXTCHECK -- text only --> RESP1[response = text]
    TEXTCHECK -- image only --> RESP2[response = Here is the visualisation]
    TEXTCHECK -- both --> RESP3[response = text + image_url]
    TEXTCHECK -- neither --> RESP4[response = no output message]

    RESP1 --> RETURN([◀ Return dict])
    RESP2 --> RETURN
    RESP3 --> RETURN
    RESP4 --> RETURN
    ERRKEY --> RETURN
    ERRQUOTA --> RETURN
    ERRAPI --> RETURN
    ERREXEC --> RETURN
```

---

## UML Class Diagram

Module-level structure of the backend.

```mermaid
classDiagram
    class Settings {
        +str PROJECT_NAME
        +str GROQ_API_KEY
        +str GROQ_MODEL
        +validate_api_key() None
    }

    class ChatRequest {
        +str message
    }

    class ChatResponse {
        +str response
        +Optional~str~ image_url
    }

    class AgentService {
        -DataFrame _df
        -ChatGroq _llm
        -int _model_index
        -list _MODEL_CASCADE
        +process_query(query: str) dict
        -_get_llm() ChatGroq
        -_try_next_model() ChatGroq
        -_build_system_prompt(plot_path: str) str
        -_clean_code(raw: str) str
        -_is_quota_error(exc: Exception) bool
    }

    class ChatEndpoint {
        +chat_endpoint(request: ChatRequest) ChatResponse
    }

    class FastAPIApp {
        +lifespan(app) ContextManager
        +mount_static_files()
        +include_router(router)
    }

    class StreamlitUI {
        -list messages
        -str trigger
        +_post_query(question: str) tuple
        +_fetch_image(path: str) bytes
        +_render_message(role, content, image_url)
        +_handle_query(question: str)
    }

    Settings <.. AgentService : reads
    ChatRequest <.. ChatEndpoint : receives
    ChatEndpoint ..> ChatResponse : returns
    ChatEndpoint ..> AgentService : calls process_query
    FastAPIApp *-- ChatEndpoint : includes router
    StreamlitUI ..> ChatEndpoint : HTTP POST /api/chat
    AgentService --> ChatGroq : invokes
    AgentService --> DataFrame : reads titanic.csv
```

---

## Features

- **Natural language queries** — ask anything about the Titanic dataset in plain English
- **Statistical answers** — percentages, averages, counts, markdown tables
- **Visualisations** — histograms, bar charts, pie charts, box plots, scatter plots, heatmaps
- **Survival analysis** — by gender, class, age, embarkation port, family size
- **Sidebar suggestions** — one-click prompts organised by category
- **Chat history** — full conversation retained in Streamlit session state
- **Backend health badge** — sidebar shows ✅ / ❌ live backend status
- **Model cascade fallback** — switches through 4 Groq models if one hits rate limits

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.12 |
| Backend API | FastAPI + Uvicorn | 0.133 / 0.34 |
| LLM Framework | LangChain + langchain-groq | 1.2 / 0.3 |
| LLM Provider | Groq Cloud — LLaMA 3.3 70B | Free tier 14,400 req/day |
| Data Processing | pandas + numpy | 3.0 / 2.1 |
| Visualisation | matplotlib + seaborn | 3.10 / 0.13 |
| Frontend | Streamlit | 1.49 |
| Containerisation | Docker + Docker Compose | 3.8 |
| Deployment | Streamlit Community Cloud / Render | — |

---

## Getting Started

### Prerequisites

- Docker & Docker Compose **or** Python 3.10+
- Free Groq API key → [console.groq.com/keys](https://console.groq.com/keys)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/titanic-chat-agent.git
cd titanic-chat-agent
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 2a. Run with Docker Compose (recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Chat UI | http://localhost:8501 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| Backend Health Check | http://localhost:8000/health |

### 2b. Run Locally (without Docker)

**Backend:**
```bash
python -m venv backend/.venv
source backend/.venv/bin/activate        # Windows: backend\.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
pip install streamlit requests
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501
```

**Standalone** (no separate backend — for Streamlit Cloud):
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run streamlit_app.py
```

---

## Project Structure

```
TailorTalk-Ass/
│
├── streamlit_app.py          # Standalone app (Streamlit Cloud)
├── requirements.txt          # Root deps for Streamlit Cloud
│
├── backend/
│   ├── main.py               # FastAPI entry — lifespan, StaticFiles, router
│   ├── requirements.txt      # Backend-only dependencies
│   ├── api/
│   │   └── endpoints.py      # POST /api/chat  →  ChatResponse
│   ├── core/
│   │   └── config.py         # Pydantic settings (reads .env)
│   ├── data/
│   │   └── titanic.csv       # Dataset — 891 rows × 12 columns
│   ├── models/
│   │   └── schemas.py        # ChatRequest / ChatResponse Pydantic models
│   ├── services/
│   │   └── agent.py          # Groq code-gen agent (exec pipeline)
│   └── static/               # Runtime-generated PNG charts (gitignored)
│
├── frontend/
│   ├── app.py                # Streamlit UI — talks to FastAPI backend
│   ├── style.css             # Professional SaaS light-mode CSS
│   └── requirements.txt      # streamlit, requests
│
├── .streamlit/
│   └── config.toml           # Streamlit theme (primaryColor, fonts)
│
├── Dockerfile.backend        # Multi-stage FastAPI image
├── Dockerfile.frontend       # Streamlit image
├── docker-compose.yml        # One-command startup
├── render.yaml               # Render.com production deploy config
├── .env.example              # Environment variable template
└── .gitignore
```

---

## Example Questions

| # | Category | Question | Response Type |
|---|----------|---------|--------------|
| 1 | Statistics | "What percentage of passengers were male on the Titanic?" | Text |
| 2 | Statistics | "What was the average ticket fare?" | Text |
| 3 | Statistics | "How many passengers embarked from each port?" | Markdown table |
| 4 | Statistics | "What is the survival rate by gender?" | Text |
| 5 | Visualisation | "Show me a histogram of passenger ages" | Chart |
| 6 | Visualisation | "Plot survival rate by passenger class as a bar chart" | Chart |
| 7 | Visualisation | "Show a heatmap of feature correlations" | Chart |
| 8 | Visualisation | "Plot fare distribution by class using a box plot" | Chart |
| 9 | Survival | "How did age affect survival chances?" | Text / Chart |
| 10 | Survival | "What percentage of children (age < 16) survived?" | Text |

---

## Deployment — Streamlit Community Cloud

```mermaid
flowchart LR
    DEV[Local Development] -->|git push| GH[GitHub Repository]
    GH -->|auto-deploy on push| STC[Streamlit Community Cloud\nstreamlit_app.py]
    STC -->|reads| SEC[App Secrets\nGROQ_API_KEY]
    STC -->|public URL| USER[👤 HR / Evaluator]

    style DEV fill:#EFF6FF,stroke:#BFDBFE
    style GH fill:#F0FDF4,stroke:#BBF7D0
    style STC fill:#FF4B4B,color:#fff,stroke:#CC0000
    style SEC fill:#FFF7ED,stroke:#FED7AA
    style USER fill:#F8FAFC,stroke:#E2E8F0
```

**Steps:**
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repo: `YOUR_USERNAME/titanic-chat-agent` | Branch: `main` | File: `streamlit_app.py`
4. **Advanced → Secrets:**
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** — get a shareable `https://your-app.streamlit.app` URL

---

## Notes

- The agent asks Groq LLaMA 3.3 70B to write Python code, then runs it with `exec()` in a private namespace containing the Titanic DataFrame. This is more reliable than tool-calling agents because LLMs sometimes skip tool calls for visualisation queries.
- Groq free tier: **14,400 requests/day** — sufficient for development and demos.
- The LLM is **lazily initialised** on the first request so the server starts instantly.
- Generated PNGs are stored in `backend/static/` (ephemeral — gitignored, Docker-volume-mounted).

---

<div align="center">
<sub>Built by <strong>PAMIDI ROHIT</strong> · Powered by Groq · LLaMA 3.3 70B · Streamlit · FastAPI</sub>
</div>
