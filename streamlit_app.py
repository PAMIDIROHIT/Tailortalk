"""
Titanic Chat Agent — Standalone Streamlit App
==============================================
Self-contained version for Streamlit Community Cloud deployment.
No separate FastAPI backend required — agent logic runs inside Streamlit.

Deploy:  streamlit run streamlit_app.py
"""

import io
import os
import re
import uuid
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "frontend", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Dataset ─────────────────────────────────────────────────────────────────
_DATA_PATH = os.path.join(os.path.dirname(__file__), "backend", "data", "titanic.csv")


@st.cache_data
def _load_data() -> pd.DataFrame:
    return pd.read_csv(_DATA_PATH)


_df = _load_data()

# ── Temp dir for plots ───────────────────────────────────────────────────────
_PLOT_DIR = os.path.join(os.path.dirname(__file__), "backend", "static")
os.makedirs(_PLOT_DIR, exist_ok=True)

# ── Groq API key — reads from st.secrets or env ─────────────────────────────
def _get_api_key() -> str:
    # Streamlit Cloud stores secrets in st.secrets
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


_MODEL_CASCADE = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
]
_QUOTA_PHRASES = ("RESOURCE_EXHAUSTED", "quota", "rate limit", "429", "Too Many Requests", "rate_limit")


def _is_quota_error(exc: Exception) -> bool:
    return any(p.lower() in str(exc).lower() for p in _QUOTA_PHRASES)


def _build_system_prompt(plot_path: str) -> str:
    return f"""You are an expert Python data analyst working with the Titanic passenger dataset.

DATAFRAME `df` — 891 rows, 12 columns:
  PassengerId  int    unique id 1-891
  Survived     int    0=died, 1=survived
  Pclass       int    ticket class: 1=First, 2=Second, 3=Third
  Name         str    full name
  Sex          str    'male' or 'female'
  Age          float  age in years (177 NaN values)
  SibSp        int    num siblings/spouses aboard
  Parch        int    num parents/children aboard
  Ticket       str    ticket number
  Fare         float  fare in GBP
  Cabin        str    cabin number (687 NaN values)
  Embarked     str    port: C=Cherbourg Q=Queenstown S=Southampton

EXECUTION NAMESPACE (already available — do NOT import or redefine):
  df, pd, np, plt, sns, PLOT_PATH="{plot_path}"

RULES:
FOR STATISTICS / DATA QUESTIONS:
  1. Compute the answer using pandas/numpy on df.
  2. print() a single clear sentence. Use **bold** for key figures.
  3. For multi-row results print a markdown table.

FOR CHARTS / PLOTS / VISUALIZATIONS:
  1. plt.style.use('seaborn-v0_8-whitegrid'); sns.set_palette("husl")
  2. fig, ax = plt.subplots(figsize=(10, 6))
  3. Build chart on ax. Add descriptive title, xlabel, ylabel.
  4. plt.tight_layout()
  5. plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
  6. plt.close('all')
  7. print() one sentence describing what the chart shows.

CRITICAL:
  - NEVER call plt.show()
  - Output ONLY valid Python code — no markdown fences, no explanations.
"""


def _clean_code(raw: str) -> str:
    raw = re.sub(r"^\s*```(?:python)?\s*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def process_query(query: str) -> tuple[str, bytes | None]:
    """Returns (text_response, image_bytes_or_None)."""
    api_key = _get_api_key()
    if not api_key:
        return (
            "⚠️ GROQ_API_KEY is not configured.\n\n"
            "**For local use:** Add `GROQ_API_KEY=your_key` to `.env`\n\n"
            "**For Streamlit Cloud:** Add it in App Settings → Secrets as:\n"
            "```toml\nGROQ_API_KEY = \"your_key_here\"\n```",
            None,
        )

    plot_fname = f"plot_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(_PLOT_DIR, plot_fname)
    system_msg = _build_system_prompt(plot_path)

    # Try each model in the cascade
    code = None
    for model in _MODEL_CASCADE:
        try:
            llm = ChatGroq(
                model=model,
                temperature=0,
                groq_api_key=api_key,
                max_retries=1,
            )
            resp = llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=query),
            ])
            code = _clean_code(resp.content)
            break
        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning("Model %s rate-limited, trying next…", model)
                continue
            return (f"Error calling Groq API: {exc}", None)

    if code is None:
        return (
            "⚠️ All Groq models are currently rate-limited. "
            "Please wait a minute and try again.",
            None,
        )

    # Execute generated code
    ns: dict = {"df": _df.copy(), "pd": pd, "np": np,
                "plt": plt, "sns": sns, "PLOT_PATH": plot_path, "os": os}
    buf = io.StringIO()
    err_msg = None
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<groq_code>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        err_msg = str(exc)
        logger.warning("exec() error: %s", exc)

    text = buf.getvalue().strip()

    # Read plot if saved
    img_bytes: bytes | None = None
    if os.path.exists(plot_path) and os.path.getsize(plot_path) > 0:
        with open(plot_path, "rb") as f:
            img_bytes = f.read()
        os.remove(plot_path)  # clean up temp file
    elif os.path.exists(plot_path):
        os.remove(plot_path)

    if err_msg and not text and not img_bytes:
        return (f"Analysis error: `{err_msg}`\n\nTry rephrasing your question.", None)
    if not text and img_bytes:
        text = "Here is the visualisation:"
    if not text and not img_bytes:
        text = "Analysis ran but produced no output. Try rephrasing."

    return (text, img_bytes)


# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trigger" not in st.session_state:
    st.session_state.trigger = None


def _handle_query(question: str):
    if not question.strip():
        return
    st.session_state.messages.append({"role": "user", "content": question, "image": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analysing…"):
            text, img = process_query(question)
        st.markdown(text)
        if img:
            st.image(img, use_container_width=True)
    st.session_state.messages.append({"role": "assistant", "content": text, "image": img})


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-head">'
        '<div class="sb-headtitle">🚢 Titanic Agent</div>'
        '<div class="sb-headsub">LLaMA 3.3 · Groq · Streamlit</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**📋 Dataset**")
    with st.expander("titanic.csv — 891 rows"):
        st.dataframe(_df.head(8), use_container_width=True)

    st.markdown("---")
    st.markdown("**💡 Try these questions**")

    SUGGESTIONS = [
        ("📊 Stats", [
            "What percentage of passengers were male?",
            "What was the average ticket fare?",
            "How many passengers survived?",
            "What was the average age of passengers?",
        ]),
        ("📈 Charts", [
            "Show me a histogram of passenger ages",
            "How many passengers embarked from each port?",
            "Show survival rate by passenger class",
            "Plot fare distribution by class",
        ]),
    ]

    for cat, questions in SUGGESTIONS:
        st.markdown(f"**{cat}**")
        for q in questions:
            if st.button(q, key=f"btn_{q[:30]}", use_container_width=True):
                st.session_state.trigger = q

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trigger = None
        st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="chat-header">'
    "<h1>🚢 Titanic Passenger Analysis</h1>"
    "<p>Ask any question about the Titanic dataset — statistics, charts, and insights</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Render history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], use_container_width=True)

# Process sidebar button trigger
if st.session_state.trigger:
    q = st.session_state.trigger
    st.session_state.trigger = None
    _handle_query(q)
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about the Titanic dataset…"):
    _handle_query(prompt)
