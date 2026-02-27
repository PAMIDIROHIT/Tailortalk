"""
Titanic Dataset Chat Agent — Streamlit Frontend
================================================
Clean, professional chat UI.
All queries (sidebar buttons + direct input) flow through the same
_handle_query() function to avoid any double-render or ghost-message issues.
"""

import os
import io
import requests
import streamlit as st

# ── Page config — MUST be the very first Streamlit call ──────────────────
st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Titanic Dataset Chat Agent — LangChain + Gemini + FastAPI + Streamlit",
    },
)

# ── Inject CSS ────────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
try:
    with open(_CSS_PATH) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Config ────────────────────────────────────────────────────────────────
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT: int = 180

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # list[dict] {role, content, image_url}
if "trigger" not in st.session_state:
    st.session_state.trigger = None     # str | None — query from sidebar button

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _post_query(question: str) -> tuple[str, str | None]:
    """Send question to FastAPI backend. Returns (text, image_url_or_None)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"message": question},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response", ""), data.get("image_url")
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return f"⚠️ Backend error {r.status_code}: {detail}", None
    except requests.exceptions.ConnectionError:
        return (
            f"❌ Cannot connect to the backend at `{BACKEND_URL}`.\n\n"
            "Make sure the FastAPI server is running and try again.",
            None,
        )
    except requests.exceptions.Timeout:
        return (
            "⏱️ Request timed out. The analysis may be complex — try again or simplify the question.",
            None,
        )
    except Exception as exc:
        return f"❌ Unexpected error: {exc}", None


def _fetch_image(path: str) -> bytes | None:
    """Fetch image bytes from backend /static/... endpoint."""
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=20)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def _render_message(role: str, content: str, image_url: str | None = None) -> None:
    """Render one chat bubble (user or assistant)."""
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if content:
            st.markdown(content)
        if image_url:
            img = _fetch_image(image_url)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.warning("Could not load visualisation image.", icon="⚠️")


def _handle_query(question: str) -> None:
    """Append user message, call backend, append assistant reply — no rerun."""
    question = question.strip()
    if not question:
        return

    # 1. Store + render user bubble
    st.session_state.messages.append({"role": "user", "content": question, "image_url": None})
    _render_message("user", question)

    # 2. Call backend and render assistant bubble
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analysing Titanic data…"):
            response_text, image_url = _post_query(question)

        if response_text:
            st.markdown(response_text)
        if image_url:
            img = _fetch_image(image_url)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.warning("Could not load visualisation image.", icon="⚠️")

    # 3. Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "image_url": image_url}
    )


# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown(
        """
        <div class="sb-head">
            <div class="sb-headtitle">🚢 Titanic Chat Agent</div>
            <div class="sb-headsub">LangChain · Gemini · FastAPI · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Backend status
    try:
        _ping = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if _ping.status_code == 200:
            st.markdown('<div class="status-ok">● Backend online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warn">⚠ Backend degraded</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-err">✕ Backend offline</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Dataset reference
    with st.expander("📋 Dataset Reference", expanded=False):
        st.markdown(
            """
| Column | Type | Description |
|--------|------|-------------|
| `Survived` | int | 0 = No, 1 = Yes |
| `Pclass` | int | 1 / 2 / 3 |
| `Sex` | str | male / female |
| `Age` | float | years (NaN) |
| `SibSp` | int | siblings/spouses |
| `Parch` | int | parents/children |
| `Fare` | float | ticket price £ |
| `Embarked` | str | C / Q / S |

**891 rows · 12 columns**
"""
        )

    st.markdown("---")

    # Suggestion buttons grouped by category
    _GROUPS: dict[str, list[str]] = {
        "📊 Statistics": [
            "What percentage of passengers survived?",
            "What percentage of passengers were male?",
            "What was the average ticket fare?",
            "What is the median age of passengers?",
            "How many passengers embarked from each port?",
            "What was the average fare by passenger class?",
        ],
        "📈 Visualizations": [
            "Show me a histogram of passenger ages",
            "Create a bar chart of survival rate by passenger class",
            "Plot fare distribution by class using a box plot",
            "Draw a pie chart of passenger embarkation ports",
            "Show a heatmap of feature correlations",
            "Plot the age distribution of survivors vs non-survivors",
        ],
        "🔍 Survival Analysis": [
            "What is the survival rate by gender?",
            "What percentage of passengers survived in each class?",
            "How did age affect survival? Compare survivors vs non-survivors",
            "What was the survival rate by embarkation port?",
            "What is the survival rate for children under 16?",
        ],
        "💳 Fare & Class": [
            "What was the highest fare paid and by whom?",
            "What percentage of passengers were in 1st class?",
            "Were higher fares correlated with survival?",
        ],
    }

    st.markdown("**💡 Suggested Questions**")
    for _cat, _qs in _GROUPS.items():
        with st.expander(_cat, expanded=False):
            for _q in _qs:
                if st.button(_q, key=f"sb__{_q}", use_container_width=True):
                    st.session_state.trigger = _q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.trigger   = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────

# Page header
st.markdown(
    """
    <div class="chat-header">
        <h1>🚢 Titanic Dataset Chat Agent</h1>
        <p>Ask anything in plain English — get instant data insights and beautiful charts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Welcome card (only shown before first message)
if not st.session_state.messages:
    st.markdown(
        """
<div class="welcome">
<h3>👋 What would you like to know?</h3>
<div class="welcome-grid">
  <div class="wcard">
    <span class="wcard-icon">🔢</span>
    <div><b>Statistics</b><br><small>"What % of passengers were male?"</small></div>
  </div>
  <div class="wcard">
    <span class="wcard-icon">📊</span>
    <div><b>Charts</b><br><small>"Show a histogram of ages"</small></div>
  </div>
  <div class="wcard">
    <span class="wcard-icon">🔍</span>
    <div><b>Survival</b><br><small>"What is the survival rate by gender?"</small></div>
  </div>
  <div class="wcard">
    <span class="wcard-icon">💡</span>
    <div><b>Insights</b><br><small>"Were higher fares correlated with survival?"</small></div>
  </div>
</div>
<p class="welcome-hint">👈 Click a suggested question from the sidebar or type below.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history
for _msg in st.session_state.messages:
    _render_message(_msg["role"], _msg.get("content", ""), _msg.get("image_url"))

# ── Process a sidebar suggestion (trigger set by button callback) ─────────
if st.session_state.trigger:
    _q = st.session_state.trigger
    st.session_state.trigger = None
    _handle_query(_q)

# ── Primary text input ────────────────────────────────────────────────────
if _user_input := st.chat_input("Ask anything about the Titanic dataset…"):
    _handle_query(_user_input)
