"""
Titanic Chat Agent — Backend Service
=====================================
Strategy: ask Groq (LLaMA-3.3-70b) to generate Python code, execute it with
exec(), capture stdout as the text answer and check STATIC_DIR for saved plots.

Groq free tier: 14,400 requests/day — no daily quota issues.
"""

import io
import os
import re
import uuid
import logging
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")          # NON-INTERACTIVE — must be before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Absolute paths  — correct regardless of process cwd (local vs Docker)
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "titanic.csv"))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset — read once at import time (read-only, shared across requests)
# ---------------------------------------------------------------------------
_df: pd.DataFrame = pd.read_csv(DATA_PATH)
logger.info("Titanic dataset loaded: %d rows x %d columns", *_df.shape)

# ---------------------------------------------------------------------------
# LLM singleton — created on the first request so startup never blocks
# ---------------------------------------------------------------------------
_llm = None

# ---------------------------------------------------------------------------
# Model cascade — tried in order if a model hits rate limits
# ---------------------------------------------------------------------------
_MODEL_CASCADE = [
    "llama-3.3-70b-versatile",   # primary   — best code gen, 128k ctx
    "llama3-70b-8192",           # fallback 1 — reliable, 8k ctx
    "mixtral-8x7b-32768",        # fallback 2 — 32k ctx
    "llama3-8b-8192",            # fallback 3 — lightweight, fast
]

# Track which model index we're currently using
_model_index: int = 0

# Detect quota / rate-limit errors by checking the error message
_QUOTA_PHRASES = (
    "RESOURCE_EXHAUSTED",
    "quota",
    "rate limit",
    "429",
    "Too Many Requests",
)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(p.lower() in msg.lower() for p in _QUOTA_PHRASES)


def _get_llm() -> ChatGroq:
    global _llm, _model_index
    if _llm is None:
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to your .env file and restart the server."
            )
        model = settings.GROQ_MODEL
        _llm = ChatGroq(
            model=model,
            temperature=0,
            groq_api_key=settings.GROQ_API_KEY,
            max_retries=1,
        )
        logger.info("Groq LLM initialised (%s).", model)
    return _llm


def _try_next_model() -> ChatGroq | None:
    """Switch to the next model in the cascade after a rate-limit error."""
    global _llm, _model_index
    _model_index += 1
    if _model_index >= len(_MODEL_CASCADE):
        logger.warning("All models in cascade are rate-limited.")
        return None
    _llm = None
    model = _MODEL_CASCADE[_model_index]
    logger.info("Switching to fallback model: %s", model)
    _llm = ChatGroq(
        model=model,
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
        max_retries=1,
    )
    return _llm


# ---------------------------------------------------------------------------
# System prompt builder — PLOT_PATH is injected per-request
# ---------------------------------------------------------------------------
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
  df         — the Titanic DataFrame
  pd         — pandas
  np         — numpy
  plt        — matplotlib.pyplot (Agg backend)
  sns        — seaborn
  PLOT_PATH  — "{plot_path}" (absolute file path; save plots here)

RULES — follow exactly:

FOR STATISTICS / DATA QUESTIONS:
  1. Compute the answer using pandas/numpy on df.
  2. Use print() to output a single clear English sentence.
     Format numbers with f-strings. Use **bold** for key figures.
     Example: print(f"**{{pct:.1f}}%** of passengers were male.")
  3. For multi-row results print a neat markdown table, e.g.:
       | Port | Count |
       |------|-------|
       | S    | 644   |

FOR CHARTS / PLOTS / VISUALIZATIONS:
  1. Apply style: plt.style.use('seaborn-v0_8-whitegrid')
  2. Use sns.set_palette("husl")
  3. Create figure: fig, ax = plt.subplots(figsize=(10, 6))
     (use figsize=(7,7) for pie/square charts)
  4. Build the chart on ax (NOT plt directly).
     Typical patterns:
       Histogram:   ax.hist(df['Age'].dropna(), bins=25, color='#3B82F6', edgecolor='white')
       Bar chart:   data.plot(kind='bar', ax=ax, color='#3B82F6', edgecolor='white')
       Count plot:  sns.countplot(data=df, x='Pclass', hue='Survived', palette='husl', ax=ax)
       Box plot:    sns.boxplot(data=df, x='Pclass', y='Fare', palette='husl', ax=ax)
       Pie chart:   ax.pie(vals, labels=lbls, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('husl', len(vals)))
       Heatmap:     sns.heatmap(df.select_dtypes('number').corr(), annot=True, fmt='.2f', cmap='Blues', ax=ax)
       Scatter:     ax.scatter(df['Age'].dropna(), df.loc[df['Age'].notna(),'Fare'], alpha=0.5, c='#3B82F6')
  5. Add labels:
       ax.set_title('Clear Descriptive Title', fontsize=14, fontweight='bold', pad=10)
       ax.set_xlabel('X Label', fontsize=11)
       ax.set_ylabel('Y Label', fontsize=11)
       ax.tick_params(labelsize=9)
  6. For bar charts rotate x labels if needed: plt.xticks(rotation=0)
  7. plt.tight_layout()
  8. SAVE: plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
  9. plt.close('all')  # always free memory
  10. print() one sentence describing what the chart shows.

CRITICAL:
  - NEVER call plt.show() — it will hang the server.
  - Output ONLY valid Python code — no markdown fences, no explanations.
  - The code is executed with exec() so it must be self-contained.
"""


def _clean_code(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    raw = re.sub(r"^\s*```(?:python)?\s*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_query(query: str) -> dict:
    """
    Process *query* and return {"response": str, "image_url": str|None}.

    Flow:
      1. Ask Gemini to generate Python code for the query.
      2. exec() the code in a namespace that contains df, pd, plt, etc.
      3. Capture stdout as the text response.
      4. If a PNG was written to PLOT_PATH, expose it via /static/.
      5. On exec failure, retry once with the error appended to the prompt.
    """
    # ── Get LLM (with model cascade) ─────────────────────────────────────────
    try:
        llm = _get_llm()
    except ValueError as exc:
        return {"response": str(exc), "image_url": None}

    # ── Reserve a plot filename ───────────────────────────────────────────────
    plot_fname = f"plot_{uuid.uuid4().hex[:8]}.png"
    plot_path  = os.path.join(STATIC_DIR, plot_fname)

    # ── Ask Gemini for code (with model cascade on quota error) ──────────────
    system_msg = _build_system_prompt(plot_path)
    code = None
    for _attempt in range(len(_MODEL_CASCADE)):
        try:
            resp = llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=query),
            ])
            code = _clean_code(resp.content)
            break                                  # success — exit retry loop
        except Exception as exc:
            logger.warning("LLM request failed (model %s): %s",
                           getattr(llm, "model", "?"), exc)
            if _is_quota_error(exc):
                llm = _try_next_model()
                if llm is None:
                    return {
                        "response": (
                            "⚠️ All Groq models are currently rate-limited. "
                            "Please wait a minute and try again, or see "
                            "[Groq rate limits](https://console.groq.com/settings/limits)."
                        ),
                        "image_url": None,
                    }
                continue                           # retry with next model
            return {"response": f"Could not contact Gemini API: {exc}", "image_url": None}

    if code is None:
        return {"response": "No response generated. Please try again.", "image_url": None}

    logger.debug("Generated code:\n%s", code)

    # ── Shared execution namespace ───────────────────────────────────────────
    ns: dict = {
        "df": _df.copy(), "pd": pd, "np": np,
        "plt": plt, "sns": sns,
        "PLOT_PATH": plot_path, "os": os,
        "uuid": uuid,
    }

    # ── Execute (attempt 1) ──────────────────────────────────────────────────
    buf     = io.StringIO()
    err_msg = None
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<groq_code>", "exec"), ns)  # noqa: S102
    except Exception as exc:
        err_msg = str(exc)
        logger.warning("exec() error: %s\n--- Code ---\n%s", exc, code)

    text = buf.getvalue().strip()

    # ── Retry on exec failure only (not on API failure — preserves quota) ────
    if err_msg and not text:
        logger.info("Retrying with error context …")
        retry_msg = (
            f"The code you wrote raised this error:\n{err_msg}\n\n"
            f"Offending code:\n```python\n{code}\n```\n\n"
            "Rewrite it correctly. Output ONLY valid Python code."
        )
        try:
            resp2 = llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=query),
                HumanMessage(content=retry_msg),
            ])
            code2 = _clean_code(resp2.content)
            buf2  = io.StringIO()
            with redirect_stdout(buf2):
                exec(compile(code2, "<groq_retry>", "exec"), ns)  # noqa: S102
            text    = buf2.getvalue().strip()
            err_msg = None
        except Exception as exc2:
            if _is_quota_error(exc2):
                return {
                    "response": (
                        "⚠️ API quota reached for today. Please wait a moment and try again."
                    ),
                    "image_url": None,
                }
            logger.warning("Retry also failed: %s", exc2)
            err_msg = str(exc2)

    # ── Check for saved plot ─────────────────────────────────────────────────
    image_url: str | None = None
    if os.path.exists(plot_path) and os.path.getsize(plot_path) > 0:
        image_url = f"/static/{plot_fname}"
        logger.info("Plot written → %s", plot_path)
    elif os.path.exists(plot_path):
        os.remove(plot_path)  # remove 0-byte placeholder

    # ── Build final response ─────────────────────────────────────────────────
    if err_msg and not text:
        response = (
            f"I ran into an error while analysing your request: `{err_msg}`\n\n"
            "Please try rephrasing your question."
        )
    elif not text and not image_url:
        response = "The analysis ran but produced no printable output. Try rephrasing."
    elif not text and image_url:
        response = "Here is the visualisation:"
    else:
        response = text

    return {"response": response, "image_url": image_url}
