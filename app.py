"""
DocuSense AI — Agentic Document Intelligence System
====================================================
Orchestrator → OCR Cleaner → Classifier → Extractor
→ Anomaly Validator → Critic → AI Summarizer → Memory → Reporter
"""

import streamlit as st
import pdfplumber
import ollama
import docx
from PIL import Image
import pytesseract
import re
import json
import os
from datetime import datetime

# ── CrewAI integration (optional, non-breaking) ──────────
# Uses Ollama (local) as the LLM — NO OpenAI / API key needed.
# Install with: pip install crewai
# Requires Ollama running locally: ollama run phi
_CREWAI_AVAILABLE = False
_OLLAMA_LLM       = None          # set below if crewai loads
try:
    # ── Block OpenAI before crewai imports anything ──────
    import os as _os
    _os.environ.setdefault("OPENAI_API_KEY",   "NOT_NEEDED_LOCAL")
    _os.environ.setdefault("OPENAI_API_BASE",  "http://localhost:11434")  # Ollama
    _os.environ.setdefault("OPENAI_MODEL_NAME","ollama/phi")

    from crewai import Agent, Task, Crew, Process

    # ── Try to build an explicit Ollama LLM object ────────
    # crewai >= 0.30 ships with LLM wrapper; older versions
    # accept the 'ollama/phi' string directly on Agent.
    try:
        from crewai import LLM as _CrewLLM
        _OLLAMA_LLM = _CrewLLM(
            model    = "ollama/phi",
            base_url = "http://localhost:11434",
        )
    except Exception:
        # Fallback: pass the string form — works on most crewai builds
        _OLLAMA_LLM = "ollama/phi"

    _CREWAI_AVAILABLE = True
except ImportError:
    pass

def _build_crewai_crew(
    doc_text: str,
    doc_type: str,
    extracted_points: list,
    validation_flags: list,
):
    """
    Build and return a CrewAI Crew fed with REAL pipeline data.
    Returns None if CrewAI is not installed.
    The existing orchestrator() is NOT replaced — this is supplemental.

    Parameters
    ----------
    doc_text         : cleaned document text (from OCR cleaner)
    doc_type         : classified document type string
    extracted_points : list of (label, value) tuples from critic agent
    validation_flags : list of (severity, message) tuples from anomaly agent
    """
    if not _CREWAI_AVAILABLE:
        return None

    # ── Format real pipeline data as readable strings ─────────
    _NO_HALLUCINATION = (
        "IMPORTANT: Do NOT generate, invent, or assume any information "
        "that is not explicitly present in the Extracted Data or Document "
        "sections below. Use ONLY the provided data."
    )

    _structured_fields = "\n".join(
        f"  - {lbl}: {val}" for lbl, val in extracted_points
    ) or "  - No fields extracted"

    _validation_output = "\n".join(
        f"  - {sev}: {msg}" for sev, msg in validation_flags
    ) or "  - No validation issues"

    _ctx = doc_text[:500]   # raw text snippet for reference only

    # ── Define agents — all use local Ollama, no OpenAI ──────
    _crew_classifier = Agent(
        role="Document Classifier",
        goal="Confirm the document type using the already-classified label",
        backstory=(
            "Expert at recognising document categories. Works strictly "
            "from the provided classification — does not re-classify."
        ),
        llm=_OLLAMA_LLM,
        verbose=False,
        allow_delegation=False,
    )
    _crew_extractor = Agent(
        role="Information Extractor",
        goal="Organise and present the pre-extracted structured fields clearly",
        backstory=(
            "Specialist who presents extracted data in clean, structured format. "
            "Never invents new fields — only uses what was already extracted."
        ),
        llm=_OLLAMA_LLM,
        verbose=False,
        allow_delegation=False,
    )
    _crew_validator = Agent(
        role="Anomaly Validator",
        goal="Report validation results based solely on the provided validation data",
        backstory=(
            "Quality-control expert who reads pre-run validation results and "
            "presents them clearly. Does not perform new validation checks."
        ),
        llm=_OLLAMA_LLM,
        verbose=False,
        allow_delegation=False,
    )
    _crew_decision = Agent(
        role="Decision Agent",
        goal="Generate a structured decision report using only provided extracted and validated data",
        backstory=(
            "Senior analyst who produces a clear, numbered decision report. "
            "Uses only the extracted fields and validation flags provided. "
            "Never invents payment amounts, dates, or names."
        ),
        llm=_OLLAMA_LLM,
        verbose=False,
        allow_delegation=False,
    )
    _crew_summarizer = Agent(
        role="Document Summarizer",
        goal="Produce a 2-3 sentence summary strictly from the provided extracted data",
        backstory=(
            "Communication specialist who writes concise summaries using only "
            "the structured fields and document snippet provided. No embellishment."
        ),
        llm=_OLLAMA_LLM,
        verbose=False,
        allow_delegation=False,
    )

    # ── Define tasks with REAL data injected ──────────────────
    _task_classify = Task(
        description=(
            f"{_NO_HALLUCINATION}\n\n"
            f"The document has already been classified. Confirm and state the type.\n\n"
            f"Classified Type: {doc_type}\n\n"
            f"Document Snippet (reference only):\n{_ctx}"
        ),
        expected_output=(
            f"Confirmed document type: {doc_type}. "
            "One sentence confirming the classification."
        ),
        agent=_crew_classifier,
    )
    _task_extract = Task(
        description=(
            f"{_NO_HALLUCINATION}\n\n"
            f"Present the following pre-extracted fields from this {doc_type} "
            f"in a clean numbered list. Do NOT add fields not listed below.\n\n"
            f"Extracted Data:\n{_structured_fields}"
        ),
        expected_output=(
            "Numbered list of extracted fields exactly as provided. "
            "Format: '1. Label: Value'"
        ),
        agent=_crew_extractor,
    )
    _task_validate = Task(
        description=(
            f"{_NO_HALLUCINATION}\n\n"
            f"Present the following pre-run validation results for this {doc_type}. "
            f"Do NOT add new issues not listed below.\n\n"
            f"Validation Results:\n{_validation_output}\n\n"
            f"Extracted Data (for context):\n{_structured_fields}"
        ),
        expected_output=(
            "Bullet list of validation results exactly as provided. "
            "State clearly if no issues were found."
        ),
        agent=_crew_validator,
    )
    _task_decision = Task(
        description=(
            f"{_NO_HALLUCINATION}\n\n"
            f"Using ONLY the extracted data and validation results below, "
            f"generate a structured decision report for this {doc_type}. "
            f"Do not invent information not present in the data.\n\n"
            f"Extracted Data:\n{_structured_fields}\n\n"
            f"Validation Results:\n{_validation_output}\n\n"
            f"Your output must follow this exact structure:\n"
            f"1. Document Type: [type]\n"
            f"2. Key Fields: [list fields from Extracted Data]\n"
            f"3. Validation: [list from Validation Results]\n"
            f"4. Decision: [Safe to Pay / Review Required / Do Not Pay / etc.]\n"
            f"5. Reason: [based only on fields present or missing above]\n"
            f"6. Suggested Action: [concrete next step based on decision]"
        ),
        expected_output=(
            "Structured numbered report with sections: Document Type, Key Fields, "
            "Validation, Decision, Reason, Suggested Action. "
            "Based strictly on provided data."
        ),
        agent=_crew_decision,
    )
    _task_summarize = Task(
        description=(
            f"{_NO_HALLUCINATION}\n\n"
            f"Summarize this {doc_type} in exactly 2-3 sentences using ONLY "
            f"the extracted fields below. Do not add facts not present.\n\n"
            f"Extracted Data:\n{_structured_fields}\n\n"
            f"Document Snippet (reference only):\n{_ctx}"
        ),
        expected_output=(
            "2-3 sentence plain-text summary using only the provided extracted fields."
        ),
        agent=_crew_summarizer,
    )

    crew = Crew(
        agents=[
            _crew_classifier, _crew_extractor,
            _crew_validator,  _crew_decision, _crew_summarizer,
        ],
        tasks=[
            _task_classify, _task_extract,
            _task_validate, _task_decision, _task_summarize,
        ],
        process=Process.sequential,
        verbose=False,
    )
    return crew


def _format_crewai_output(
    crew_result,
    doc_type: str,
    extracted_points: list,
    validation_flags: list,
) -> str:
    """
    Format CrewAI kickoff result into a clean structured completion box.
    Falls back to a rule-based format if crew_result is empty/None.
    """
    # ── Rule-based structured report (always accurate, no hallucination) ──
    lines = []
    lines.append("Crew Execution Completed")
    lines.append("=" * 44)
    lines.append("Final Output:")
    lines.append("")
    lines.append(f"1. Document Type: {doc_type}")
    lines.append("")
    lines.append("2. Key Fields:")
    if extracted_points:
        for lbl, val in extracted_points:
            lines.append(f"   • {lbl}: {val}")
    else:
        lines.append("   • No fields extracted")
    lines.append("")
    lines.append("3. Validation:")
    if validation_flags:
        for sev, msg in validation_flags:
            lines.append(f"   • {sev}: {msg}")
    else:
        lines.append("   • No validation issues")
    lines.append("")

    # Decision — derive from validation flags (rule-based, no hallucination)
    _labels_present = {lbl for lbl, _ in extracted_points}
    _is_overdue   = any("OVERDUE" in m for _, m in validation_flags)
    _has_critical = any("Critical" in s for s, _ in validation_flags)
    _missing_vendor = "Vendor / From" not in _labels_present
    _missing_total  = "Total Amount"  not in _labels_present

    if "Invoice" in doc_type:
        if _missing_vendor or _missing_total:
            _decision_str = "❌ Do Not Pay"
            _reason_str   = "Critical field(s) missing — vendor or total not found."
            _action_str   = "Obtain missing details before processing payment."
        elif _is_overdue:
            _decision_str = "🔴 Pay Immediately — Overdue"
            _reason_str   = "Invoice is past due date. All critical fields are present."
            _action_str   = "Escalate to accounts payable immediately."
        elif _has_critical:
            _decision_str = "⚠️ Review Required"
            _reason_str   = "One or more critical validation flags raised."
            _action_str   = "Resolve flagged issues before approving payment."
        else:
            _decision_str = "✔ Safe to Pay"
            _reason_str   = "All critical fields present. No critical anomalies detected."
            _action_str   = "Proceed with payment and archive invoice."
    elif "Resume" in doc_type:
        _missing_skills = "Technical Skills" not in _labels_present
        _missing_edu    = "Education"        not in _labels_present
        if _missing_skills or _missing_edu:
            _decision_str = "⚠️ Incomplete Profile"
            _reason_str   = "Key resume sections are missing."
            _action_str   = "Request updated resume from candidate."
        else:
            _decision_str = "✔ Profile Complete"
            _reason_str   = "Key sections detected — skills and education present."
            _action_str   = "Proceed to screening interview."
    elif "Complaint" in doc_type:
        _decision_str = "🚨 Action Required"
        _reason_str   = "Complaint document requires a formal response."
        _action_str   = "Assign to customer support team and respond within SLA."
    else:
        _decision_str = "ℹ️ Review Document"
        _reason_str   = "Document reviewed and fields extracted."
        _action_str   = "Route to relevant department for further action."

    lines.append(f"4. Decision: {_decision_str}")
    lines.append("")
    lines.append("5. Reason:")
    lines.append(f"   • {_reason_str}")
    # Add field-level reasons
    for lbl, val in extracted_points[:4]:
        lines.append(f"   • {lbl} confirmed: {val[:60]}")
    lines.append("")
    lines.append("6. Suggested Action:")
    lines.append(f"   • {_action_str}")
    lines.append("")

    # Append actual CrewAI LLM output if available (supplemental)
    if crew_result:
        _raw = str(crew_result).strip()
        if _raw:
            lines.append("-" * 44)
            lines.append("CrewAI Agent Insights (supplemental):")
            lines.append(_raw[:800])

    lines.append("")
    lines.append("=" * 44)
    lines.append("Generated by DocuSense AI — CrewAI Layer")
    return "\n".join(lines)

# ── Tesseract path (Windows) ─────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ════════════════════════════════════════════════════════════════
# ⚙️  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocuSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════
# 🎨 STYLES  — refined dark professional theme
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12151e;
    border-right: 1px solid #1e2130;
}

/* ── Header bar ── */
.ds-header {
    background: linear-gradient(135deg, #13161f 0%, #1a1d2e 100%);
    border-bottom: 2px solid #2563eb;
    padding: 24px 32px 20px;
    margin: -1rem -1rem 2rem;
    border-radius: 0 0 16px 16px;
}
.ds-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
}
.ds-subtitle {
    font-size: 0.82rem;
    color: #6b7280;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}
.ds-badge {
    display: inline-block;
    background: #2563eb20;
    color: #60a5fa;
    border: 1px solid #2563eb50;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    margin-left: 12px;
    vertical-align: middle;
}

/* ── Metric cards ── */
.metric-card {
    background: #12151e;
    border: 1px solid #1e2130;
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    border-radius: 14px 14px 0 0;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b7280;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #4b5563;
    margin-top: 6px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    border-left: 3px solid #2563eb;
    padding-left: 14px;
    margin: 28px 0 14px;
}

/* ── Key-point cards ── */
.kp-card {
    background: #12151e;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 7px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.15s;
}
.kp-card:hover { border-color: #2563eb50; }
.kp-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    min-width: 130px;
    padding-top: 2px;
}
.kp-value {
    font-size: 0.9rem;
    color: #e2e8f0;
    line-height: 1.5;
}

/* ── Flags ── */
.flag-card {
    border-radius: 10px;
    padding: 13px 18px;
    margin: 7px 0;
    font-size: 0.88rem;
    border-left: 4px solid transparent;
}
.flag-ok   { background:#052e1620; border-color:#22c55e; color:#86efac; }
.flag-warn { background:#42180020; border-color:#f59e0b; color:#fbbf24; }
.flag-crit { background:#45002520; border-color:#ef4444; color:#fca5a5; }

/* ── Keyword pills ── */
.kw-pill {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2a2f45;
    border-radius: 20px;
    padding: 5px 14px;
    margin: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    transition: all 0.15s;
}
.kw-pill:hover { border-color: #2563eb; color: #60a5fa; }

/* ── Summary box ── */
.summary-box {
    background: #12151e;
    border: 1px solid #1e2130;
    border-radius: 14px;
    padding: 22px 24px;
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.75;
    font-style: italic;
    position: relative;
}
.summary-box::before {
    content: '❝';
    position: absolute;
    top: 12px; left: 18px;
    font-size: 2.5rem;
    color: #2563eb20;
    line-height: 1;
    font-style: normal;
}

/* ── Chat ── */
.chat-user {
    background: #1e2949;
    border: 1px solid #2563eb40;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 4px auto;
    max-width: 75%;
    font-size: 0.88rem;
    color: #bfdbfe;
    float: right;
    clear: both;
}
.chat-ai {
    background: #131720;
    border: 1px solid #1e2130;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 4px 0 8px;
    max-width: 85%;
    font-size: 0.88rem;
    color: #e2e8f0;
    float: left;
    clear: both;
    line-height: 1.65;
}
.chat-wrap { overflow: hidden; margin-bottom: 8px; }

/* ── Memory card (sidebar) ── */
.mem-card {
    background: #0d0f14;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 6px 0;
    font-size: 0.78rem;
}
.mem-type { color: #60a5fa; font-family: 'DM Mono', monospace; font-size: 0.7rem; }
.mem-file { color: #e2e8f0; font-weight: 500; margin: 2px 0; }
.mem-time { color: #4b5563; font-size: 0.68rem; }

/* ── Quality gauge ── */
.gauge-wrap { text-align: center; padding: 10px 0; }
.gauge-num  {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
}
.gauge-label { font-size: 0.72rem; color: #6b7280; margin-top: 4px; font-family: 'DM Mono', monospace; }

/* ── Divider ── */
.ds-divider {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 24px 0;
}

/* ── Buttons ── */
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #12151e;
    border: 2px dashed #1e2130;
    border-radius: 14px;
    padding: 20px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #2563eb50; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #0f172a;
    border: 1px solid #2563eb;
    color: #60a5fa;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #12151e; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7280;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] { background: #1e2130 !important; color: #f1f5f9 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #2563eb; }

/* ── Chat input ── */
.stChatInput > div { background: #12151e; border: 1px solid #1e2130; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 🏷️  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ds-header">
    <div class="ds-title">DocuSense AI <span class="ds-badge">v2.0</span></div>
    <div class="ds-subtitle">Intelligent Document Analysis · Extraction · Q&amp;A · Memory</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 💾 LONG-TERM MEMORY
# ════════════════════════════════════════════════════════════════
MEMORY_FILE = "docusense_memory.json"

def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(record: dict):
    memory = load_memory()
    memory.append(record)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


# ════════════════════════════════════════════════════════════════
# 📂 FILE EXTRACTION
# ════════════════════════════════════════════════════════════════
@st.cache_data
def extract_text(file) -> str:
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        d = docx.Document(file)
        for para in d.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    return text


# ════════════════════════════════════════════════════════════════
# 🤖  AGENTS
# ════════════════════════════════════════════════════════════════

# ── AGENT 1: OCR CLEANER ─────────────────────────────────────
def ocr_cleaner_agent(raw: str) -> dict:
    text = re.sub(r"[^\x20-\x7E\n]", " ", raw)
    text = re.sub(r" {2,}", " ", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        alpha = sum(c.isalpha() for c in line)
        if len(line) > 3 and alpha / max(len(line), 1) < 0.20:
            continue
        if re.search(r"(.)\1{5,}", line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    return {"output": cleaned, "removed": max(len(raw) - len(cleaned), 0)}


# ── AGENT 2: CLASSIFIER ──────────────────────────────────────
def classifier_agent(text: str) -> dict:
    lower = text.lower()
    scores = {
        "📄 Resume":          sum([
            "education" in lower, "skills" in lower, "experience" in lower,
            "project" in lower, "objective" in lower, "certification" in lower,
            "gpa" in lower, "internship" in lower, "linkedin" in lower,
        ]),
        "🧾 Invoice":         sum([
            "invoice" in lower, "amount" in lower, "total" in lower,
            "bill" in lower, "payable" in lower, "vendor" in lower,
            "subtotal" in lower, "tax" in lower, "due" in lower,
        ]),
        "⚠️ Complaint":       sum([
            "complaint" in lower, "issue" in lower, "problem" in lower,
            "dissatisfied" in lower, "refund" in lower, "unhappy" in lower,
        ]),
        "📑 Legal Document":  sum([
            "whereas" in lower, "hereby" in lower, "agreement" in lower,
            "clause" in lower, "jurisdiction" in lower, "party" in lower,
        ]),
        "📰 News/Article":    sum([
            "according" in lower, "reported" in lower, "published" in lower,
            "journalist" in lower, "correspondent" in lower,
        ]),
        "🔬 Research Paper":  sum([
            "abstract" in lower, "methodology" in lower, "conclusion" in lower,
            "references" in lower, "hypothesis" in lower, "findings" in lower,
        ]),
        "📃 General Document": 1,
    }
    best  = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round((scores[best] / total) * 100) if total else 0
    return {"doc_type": best, "confidence": conf, "all_scores": scores}


# ── AGENT 3: EXTRACTOR ───────────────────────────────────────
def extractor_agent(text: str, doc_type: str) -> dict:
    points = []
    flat = text.replace("\n", " ")

    # ── INVOICE ──────────────────────────────────────────────
    if "Invoice" in doc_type:
        vendor = re.search(
            r"((?:From|Vendor|Billed\s+by|Company)[:\s]+)([A-Z][A-Za-z0-9\s&,\.]{2,50}?)(?=\n|,|\.|$)",
            text, re.IGNORECASE)
        if not vendor:
            vendor = re.search(
                r"([A-Z][a-zA-Z]+\s+(?:Solutions|Corp|Inc|Ltd|LLC|Services|Technologies|Enterprises|Group))",
                text)
        if vendor:
            v = (vendor.group(2) if vendor.lastindex and vendor.lastindex >= 2 else vendor.group(0)).strip()
            points.append(("Vendor / From", v))

        billed = re.search(
            r"(?:Billed\s+To|Bill\s+To|Client|Customer)[:\s]+([A-Z][A-Za-z0-9\s&,\.]{2,50}?)(?=\n|$)",
            text, re.IGNORECASE)
        if billed:
            points.append(("Billed To", billed.group(1).strip()))

        inv = re.search(r"Invoice\s*(?:Number|No\.?|#|ID)\s*[:\-]?\s*([A-Z0-9\-\/]+)", text, re.IGNORECASE)
        if inv:
            points.append(("Invoice Number", inv.group(1).strip()))

        inv_date = re.search(r"(?:Invoice\s+)?Date\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})", text, re.IGNORECASE)
        if inv_date:
            points.append(("Invoice Date", inv_date.group(1).strip()))

        due = re.search(r"(?:Due|Payable\s+by|Due\s+Date)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})", text, re.IGNORECASE)
        if due:
            points.append(("Payment Due", due.group(1).strip()))

        totals = re.findall(r"(?:Total\s+Amount|Grand\s+Total|Amount\s+Due|Total)\s*[:\-]?\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if totals:
            points.append(("Total Amount", f"${totals[-1].strip()}"))

        subtotal = re.search(r"Sub\s*total\s*[:\-]?\s*\$?([\d,]+\.\d{2})", text, re.IGNORECASE)
        if subtotal:
            points.append(("Subtotal", f"${subtotal.group(1).strip()}"))

        tax = re.search(r"(?:Tax\s+Amount|Tax|VAT|GST)\s*[:\-]?\s*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
        if tax:
            points.append(("Tax", f"${tax.group(1).strip()}"))

        terms = re.search(r"(Net\s*\d+\s*days?|COD|Due on Receipt)", text, re.IGNORECASE)
        if terms:
            points.append(("Payment Terms", terms.group(1).strip()))

    # ── RESUME ───────────────────────────────────────────────
    elif "Resume" in doc_type:

        # Candidate name — improved header-based extraction
        name_found = False

        # Extended blacklist: location names, institutions, section headers
        _name_blacklist = {
            "resume", "curriculum", "vitae", "profile", "objective", "summary",
            "skills", "experience", "education", "contact", "email", "phone",
            "address", "linkedin", "github", "portfolio", "references",
            "karnataka", "india", "ballari", "technology", "institute",
            "university", "college", "department", "engineering", "science",
            "bachelor", "master", "doctor", "graduate", "student",
            "declaration", "project", "internship", "achievement", "award",
            "certification", "language", "hobby", "interest", "reference",
        }

        # Step 1: Work only on first 60 words (resume header area)
        _header_words = text.split()
        _header_text  = " ".join(_header_words[:60])

        # Step 2: Strip emails and URLs from header text before searching
        _clean_header = re.sub(r"\S+@\S+", "", _header_text)
        _clean_header = re.sub(r"http\S+", "", _clean_header)
        _clean_header = re.sub(r"[|@#\(\)\[\]/\\]", " ", _clean_header)
        _clean_header = re.sub(r"\s{2,}", " ", _clean_header).strip()

        # Strategy A: explicit "Name:" label anywhere in full text
        _nm_label = re.search(
            r"(?:Name|Full\s+Name)\s*[:\-]\s*([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,3})",
            text, re.IGNORECASE
        )
        if _nm_label:
            _nm_val = _nm_label.group(1).strip()
            _nm_words = _nm_val.split()
            if not any(w.lower() in _name_blacklist for w in _nm_words):
                points.append(("Candidate Name", _nm_val))
                name_found = True

        # Strategy B: Title-case pattern (e.g. "Rahul Sharma") in cleaned header
        if not name_found:
            _name_candidates = re.findall(
                r"\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20}))?\b",
                _clean_header
            )
            for _match in _name_candidates:
                _parts     = [p for p in _match if p]
                _candidate = " ".join(_parts)
                if (
                    not any(p.lower() in _name_blacklist for p in _parts)
                    and all(len(p) > 1 for p in _parts)
                    and 2 <= len(_parts) <= 3
                ):
                    points.append(("Candidate Name", _candidate))
                    name_found = True
                    break

        # Strategy C: ALL CAPS pattern (e.g. "RAHUL SHARMA") in cleaned header
        if not name_found:
            _caps_candidates = re.findall(
                r"\b([A-Z]{3,20})\s+([A-Z]{3,20})(?:\s+([A-Z]{3,20}))?\b",
                _clean_header
            )
            for _cmatch in _caps_candidates:
                _cparts     = [p for p in _cmatch if p]
                _ccandidate = " ".join(_cparts)
                if (
                    not any(p.lower() in _name_blacklist for p in _cparts)
                    and all(len(p) > 1 for p in _cparts)
                    and 2 <= len(_cparts) <= 3
                ):
                    # Convert ALL CAPS to Title Case for display
                    points.append(("Candidate Name", _ccandidate.title()))
                    name_found = True
                    break

        # Strategy D: Line-by-line scan of first 10 non-empty lines
        if not name_found:
            _resume_lines = [l.strip() for l in text.split("\n") if l.strip()]
            for _line in _resume_lines[:10]:
                _words = _line.split()
                if (
                    2 <= len(_words) <= 3
                    and all(w[0].isupper() for w in _words if w and w[0].isalpha())
                    and not re.search(r"[\d@|:\/\\\(\)\[\]#]", _line)
                    and not any(w.lower() in _name_blacklist for w in _words)
                    and all(len(w) > 1 for w in _words)
                ):
                    points.append(("Candidate Name", _line))
                    name_found = True
                    break

        # Strategy E: Fallback — first capitalized word sequence from line 1
        if not name_found:
            _first_lines = [l.strip() for l in text.split("\n") if l.strip()]
            if _first_lines:
                _fl_clean = re.sub(r"\S+@\S+", "", _first_lines[0])
                _fl_clean = re.sub(r"http\S+", "", _fl_clean).strip()
                _fl_words = _fl_clean.split()
                _cap_seq  = [w for w in _fl_words
                             if w and w[0].isupper()
                             and len(w) > 1
                             and w.lower() not in _name_blacklist]
                if len(_cap_seq) >= 2:
                    points.append(("Candidate Name", " ".join(_cap_seq[:3])))
                    name_found = True

        if not name_found:
            points.append(("Candidate Name", "Not detected"))

        # Contact info
        em = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        if em:
            points.append(("Email", em.group(0)))

        ph = re.search(r"(\+?[\d][\d\s\-\(\)]{8,15}[\d])", text)
        if ph:
            cleaned_ph = re.sub(r"\s+", "", ph.group(1).strip())
            if 8 <= len(cleaned_ph) <= 15:
                points.append(("Phone", ph.group(1).strip()))

        li = re.search(r"linkedin\.com\/in\/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if li:
            points.append(("LinkedIn", f"linkedin.com/in/{li.group(1)}"))

        gh = re.search(r"github\.com\/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if gh:
            points.append(("GitHub", f"github.com/{gh.group(1)}"))

        # Current role / objective
        obj = re.search(r"(?:Objective|Summary|Profile|About Me)\s*[:\-]?\s*\n?([\s\S]{20,200}?)(?=\n[A-Z][A-Z\s]{3,}|\Z)", text, re.IGNORECASE)
        if obj:
            flat_obj = re.sub(r"\s+", " ", obj.group(1)).strip()
            points.append(("Profile / Objective", flat_obj[:150]))

        # Technical Skills — grab the actual section
        tech = re.search(
            r"(?:Technical\s+)?Skills?\s*[:\-]?\s*\n?([\s\S]{10,400}?)(?=\n[A-Z][A-Z\s]{3,}|\Z)",
            text, re.IGNORECASE)
        if tech:
            raw = re.sub(r"\s+", " ", tech.group(1)).strip()
            # Remove section headers bleed
            raw = re.sub(r"\b(?:Experience|Education|Projects|Certifications|Awards)\b.*", "", raw, flags=re.IGNORECASE).strip()
            points.append(("Technical Skills", raw[:180] + ("..." if len(raw) > 180 else "")))

        # Work Experience — broadened regex + inline keyword fallback
        exp_sec = re.search(
            r"(?:Professional\s+Experience|Work\s+Experience|Internship|Experience)"
            r"\s*[:\-]?\s*\n?([\s\S]{10,500}?)(?=\n[A-Z][A-Z\s]{3,}|\Z)",
            text, re.IGNORECASE)
        if exp_sec:
            exp_text = exp_sec.group(1).strip()
            exp_lines = [l.strip() for l in exp_text.split("\n") if len(l.strip()) > 10]
            if exp_lines:
                points.append(("Latest Experience", re.sub(r"\s+", " ", exp_lines[0])[:150]))
                if len(exp_lines) > 2:
                    dates = re.findall(r"(\d{4}\s*[-–]\s*(?:\d{4}|Present|Current))", exp_text, re.IGNORECASE)
                    if dates:
                        points.append(("Experience Period", dates[0]))
        # Inline keyword fallback: 'Intern', 'Worked at', 'Experience in'
        if not any(lbl in ("Latest Experience", "Experience Period") for lbl, _ in points):
            _exp_inline = re.search(
                r"(?:Intern(?:ship)?|Worked\s+at|Experience\s+in)[^\n]{5,150}",
                text, re.IGNORECASE)
            if _exp_inline:
                points.append(("Latest Experience",
                               re.sub(r"\s+", " ", _exp_inline.group(0).strip())[:150]))
        # Education
        edu_sec = re.search(
            r"Education\s*[:\-]?\s*\n?([\s\S]{10,400}?)(?=\n[A-Z][A-Z\s]{3,}|\Z)",
            text, re.IGNORECASE)
        if edu_sec:
            edu_text = edu_sec.group(1).strip()
            for line in edu_text.split("\n"):
                line = line.strip()
                if len(line) > 10:
                    points.append(("Education", re.sub(r"\s+", " ", line)[:150]))
                    break

        # GPA
        gpa = re.search(r"GPA\s*[:\-]?\s*([\d.]+)\s*(?:\/\s*[\d.]+)?", text, re.IGNORECASE)
        if gpa:
            points.append(("GPA", gpa.group(1)))

    # ── COMPLAINT ────────────────────────────────────────────
    elif "Complaint" in doc_type:
        sub = re.search(r"Subject\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if sub: points.append(("Subject", sub.group(1).strip()[:120]))

        comp = re.search(r"(?:complaint|issue|problem|concern)[^\.\n]{5,120}", text, re.IGNORECASE)
        if comp: points.append(("Core Issue", comp.group(0).strip()[:120]))

        sender = re.search(r"(?:From|Submitted by|Complainant)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if sender: points.append(("Submitted By", sender.group(1).strip()[:80]))

        dt = re.search(r"Date\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})", text, re.IGNORECASE)
        if dt: points.append(("Date", dt.group(1).strip()))

        ref = re.search(r"(?:Order|Reference|Ticket|Case)\s*(?:No\.?|#|ID)?\s*[:\-]?\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
        if ref: points.append(("Reference", ref.group(1).strip()))

    # ── LEGAL ────────────────────────────────────────────────
    elif "Legal" in doc_type:
        parties = re.findall(r"(?:between|party|parties)[^\.\n]{0,100}", text, re.IGNORECASE)
        for p in parties[:2]:
            points.append(("Party", p.strip()[:120]))

        eff = re.search(r"[Ee]ffective\s*[Dd]ate?\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})", text, re.IGNORECASE)
        if eff: points.append(("Effective Date", eff.group(1)))

        exp = re.search(r"[Ee]xpir(?:y|ation)\s*[Dd]ate?\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},? \d{4})", text, re.IGNORECASE)
        if exp: points.append(("Expiry Date", exp.group(1)))

        jur = re.search(r"[Jj]urisdiction\s*[:\-]?\s*(.+?)(?=\.|,|\n)", text, re.IGNORECASE)
        if jur: points.append(("Jurisdiction", jur.group(1).strip()[:100]))

    # ── RESEARCH ─────────────────────────────────────────────
    elif "Research" in doc_type:
        ab = re.search(r"Abstract\s*[:\-]?\s*(.{30,400})", text, re.IGNORECASE)
        if ab: points.append(("Abstract", ab.group(1).strip()[:200]))

        auth = re.search(r"Author[s]?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if auth: points.append(("Author(s)", auth.group(1).strip()[:100]))

        kw = re.search(r"[Kk]ey\s*[Ww]ords?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
        if kw: points.append(("Keywords", kw.group(1).strip()[:120]))

        doi = re.search(r"(?:DOI|doi)[:\s]+([^\s,\)]+)", text)
        if doi: points.append(("DOI", doi.group(1).strip()))

    # ── GENERAL ──────────────────────────────────────────────
    else:
        sentences = re.split(r"[.!?]\s+", flat)
        count = 0
        for s in sentences:
            s = s.strip()
            alpha = sum(c.isalpha() for c in s)
            if len(s) > 40 and alpha / max(len(s), 1) > 0.60:
                points.append(("Excerpt", s[:140]))
                count += 1
            if count >= 5:
                break

    # Safety: strip non-ASCII garble
    safe = [
        (lbl, val) for lbl, val in points
        if not re.search(r"[^\x00-\x7F]", val)
        and not re.search(r"(.)\1{5,}", val)
        and len(val.strip()) > 1
    ]
    return {"points": safe[:9]}


# ── AGENT 4: ANOMALY ─────────────────────────────────────────
def anomaly_agent(text: str, doc_type: str, points: list) -> dict:
    flags  = []
    labels = {lbl for lbl, _ in points}
    vals   = {lbl: val for lbl, val in points}

    if "Invoice" in doc_type:
        amts = re.findall(r"\$([\d,]+\.\d{2})", text)
        nums = []
        for a in amts:
            try: nums.append(float(a.replace(",", "")))
            except: pass
        if len(nums) >= 3:
            items_sum = sum(sorted(nums)[:-1])
            if abs(sorted(nums)[-1] - items_sum) > 1.0:
                flags.append(("🔴 Critical", f"Total (${sorted(nums)[-1]:,.2f}) does not match line item sum (${items_sum:,.2f})."))
        due = re.search(r"(?:Due|Payable)\s*(?:by)?\s*[:\-]?\s*(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})", text, re.IGNORECASE)
        if due:
            try:
                m, d, y = int(due.group(1)), int(due.group(2)), int(due.group(3))
                if y < 100: y += 2000
                due_dt = datetime(y, m, d)
                if due_dt < datetime.now():
                    days = (datetime.now() - due_dt).days
                    flags.append(("🔴 Critical", f"Invoice OVERDUE by {days} day(s). Due: {due_dt.strftime('%b %d, %Y')}"))
                else:
                    days = (due_dt - datetime.now()).days
                    flags.append(("🟡 Warning", f"Payment due in {days} day(s)."))
            except: pass
        for req in ["Invoice Number", "Total Amount", "Vendor / From"]:
            if req not in labels:
                flags.append(("🟡 Warning", f"Missing field: {req}"))

    elif "Resume" in doc_type:
        for req in ["Candidate Name", "Email", "Technical Skills", "Education"]:
            if req not in labels:
                flags.append(("🟡 Warning", f"Resume missing: {req}"))
        if vals.get("Candidate Name") == "Not detected":
            flags.append(("🟡 Warning", "Candidate name could not be auto-detected."))
        if len(text.split()) < 150:
            flags.append(("🟡 Warning", "Resume is very short — may be incomplete."))

    elif "Complaint" in doc_type:
        if "Subject" not in labels:
            flags.append(("🟡 Warning", "No Subject line found in complaint."))
        if len(text.split()) < 50:
            flags.append(("🟡 Warning", "Complaint body is very brief."))

    if len(text.strip()) < 30:
        flags.append(("🔴 Critical", "Document appears empty or unreadable."))

    if not flags:
        flags.append(("🟢 OK", "All validation checks passed. Document looks complete."))

    return {"flags": flags}


# ── AGENT 5: CRITIC ──────────────────────────────────────────
def critic_agent(points: list, doc_type: str, text: str) -> dict:
    """Score each extracted point; retry with Ollama if quality too low."""
    scored = []
    for lbl, val in points:
        if (
            len(val.strip()) < 3
            or val.lower() in {"not detected", "n/a", "none", "not found", "unknown"}
        ):
            scored.append((lbl, val, "❌"))
        elif len(val.strip()) >= 3:
            scored.append((lbl, val, "✅"))
        else:
            scored.append((lbl, val, "⚠️"))

    good   = [(l, v) for l, v, s in scored if s == "✅"]
    qs     = round(len(good) / max(len(scored), 1) * 100)
    retry  = False
    final  = good

    if qs < 55 or len(final) < 2:
        retry = True
        try:
            res = ollama.chat(
                model="phi",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Document type: {doc_type}\n"
                        "Extract the 5 most important facts as exactly 'Label: Value' lines.\n"
                        "Be specific. No markdown. No symbols outside ASCII.\n\n"
                        f"{text[:600]}"
                    )
                }]
            )
            raw = res["message"]["content"]
            retry_pts = []
            for line in raw.split("\n"):
                line = line.strip("•-– 1234567890.")
                if ":" in line:
                    parts = line.split(":", 1)
                    lbl2  = parts[0].strip()
                    val2  = parts[1].strip()
                    if (
                        len(lbl2) > 1 and len(val2) > 2
                        and not re.search(r"[^\x00-\x7F]", lbl2 + val2)
                        and val2.lower() not in {"n/a", "none", "unknown"}
                    ):
                        retry_pts.append((lbl2, val2))
            if retry_pts:
                final = retry_pts[:6]
                qs    = 75
        except Exception:
            pass

    return {"quality_score": qs, "scored": scored, "final": final, "retry": retry}


# ── AGENT 6: AI SUMMARIZER ───────────────────────────────────
def summarizer_agent(text: str, doc_type: str) -> dict:
    """Generate a concise natural-language summary using Ollama (local model)."""
    try:
        res = ollama.chat(
            model="phi",
            messages=[{
                "role": "user",
                "content": (
                    f"You are analyzing a {doc_type}. "
                    "Write a clear, professional 3-4 sentence summary of the document. "
                    "Mention the most important facts. Do NOT use bullet points. "
                    "Output plain text only.\n\n"
                    f"Document text:\n{text[:2000]}"
                )
            }]
        )
        summary = res["message"]["content"].strip()
        summary = re.sub(r"[^\x00-\x7E\n]", "", summary).strip()
        return {"summary": summary, "ok": True}
    except Exception as e:
        return {"summary": "", "ok": False, "error": str(e)}


# ── AGENT 7: MEMORY ──────────────────────────────────────────
def memory_agent(doc_type: str, filename: str, points: list, flags: list) -> dict:
    record = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename":   filename,
        "doc_type":   doc_type,
        "key_points": [f"{l}: {v}" for l, v in points],
        "flags":      [f"{s}: {m}" for s, m in flags],
    }
    if "doc_history" not in st.session_state:
        st.session_state["doc_history"] = []
    st.session_state["doc_history"].append(record)
    save_memory(record)
    return {"record": record, "total": len(st.session_state["doc_history"])}


# ── AGENT 8: REPORTER ─────────────────────────────────────────
def reporter_agent(doc_type, filename, points, flags, qs, conf, summary) -> dict:
    sep = "─" * 50
    lines = [
        "DOCUSENSE AI — DOCUMENT INTELLIGENCE REPORT",
        sep,
        f"File          : {filename}",
        f"Document Type : {doc_type}",
        f"Confidence    : {conf}%",
        f"Quality Score : {qs}%",
        f"Processed At  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if summary:
        lines += ["SUMMARY:", summary, ""]

    lines += [sep, "EXTRACTED INFORMATION:"]
    for lbl, val in points:
        lines.append(f"  {lbl:<25} {val}")

    lines += ["", sep, "VALIDATION FLAGS:"]
    for sev, msg in flags:
        lines.append(f"  {sev}: {msg}")

    lines += ["", sep, "Generated by DocuSense AI v2.0"]
    return {"report": "\n".join(lines)}


# ════════════════════════════════════════════════════════════════
# 🎯 ORCHESTRATOR
# ════════════════════════════════════════════════════════════════
def orchestrator(raw_text: str, filename: str) -> dict:
    r_ocr   = ocr_cleaner_agent(raw_text)
    clean   = r_ocr["output"]
    flat    = clean.replace("\n", " ")

    r_cls   = classifier_agent(clean)
    doc_type = r_cls["doc_type"]

    r_ext   = extractor_agent(clean, doc_type)
    r_anom  = anomaly_agent(flat, doc_type, r_ext["points"])
    r_crit  = critic_agent(r_ext["points"], doc_type, flat)
    r_sum   = summarizer_agent(flat, doc_type)
    r_mem   = memory_agent(doc_type, filename, r_crit["final"], r_anom["flags"])
    r_rep   = reporter_agent(
        doc_type, filename, r_crit["final"],
        r_anom["flags"], r_crit["quality_score"],
        r_cls["confidence"], r_sum["summary"]
    )

    return {
        "ocr":        r_ocr,
        "classifier": r_cls,
        "extractor":  r_ext,
        "anomaly":    r_anom,
        "critic":     r_crit,
        "summarizer": r_sum,
        "memory":     r_mem,
        "reporter":   r_rep,
        "clean_text": clean,
        "flat_text":  flat,
    }


# ════════════════════════════════════════════════════════════════
# 🔍 KEYWORD EXTRACTOR
# ════════════════════════════════════════════════════════════════
STOPWORDS = {
    "this","that","with","from","have","using","above","there","their",
    "about","which","when","where","been","being","thank","your","invoice",
    "number","details","total","date","payment","terms","payable","will",
    "shall","also","such","they","them","through","these","those","other",
    "after","before","during","between","under","over","individual",
    "content","service","general","within","without","around","based",
    "would","could","should","while","since","every","document","information",
    "please","given","value","field","found","amount","years","month",
}

def get_keywords(text: str, n: int = 10) -> list:
    words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
    freq  = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:n]


# ════════════════════════════════════════════════════════════════
# 💬 DOCUMENT Q&A CHAT
# ════════════════════════════════════════════════════════════════
def ask_document(question: str, doc_text: str, doc_type: str) -> str:
    """Answer any question about the document using Ollama (local model)."""
    try:
        res = ollama.chat(
            model="phi",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert document analyst. The user has uploaded a {doc_type}. "
                        "Answer every question STRICTLY based on the document content provided. "
                        "If the information is not present in the document, say so clearly. "
                        "Be concise, accurate, and factual. Do not guess or hallucinate."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the full document content:\n\n{doc_text[:4000]}\n\n"
                        f"Question: {question}"
                    )
                }
            ]
        )
        answer = res["message"]["content"].strip()
        return re.sub(r"[^\x00-\x7E\n]", "", answer)
    except Exception as e:
        return f"Could not get an answer: {e}"


# ════════════════════════════════════════════════════════════════
# 🖥️  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 DocuSense AI")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Stats
    all_mem = load_memory()
    col_a, col_b = st.columns(2)
    col_a.metric("Session Docs", len(st.session_state.get("doc_history", [])))
    col_b.metric("All-Time Docs", len(all_mem))

    st.markdown("---")
    st.markdown("**📂 Recent Documents**")

    hist = st.session_state.get("doc_history", [])
    if hist:
        for rec in reversed(hist[-5:]):
            st.markdown(f"""
            <div class='mem-card'>
                <div class='mem-type'>{rec['doc_type']}</div>
                <div class='mem-file'>📄 {rec['filename']}</div>
                <div class='mem-time'>🕐 {rec['timestamp']}</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Session Memory"):
            st.session_state["doc_history"] = []
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
            st.rerun()
    else:
        st.info("No documents this session.")

    if len(all_mem) > 1:
        st.markdown("---")
        st.markdown("**📊 Type Breakdown**")
        tc = {}
        for r in all_mem:
            t = r.get("doc_type", "Unknown")
            tc[t] = tc.get(t, 0) + 1
        for t, c in sorted(tc.items(), key=lambda x: -x[1]):
            bar = "█" * c + "░" * max(0, 5 - c)
            st.markdown(
                f"<div style='font-size:0.75rem;color:#94a3b8;font-family:DM Mono,monospace'>"
                f"{t[:18]}<br><span style='color:#2563eb'>{bar}</span> {c}</div>",
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════════
# 🖥️  MAIN
# ════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"],
    help="Supported: PDF, Word (.docx), plain text (.txt), images (PNG/JPG)"
)

if not uploaded_file:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#374151;'>
        <div style='font-size:3rem;margin-bottom:16px'>📂</div>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:600;color:#6b7280'>
            Upload a document to begin analysis
        </div>
        <div style='font-size:0.8rem;margin-top:8px;color:#4b5563'>
            Supports PDF · Word · Text · Images (OCR)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════
# 🧠 VERBOSE MODE — toggle & helper (ADDED: non-breaking)
# ════════════════════════════════════════════════════════════════
verbose = st.checkbox("🧠 Enable Verbose Mode", value=False)

def log_step(message: str):
    if verbose:
        st.markdown(
            f"<div class='card' style='border-left:3px solid #2563eb;"
            f"background:#0d1526;font-family:DM Mono,monospace;font-size:0.8rem;"
            f"color:#93c5fd;padding:10px 16px;margin:4px 0;border-radius:8px'>"
            f"🔍 {message}</div>",
            unsafe_allow_html=True
        )

# ── Run pipeline ─────────────────────────────────────────────
raw_text = extract_text(uploaded_file)

with st.spinner("🤖 Analyzing document..."):
    R = orchestrator(raw_text, uploaded_file.name)
# ── Verbose agent execution log (ADDED: non-breaking) ────────
log_step("📂 OCR Cleaner Agent → text extracted and cleaned")
log_step(f"🏷️  Classifier Agent → {R['classifier']['doc_type']} detected "
         f"(confidence: {R['classifier']['confidence']}%)")
log_step(f"🔎 Extractor Agent → {len(R['extractor']['points'])} raw field(s) found")
log_step(f"🚨 Anomaly Agent → {len(R['anomaly']['flags'])} validation flag(s) raised")
log_step(f"🧠 Critic Agent → quality score: {R['critic']['quality_score']}% "
         f"| final fields: {len(R['critic']['final'])}"
         + (" | Ollama retry triggered" if R['critic']['retry'] else ""))
log_step(f"🤖 Summarizer Agent → "
         + ("summary generated successfully" if R['summarizer']['ok']
            else f"failed: {R['summarizer'].get('error', 'unknown')}"))
log_step(f"💾 Memory Agent → record saved "
         f"({R['memory']['total']} doc(s) in session)")
log_step("📄 Reporter Agent → full report compiled")
if verbose:
    st.markdown(
        "<div style='margin:6px 0 16px;font-size:0.72rem;"
        "color:#374151;font-family:DM Mono,monospace'>"
        "── end of agent pipeline ──</div>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════
# 📊 RESULT TABS
# ════════════════════════════════════════════════════════════════
tab_over, tab_detail, tab_valid, tab_sum, tab_qa, tab_raw = st.tabs([
    "📊 Overview", "🧾 Key Details", "🚨 Validation",
    "🤖 AI Summary", "💬 Ask Document", "📄 Raw Text"
])


# ── TAB 1: OVERVIEW ──────────────────────────────────────────
with tab_over:
    doc_type   = R["classifier"]["doc_type"]
    confidence = R["classifier"]["confidence"]
    flat_text  = R["flat_text"]
    word_count = len(flat_text.split())
    complexity = ("🟢 Simple" if word_count < 100
                  else "🟡 Moderate" if word_count < 300
                  else "🔴 Complex")
    qs         = R["critic"]["quality_score"]
    qs_color   = "#22c55e" if qs >= 70 else ("#f59e0b" if qs >= 40 else "#ef4444")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Document Type</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#f1f5f9'>{doc_type}</div>
            <div class='metric-sub'>Confidence: {confidence}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Complexity</div>
            <div class='metric-value' style='font-size:1.1rem'>{complexity}</div>
            <div class='metric-sub'>{word_count:,} words</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Extraction Quality</div>
            <div class='gauge-wrap'>
                <div class='gauge-num' style='color:{qs_color}'>{qs}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        fields = len(R["critic"]["final"])
        flags  = len(R["anomaly"]["flags"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Extracted Fields</div>
            <div class='metric-value'>{fields}</div>
            <div class='metric-sub'>{flags} validation flag(s)</div>
        </div>""", unsafe_allow_html=True)

    # Keywords
    st.markdown("<div class='section-header'>🔑 Keywords</div>", unsafe_allow_html=True)
    kws = get_keywords(flat_text, 10)
    pills = "".join(f"<span class='kw-pill'>{w.capitalize()}</span>" for w in kws)
    st.markdown(f"<div style='margin:8px 0 16px'>{pills}</div>", unsafe_allow_html=True)


# ── TAB 2: KEY DETAILS ────────────────────────────────────────
with tab_detail:
    final = R["critic"]["final"]
    if final:
        st.markdown("<div class='section-header'>📌 Extracted Information</div>", unsafe_allow_html=True)
        for lbl, val in final:
            st.markdown(f"""
            <div class='kp-card'>
                <div class='kp-label'>{lbl}</div>
                <div class='kp-value'>{val}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No key details could be extracted from this document.")

    # Critic scored view
    with st.expander("🧠 Extraction Quality Breakdown"):
        for lbl, val, status in R["critic"]["scored"]:
            color = "#22c55e" if status == "✅" else ("#f59e0b" if status == "⚠️" else "#ef4444")
            st.markdown(
                f"<div style='padding:8px 14px;margin:5px 0;background:#12151e;"
                f"border-radius:8px;border-left:3px solid {color};font-size:0.85rem'>"
                f"<span style='color:{color}'>{status}</span> "
                f"<b style='color:#94a3b8'>{lbl}:</b> "
                f"<span style='color:#e2e8f0'>{val}</span></div>",
                unsafe_allow_html=True
            )
        if R["critic"]["retry"]:
            st.info("♻️ Initial extraction quality was low — AI fallback (Ollama) was used.")


# ── TAB 3: VALIDATION ────────────────────────────────────────
with tab_valid:
    st.markdown("<div class='section-header'>🚨 Validation & Anomaly Flags</div>", unsafe_allow_html=True)
    for sev, msg in R["anomaly"]["flags"]:
        css = ("flag-ok" if "OK" in sev
               else "flag-crit" if "Critical" in sev
               else "flag-warn")
        st.markdown(f"<div class='flag-card {css}'><b>{sev}</b> &nbsp; {msg}</div>", unsafe_allow_html=True)


# ── TAB 4: AI SUMMARY ────────────────────────────────────────
with tab_sum:
    st.markdown("<div class='section-header'>🤖 AI-Generated Summary</div>", unsafe_allow_html=True)
    if R["summarizer"]["ok"] and R["summarizer"]["summary"]:
        st.markdown(
            f"<div class='summary-box'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            f"{R['summarizer']['summary']}</div>",
            unsafe_allow_html=True
        )
    else:
        err = R["summarizer"].get("error", "Unknown error")
        st.warning(
            f"⚠️ AI summary unavailable. Ensure Ollama is running with the 'phi' model.\n\nError: {err}"
        )
        st.code("ollama pull phi\nollama run phi", language="bash")

    st.markdown("<hr class='ds-divider'>", unsafe_allow_html=True)

    # Download report
    st.markdown("<div class='section-header'>📤 Download Report</div>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Full Intelligence Report (.txt)",
        data=R["reporter"]["report"],
        file_name=f"docusense_report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )


# ── TAB 5: Q&A ───────────────────────────────────────────────
with tab_qa:
    st.markdown("<div class='section-header'>💬 Ask Questions About This Document</div>", unsafe_allow_html=True)
    st.caption("Powered by Ollama · phi model · answers grounded in document content")

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_doc" not in st.session_state:
        st.session_state["chat_doc"] = ""

    # Reset chat when new file is uploaded
    if st.session_state["chat_doc"] != uploaded_file.name:
        st.session_state["chat_history"] = []
        st.session_state["chat_doc"] = uploaded_file.name

    # Show chat history
    for turn in st.session_state["chat_history"]:
        st.markdown(f"""
        <div class='chat-wrap'>
            <div class='chat-user'>{turn['q']}</div>
        </div>
        <div class='chat-wrap'>
            <div class='chat-ai'><b style='color:#60a5fa'>DocuSense</b><br>{turn['a']}</div>
        </div>""", unsafe_allow_html=True)

    # Suggested questions
    doc_type = R["classifier"]["doc_type"]
    suggestions = {
        "📄 Resume":          ["What are this candidate's top skills?", "What is their highest education?", "What companies have they worked at?"],
        "🧾 Invoice":         ["What is the total amount due?", "Is this invoice overdue?", "Who is the vendor?"],
        "⚠️ Complaint":       ["What is the core complaint?", "Who submitted this complaint?", "What resolution is being requested?"],
        "📑 Legal Document":  ["Who are the parties involved?", "What is the effective date?", "What is the jurisdiction?"],
        "🔬 Research Paper":  ["What is the main hypothesis?", "What are the key findings?", "Who are the authors?"],
    }
    if doc_type in suggestions:
        st.markdown("**💡 Suggested questions:**")
        cols = st.columns(len(suggestions[doc_type]))
        for i, sugg in enumerate(suggestions[doc_type]):
            if cols[i].button(sugg, key=f"sugg_{i}"):
                log_step("💬 QA Agent → processing suggested question via Ollama")
                with st.spinner("Thinking..."):
                    ans = ask_document(sugg, R["flat_text"], doc_type)
                st.session_state["chat_history"].append({"q": sugg, "a": ans})
                st.rerun()

    # Chat input
    user_q = st.chat_input("Ask anything about this document...")
    if user_q:
        log_step("💬 QA Agent → processing question via Ollama")
        with st.spinner("🤖 Analyzing..."):
            answer = ask_document(user_q, R["flat_text"], doc_type)
        st.session_state["chat_history"].append({"q": user_q, "a": answer})
        st.rerun()

    if st.session_state["chat_history"] and st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()


# ── TAB 6: RAW TEXT ──────────────────────────────────────────
with tab_raw:
    st.markdown("<div class='section-header'>📄 Cleaned Document Text</div>", unsafe_allow_html=True)
    st.caption(f"After OCR cleaning · {len(R['clean_text'].split()):,} words · {len(R['clean_text'])} characters")
    st.text_area(
        label="",
        value=R["clean_text"],
        height=400,
        label_visibility="collapsed"
    )
    st.download_button(
        "⬇️ Download Cleaned Text",
        R["clean_text"],
        file_name=f"cleaned_{uploaded_file.name}.txt",
        mime="text/plain"
    )


# ════════════════════════════════════════════════════════════════
# 🤖 CREWAI EXECUTION PANEL (optional, non-breaking)
# Uses real extracted + validated data. Only visible when run.
# ════════════════════════════════════════════════════════════════
st.markdown("<hr class='ds-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>🤖 CrewAI Agent Report</div>",
    unsafe_allow_html=True
)
st.caption(
    "Optional AI layer — runs 5 specialised agents using the real extracted "
    "and validated data from this document. Requires: pip install crewai"
)

if not _CREWAI_AVAILABLE:
    st.info(
        "CrewAI is not installed. Install it to enable this feature:\n"
        "```\npip install crewai\n```"
    )
else:
    if st.button("🚀 Run CrewAI Agent Pipeline", key="run_crewai_btn"):
        with st.spinner("🤖 CrewAI agents running — this may take a moment…"):
            try:
                # Pass REAL pipeline data into the crew
                _crew_obj = _build_crewai_crew(
                    doc_text         = R["flat_text"],
                    doc_type         = R["classifier"]["doc_type"],
                    extracted_points = R["critic"]["final"],
                    validation_flags = R["anomaly"]["flags"],
                )
                _crew_raw_result = _crew_obj.kickoff()
                _crew_formatted  = _format_crewai_output(
                    crew_result      = _crew_raw_result,
                    doc_type         = R["classifier"]["doc_type"],
                    extracted_points = R["critic"]["final"],
                    validation_flags = R["anomaly"]["flags"],
                )
                st.session_state["crewai_output"]     = _crew_formatted
                st.session_state["crewai_output_doc"] = uploaded_file.name
            except Exception as _crew_err:
                # Fallback: render rule-based structured report without LLM
                _crew_formatted = _format_crewai_output(
                    crew_result      = None,
                    doc_type         = R["classifier"]["doc_type"],
                    extracted_points = R["critic"]["final"],
                    validation_flags = R["anomaly"]["flags"],
                )
                st.session_state["crewai_output"]     = _crew_formatted
                st.session_state["crewai_output_doc"] = uploaded_file.name
                st.warning(f"CrewAI LLM unavailable ({_crew_err}). "
                           "Showing rule-based structured report.")

    # Clear output when document changes
    if st.session_state.get("crewai_output_doc") != uploaded_file.name:
        st.session_state["crewai_output"]     = ""
        st.session_state["crewai_output_doc"] = ""

    if st.session_state.get("crewai_output"):
        # ── Structured visual completion box ────────────────
        _co = st.session_state["crewai_output"]
        st.markdown(
            f"<div style='background:#0d1526;border:2px solid #2563eb30;"
            f"border-radius:14px;padding:22px 24px;font-family:DM Mono,monospace;"
            f"font-size:0.82rem;color:#93c5fd;line-height:1.85;"
            f"white-space:pre-wrap'>{_co}</div>",
            unsafe_allow_html=True
        )
        st.download_button(
            "⬇️ Download CrewAI Report",
            _co,
            file_name=f"crewai_report_{uploaded_file.name}.txt",
            mime="text/plain",
            key="dl_crewai_btn"
        )
        if st.button("🗑️ Clear Report", key="clear_crewai_btn"):
            st.session_state["crewai_output"]     = ""
            st.session_state["crewai_output_doc"] = ""
            st.rerun()


# ════════════════════════════════════════════════════════════════
# ✨ ADDED FEATURES (non-breaking — appended below all existing code)
# Features: 1) Multi-doc  2) Payment Decision  3) Next Steps
#           4) Why Decision  5) Smart Alerts  6) Auto Response
# ════════════════════════════════════════════════════════════════

# ── Convenience: alias primary-document results ──────────────
_doc_type   = R["classifier"]["doc_type"]
_final_pts  = R["critic"]["final"]           # list of (label, value) tuples
_flags      = R["anomaly"]["flags"]          # list of (severity, message) tuples
_flat_text  = R["flat_text"]
_labels     = {lbl for lbl, _ in _final_pts}
_vals       = {lbl: val for lbl, val in _final_pts}

# ── FEATURE 5: SMART ALERTS ──────────────────────────────────
# (rendered before payment decision so alerts appear at the top
#  of the new section)
st.markdown("<hr class=\'ds-divider\'>", unsafe_allow_html=True)
st.markdown("### 🔔 Smart Alerts", unsafe_allow_html=False)

_alerts = []

# Invoice alerts
if "Invoice" in _doc_type:
    for _sev, _msg in _flags:
        if "OVERDUE" in _msg:
            _alerts.append(("🔴", "Overdue Invoice", _msg))
        elif "due in" in _msg.lower():
            # Extract days if possible
            import re as _re
            _days_m = _re.search(r"due in (\d+) day", _msg, _re.IGNORECASE)
            if _days_m and int(_days_m.group(1)) <= 3:
                _alerts.append(("🟡", "Payment Due Very Soon",
                                f"Payment due in {_days_m.group(1)} day(s). Act promptly."))
            elif _days_m:
                _alerts.append(("🟢", "Upcoming Payment",
                                f"Payment due in {_days_m.group(1)} day(s)."))
    if "Total Amount" not in _labels:
        _alerts.append(("🟡", "Missing Total", "Invoice total amount not found — verify before payment."))
    if "Vendor / From" not in _labels:
        _alerts.append(("🟡", "Unknown Vendor", "Vendor name missing — cannot confirm payee identity."))

# Complaint alerts
if "Complaint" in _doc_type:
    _alerts.append(("🚨", "High Priority Complaint",
                    "A complaint document has been detected. Requires immediate attention."))
    if len(_flat_text.split()) < 50:
        _alerts.append(("🟡", "Brief Complaint",
                        "Complaint text is very short — may lack sufficient detail for resolution."))

# Resume alerts
if "Resume" in _doc_type:
    if "Email" not in _labels:
        _alerts.append(("🟡", "No Contact Email", "Candidate email address is missing from the resume."))
    if "Technical Skills" not in _labels:
        _alerts.append(("🟡", "Skills Section Missing", "No technical skills section detected in this resume."))

# Legal alerts
if "Legal" in _doc_type:
    _alerts.append(("🔵", "Legal Review Required",
                    "Legal document detected. Ensure review by a qualified legal professional."))

# Generic low-quality alert
if R["critic"]["quality_score"] < 40:
    _alerts.append(("🔴", "Low Extraction Quality",
                    f"Extraction quality is only {R['critic']['quality_score']}%. "
                    "Document may be poorly formatted or scanned."))

if _alerts:
    for _icon, _title, _detail in _alerts:
        _bg  = ("#1a0a0a" if _icon in ("🔴", "🚨")
                else "#1a1400" if _icon == "🟡"
                else "#001a0a" if _icon == "🟢"
                else "#0a0a1a")
        _bdr = ("#ef4444" if _icon in ("🔴", "🚨")
                else "#f59e0b" if _icon == "🟡"
                else "#22c55e" if _icon == "🟢"
                else "#3b82f6")
        st.markdown(
            f"<div style=\'background:{_bg};border-left:4px solid {_bdr};"
            f"border-radius:10px;padding:14px 18px;margin:7px 0;\'>"
            f"<span style=\'font-size:1rem\'>{_icon}</span> "
            f"<b style=\'color:#f1f5f9\'>{_title}</b>"
            f"<div style=\'color:#94a3b8;font-size:0.83rem;margin-top:4px\'>{_detail}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        "<div class=\'flag-card flag-ok\'>🟢 &nbsp; <b>No alerts.</b> "
        "Document passed all smart checks.</div>",
        unsafe_allow_html=True
    )

# ── FEATURE 1–3: INVOICE-SPECIFIC PANEL ─────────────────────
if "Invoice" in _doc_type:

    st.markdown("<hr class=\'ds-divider\'>", unsafe_allow_html=True)

    _col_pay, _col_why = st.columns([1, 1])

    # ── FEATURE 2: PAYMENT DECISION AGENT ────────────────────
    with _col_pay:
        st.markdown("### 💳 Payment Decision")

        _has_vendor = "Vendor / From" in _labels
        _has_total  = "Total Amount"  in _labels
        _has_due    = "Payment Due"   in _labels
        _is_overdue = any("OVERDUE" in m for _, m in _flags)

        # ── Refined decision: overdue + field checks ─────────
        _has_tax          = "Tax"          in _labels
        _pd_total_mismatch = any("does not match" in _m for _, _m in _flags)
        _pd_any_critical_missing = (
            not _has_vendor or not _has_total or not _has_tax or _pd_total_mismatch
        )

        if not _has_vendor or not _has_total:
            _decision      = "❌ Do Not Pay"
            _decision_color = "#ef4444"
            _decision_bg    = "#1a0505"
            _decision_bdr   = "#ef4444"
        elif _is_overdue and _pd_any_critical_missing:
            _decision      = "⚠️ Review Before Payment"
            _decision_color = "#f59e0b"
            _decision_bg    = "#1a1000"
            _decision_bdr   = "#f59e0b"
        elif _is_overdue and not _pd_any_critical_missing:
            _decision      = "🔴 Pay Immediately — Overdue"
            _decision_color = "#f87171"
            _decision_bg    = "#1a0505"
            _decision_bdr   = "#ef4444"
        elif not _has_due or _pd_total_mismatch:
            _decision      = "⚠️ Review Required"
            _decision_color = "#f59e0b"
            _decision_bg    = "#1a1000"
            _decision_bdr   = "#f59e0b"
        else:
            _decision      = "✔ Safe to Pay"
            _decision_color = "#22c55e"
            _decision_bg    = "#011a05"
            _decision_bdr   = "#22c55e"

        st.markdown(
            f"<div style=\'background:{_decision_bg};border:2px solid {_decision_bdr};"
            f"border-radius:14px;padding:22px 24px;text-align:center;\'>"
            f"<div style=\'font-family:Syne,sans-serif;font-size:1.5rem;"
            f"font-weight:800;color:{_decision_color}\'>{_decision}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ── FEATURE 4: NEXT STEPS ─────────────────────────────
        st.markdown("### 📌 Next Steps")
        _next_steps = []

        if not _has_vendor:
            _next_steps.append("🔍 Verify vendor details — payee identity is unconfirmed")
        if not _has_total:
            _next_steps.append("💰 Obtain the correct total amount from the vendor")
        if _pd_total_mismatch:
            _next_steps.append("🔁 Recalculate total amount — line item sum does not match stated total")
        if not _has_due:
            _next_steps.append("📅 Confirm payment due date with the issuing party")
        if _is_overdue:
            _next_steps.append("⚡ Process payment immediately — invoice is overdue and accruing risk")
        if "Tax" not in _labels:
            _next_steps.append("🧾 Verify tax details before payment — tax amount is unconfirmed")
        if "Invoice Number" not in _labels:
            _next_steps.append("🔢 Request invoice number for audit and record-keeping")
        if not _next_steps:
            _next_steps.append("✅ All checks passed — proceed with payment and record the transaction")
            _next_steps.append("📁 Archive invoice in the accounts payable system after confirmation")

        for _step in _next_steps:
            st.markdown(
                f"<div class=\'kp-card\'>"
                f"<div class=\'kp-value\'>{_step}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # ── FEATURE 3: WHY THIS DECISION ─────────────────────────
    with _col_why:
        st.markdown("### 🧠 Why This Decision")

        _reasons = []
        if not _has_vendor:
            _reasons.append(("❌", "Vendor / From field is missing",
                             "Cannot identify who to pay — payment blocked for safety."))
        if not _has_total:
            _reasons.append(("❌", "Total Amount field is missing",
                             "No confirmed amount found — cannot authorise payment."))
        if not _has_due:
            _reasons.append(("⚠️", "Payment Due Date not found",
                             "Due date is absent — unable to assess urgency or lateness."))
        if _is_overdue:
            _reasons.append(("🔴", "Invoice is past its due date",
                             "The payment deadline has already passed."))
        if "Tax" not in _labels:
            _reasons.append(("⚠️", "Tax information not detected",
                             "Tax details may affect the final payable amount."))
        if "Invoice Number" not in _labels:
            _reasons.append(("⚠️", "Invoice Number not found",
                             "Missing reference number makes tracking difficult."))
        if _has_vendor and _has_total and _has_due and not _is_overdue:
            _reasons.append(("✅", "All critical fields present",
                             "Vendor, total, and due date are all confirmed."))
            _reasons.append(("✅", "No critical anomalies detected",
                             "Validation checks passed — invoice appears legitimate."))

        for _icon, _title, _expl in _reasons:
            _rc = ("#22c55e" if _icon == "✅"
                   else "#ef4444" if _icon in ("❌", "🔴")
                   else "#f59e0b")
            st.markdown(
                f"<div style=\'background:#12151e;border:1px solid #1e2130;"
                f"border-left:4px solid {_rc};border-radius:10px;"
                f"padding:12px 16px;margin:6px 0;\'>"
                f"<b style=\'color:{_rc}\'>{_icon} {_title}</b>"
                f"<div style=\'color:#94a3b8;font-size:0.8rem;margin-top:4px\'>"
                f"{_expl}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── FEATURE 3 extended: Why This Decision for non-invoice docs ──
if "Invoice" not in _doc_type:
    st.markdown("<hr class=\'ds-divider\'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Why This Decision")
    _gen_reasons = []

    if "Resume" in _doc_type:
        if "Candidate Name" not in _labels or _vals.get("Candidate Name") == "Not detected":
            _gen_reasons.append(("⚠️", "Candidate name not detected",
                                 "Name could not be auto-identified from the resume layout."))
        if "Email" not in _labels:
            _gen_reasons.append(("⚠️", "No email found",
                                 "Candidate email is absent — follow-up may be difficult."))
        if "Technical Skills" in _labels:
            _gen_reasons.append(("✅", "Skills section found",
                                 "Technical skills were successfully extracted."))
        if "Education" in _labels:
            _gen_reasons.append(("✅", "Education section found",
                                 "Education background is present in the document."))

    if "Complaint" in _doc_type:
        _gen_reasons.append(("🚨", "Complaint requires action",
                             "This document contains a complaint that needs a timely response."))
        if "Subject" in _labels:
            _gen_reasons.append(("✅", "Subject clearly stated",
                                 "The complaint subject has been identified."))

    if "Legal" in _doc_type:
        _gen_reasons.append(("🔵", "Legal review mandatory",
                             "Legal documents must be reviewed by authorised personnel."))

    if "Research" in _doc_type:
        if "Abstract" in _labels:
            _gen_reasons.append(("✅", "Abstract extracted",
                                 "Research paper abstract was found and extracted."))

    if not _gen_reasons:
        _gen_reasons.append(("ℹ️", "General document",
                             "No specific decision logic applies to this document type."))

    for _icon, _title, _expl in _gen_reasons:
        _rc = ("#22c55e" if _icon == "✅"
               else "#ef4444" if _icon in ("❌", "🔴", "🚨")
               else "#3b82f6" if _icon == "🔵"
               else "#f59e0b")
        st.markdown(
            f"<div style=\'background:#12151e;border:1px solid #1e2130;"
            f"border-left:4px solid {_rc};border-radius:10px;"
            f"padding:12px 16px;margin:6px 0;\'>"
            f"<b style=\'color:{_rc}\'>{_icon} {_title}</b>"
            f"<div style=\'color:#94a3b8;font-size:0.8rem;margin-top:4px\'>"
            f"{_expl}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── FEATURE 6: AUTO RESPONSE GENERATOR (OLLAMA) ──────────────
st.markdown("<hr class=\'ds-divider\'>", unsafe_allow_html=True)
st.markdown("### 🤖 Suggested Response")
st.caption("AI-generated reply based on document content · Powered by Ollama (phi)")

_STRICT_PREFIX = (
    "Generate a professional response strictly based on the following document "
    "content. Do not add external information. Keep it concise and relevant. "
    "2-3 sentences only. Plain text, no markdown.\n\n"
)
_response_prompt_map = {
    "Complaint":        (
        _STRICT_PREFIX
        + "Role: professional customer service agent responding to the complaint below.\n"
        + "Acknowledge the issue, apologise briefly, and state the next step.\n\nComplaint:\n"
    ),
    "Invoice":          (
        _STRICT_PREFIX
        + "Role: accounts payable officer. Acknowledge receipt and confirm "
        + "the key invoice details found in the text below. "
        + "Do not invent amounts or dates not present.\n\nInvoice details:\n"
    ),
    "Resume":           (
        _STRICT_PREFIX
        + "Role: HR recruiter. Acknowledge receipt of the resume below and "
        + "mention one specific detail (name or skill) found in the text.\n\nResume:\n"
    ),
    "Legal Document":   (
        _STRICT_PREFIX
        + "Role: legal assistant. Acknowledge receipt of the legal document below "
        + "and confirm it will be reviewed. Mention the document type if clear.\n\nDocument:\n"
    ),
    "Research Paper":   (
        _STRICT_PREFIX
        + "Role: academic reviewer. Acknowledge receipt of the paper below and "
        + "mention the topic if identifiable from the text.\n\nPaper:\n"
    ),
}

# Pick best matching prompt
_resp_prompt = None
for _key, _prompt in _response_prompt_map.items():
    if _key in _doc_type:
        _resp_prompt = _prompt
        break
if not _resp_prompt:
    _resp_prompt = (
        _STRICT_PREFIX
        + "Role: professional assistant. Acknowledge the document below "
        + "and state one specific detail found in the text.\n\nDocument:\n"
    )

if "auto_response" not in st.session_state:
    st.session_state["auto_response"] = ""
if "auto_response_doc" not in st.session_state:
    st.session_state["auto_response_doc"] = ""

# Auto-clear response when document changes
if st.session_state["auto_response_doc"] != uploaded_file.name:
    st.session_state["auto_response"] = ""
    st.session_state["auto_response_doc"] = uploaded_file.name

if st.button("✨ Generate Suggested Response", key="gen_response_btn"):
    with st.spinner("🤖 Generating response…"):
        try:
            _ar = ollama.chat(
                model="phi",
                messages=[{
                    "role": "user",
                    "content": _resp_prompt + _flat_text[:300]
                }]
            )
            _ar_text = _ar["message"]["content"].strip()
            import re as _re2
            _ar_text = _re2.sub(r"[^\x00-\x7E\n]", "", _ar_text).strip()
            st.session_state["auto_response"] = _ar_text
        except Exception as _ar_e:
            st.session_state["auto_response"] = (
                f"Could not generate response: {_ar_e}\n"
                "Ensure Ollama is running: ollama run phi"
            )

if st.session_state["auto_response"]:
    _ar_display = st.session_state["auto_response"]
    st.markdown(
        f"<div class='summary-box'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        f"{_ar_display}</div>",
        unsafe_allow_html=True
    )
    _copy_col, _clear_col = st.columns([3, 1])
    with _copy_col:
        st.text_area(
            "Copy response:",
            value=st.session_state["auto_response"],
            height=120,
            key="auto_response_textarea",
            label_visibility="collapsed"
        )
    with _clear_col:
        if st.button("🗑️ Clear", key="clear_response_btn"):
            st.session_state["auto_response"] = ""
            st.rerun()
else:
    st.markdown(
        "<div style=\'color:#4b5563;font-size:0.85rem;padding:12px 0\'>"
        "Click the button above to generate a context-aware suggested response.</div>",
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════
# ✨ ADDED AGENTS — Round 2 (non-breaking, appended)
# 1) Resume Candidate Fit Decision Agent
# 2) Invoice Risk Score Agent
# 3) Complaint Priority Decision Agent
# ════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# AGENT A: RESUME — CANDIDATE FIT DECISION
# ─────────────────────────────────────────────────────────────────
if "Resume" in _doc_type:

    st.markdown("<hr class='ds-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>🎯 Candidate Fit Decision</div>",
        unsafe_allow_html=True
    )

    # ── Gather signals from extracted fields ──────────────────
    _cf_has_skills  = "Technical Skills" in _labels
    _cf_has_edu     = "Education"         in _labels
    _cf_has_exp     = ("Latest Experience" in _labels or "Experience" in _labels)
    _cf_has_name    = (
        "Candidate Name" in _labels
        and _vals.get("Candidate Name", "Not detected") != "Not detected"
    )

    # GPA extraction — look in _vals first, then raw text
    _cf_gpa_val  = None
    _cf_gpa_str  = _vals.get("GPA", "")
    if _cf_gpa_str:
        import re as _re_cf
        _gpa_m = _re_cf.search(r"([\d]+\.[\d]+|[\d]+)", _cf_gpa_str)
        if _gpa_m:
            try:
                _cf_gpa_val = float(_gpa_m.group(1))
            except ValueError:
                pass
    if _cf_gpa_val is None:
        import re as _re_cf2
        _gpa_raw = _re_cf2.search(r"GPA\s*[:\-]?\s*([\d]+\.[\d]+|[\d]+)", _flat_text, _re_cf2.IGNORECASE)
        if _gpa_raw:
            try:
                _cf_gpa_val = float(_gpa_raw.group(1))
            except ValueError:
                pass

    _cf_gpa_ok   = (_cf_gpa_val is not None and _cf_gpa_val >= 7.0)
    _cf_gpa_low  = (_cf_gpa_val is not None and _cf_gpa_val < 7.0)

    # ── Decision logic (rule-based, fully explainable) ────────
    _cf_missing = []
    if not _cf_has_skills: _cf_missing.append("Technical Skills")
    if not _cf_has_edu:    _cf_missing.append("Education")
    if not _cf_has_exp:    _cf_missing.append("Experience")

    if not _cf_has_skills or not _cf_has_edu:
        _cf_decision      = "❌ Not Suitable"
        _cf_decision_color = "#ef4444"
        _cf_decision_bg    = "#1a0505"
        _cf_decision_bdr   = "#ef4444"
    elif len(_cf_missing) >= 1 or _cf_gpa_low:
        _cf_decision      = "⚠️ Average Candidate"
        _cf_decision_color = "#f59e0b"
        _cf_decision_bg    = "#1a1000"
        _cf_decision_bdr   = "#f59e0b"
    else:
        _cf_decision      = "✔ Strong Candidate"
        _cf_decision_color = "#22c55e"
        _cf_decision_bg    = "#011a05"
        _cf_decision_bdr   = "#22c55e"

    # ── Layout: decision card + why side by side ──────────────
    _cf_col1, _cf_col2 = st.columns([1, 1])

    with _cf_col1:
        st.markdown(
            f"<div style='background:{_cf_decision_bg};"
            f"border:2px solid {_cf_decision_bdr};"
            f"border-radius:14px;padding:24px;text-align:center;margin-bottom:12px'>"
            f"<div style='font-family:Syne,sans-serif;font-size:1.4rem;"
            f"font-weight:800;color:{_cf_decision_color}'>{_cf_decision}</div>"
            f"<div style='color:#6b7280;font-size:0.78rem;margin-top:8px;"
            f"font-family:DM Mono,monospace'>Based on {4 - len(_cf_missing)} / 4 "
            f"key signals detected</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Signal checklist
        for _sig_lbl, _sig_ok in [
            ("Technical Skills", _cf_has_skills),
            ("Education",        _cf_has_edu),
            ("Experience",       _cf_has_exp),
            ("Candidate Name",   _cf_has_name),
        ]:
            _sig_icon  = "✅" if _sig_ok else "❌"
            _sig_color = "#22c55e" if _sig_ok else "#ef4444"
            st.markdown(
                f"<div class='kp-card'>"
                f"<div class='kp-label'>{_sig_lbl}</div>"
                f"<div class='kp-value' style='color:{_sig_color}'>{_sig_icon} "
                f"{'Detected' if _sig_ok else 'Not found'}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # GPA card
        if _cf_gpa_val is not None:
            _gpa_c = "#22c55e" if _cf_gpa_ok else "#f59e0b"
            st.markdown(
                f"<div class='kp-card'>"
                f"<div class='kp-label'>GPA</div>"
                f"<div class='kp-value' style='color:{_gpa_c}'>"
                f"{_cf_gpa_val} &nbsp; {'✅ Meets threshold (≥7.0)' if _cf_gpa_ok else '⚠️ Below threshold (7.0)'}"
                f"</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='kp-card'>"
                "<div class='kp-label'>GPA</div>"
                "<div class='kp-value' style='color:#6b7280'>Not mentioned in resume</div>"
                "</div>",
                unsafe_allow_html=True
            )

    with _cf_col2:
        st.markdown(
            "<div class='section-header' style='font-size:0.95rem'>🧠 Why This Fit?</div>",
            unsafe_allow_html=True
        )

        _cf_reasons = []

        if _cf_has_skills:
            _cf_reasons.append(("✅", "#22c55e", "Skills detected",
                                 "Technical skills section is present — candidate has demonstrable competencies."))
        else:
            _cf_reasons.append(("❌", "#ef4444", "No skills section",
                                 "Technical skills are absent — core requirement not met."))

        if _cf_has_edu:
            _cf_reasons.append(("✅", "#22c55e", "Education verified",
                                 "Education background is present and extractable."))
        else:
            _cf_reasons.append(("❌", "#ef4444", "Education missing",
                                 "No education section detected — key qualification unverifiable."))

        if _cf_has_exp:
            _cf_reasons.append(("✅", "#22c55e", "Experience present",
                                 "Work experience section was found — candidate has prior roles."))
        else:
            _cf_reasons.append(("⚠️", "#f59e0b", "Experience not found",
                                 "Experience section is missing — may be a fresher or poorly formatted."))

        if _cf_gpa_ok:
            _cf_reasons.append(("✅", "#22c55e", f"GPA {_cf_gpa_val} meets threshold",
                                 "GPA is at or above 7.0 — academic performance is satisfactory."))
        elif _cf_gpa_low:
            _cf_reasons.append(("⚠️", "#f59e0b", f"GPA {_cf_gpa_val} is below 7.0",
                                 "Academic score is below the preferred threshold of 7.0."))
        else:
            _cf_reasons.append(("ℹ️", "#6b7280", "GPA not mentioned",
                                 "GPA was not found in the resume — cannot evaluate academic score."))

        if not _cf_has_name:
            _cf_reasons.append(("⚠️", "#f59e0b", "Candidate name unclear",
                                 "Name could not be reliably extracted — manual verification needed."))

        for _r_icon, _r_color, _r_title, _r_detail in _cf_reasons:
            st.markdown(
                f"<div style='background:#12151e;border:1px solid #1e2130;"
                f"border-left:4px solid {_r_color};border-radius:10px;"
                f"padding:12px 16px;margin:6px 0'>"
                f"<b style='color:{_r_color}'>{_r_icon} {_r_title}</b>"
                f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:4px'>{_r_detail}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Suggested actions
        st.markdown(
            "<div class='section-header' style='font-size:0.9rem;margin-top:16px'>"
            "📌 Suggested Actions</div>",
            unsafe_allow_html=True
        )
        _cf_actions = []
        if not _cf_has_skills:
            _cf_actions.append("Ask candidate to provide a detailed skills inventory")
        if not _cf_has_edu:
            _cf_actions.append("Request educational certificates for verification")
        if not _cf_has_exp:
            _cf_actions.append("Conduct an interview to assess practical experience")
        if _cf_gpa_low:
            _cf_actions.append("Ask candidate to explain academic performance")
        if not _cf_has_name:
            _cf_actions.append("Confirm candidate identity — name not auto-detected")
        if not _cf_actions:
            _cf_actions.append("Proceed to screening interview — profile looks complete")
            _cf_actions.append("Cross-check LinkedIn profile before final shortlisting")

        for _act in _cf_actions:
            st.markdown(
                f"<div class='kp-card'>"
                f"<div class='kp-value'>➤ {_act}</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────
# AGENT B: INVOICE — RISK SCORE AGENT
# ─────────────────────────────────────────────────────────────────
if "Invoice" in _doc_type:

    st.markdown("<hr class='ds-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>📊 Invoice Risk Score</div>",
        unsafe_allow_html=True
    )

    # ── Gather risk signals ───────────────────────────────────
    _rs_missing_vendor  = "Vendor / From"  not in _labels
    _rs_missing_total   = "Total Amount"   not in _labels
    _rs_missing_due     = "Payment Due"    not in _labels
    _rs_missing_inv_num = "Invoice Number" not in _labels
    _rs_missing_tax     = "Tax"            not in _labels
    _rs_is_overdue      = any("OVERDUE" in _m for _, _m in _flags)
    _rs_total_mismatch  = any("does not match" in _m for _, _m in _flags)

    # ── Score calculation ─────────────────────────────────────
    # Critical fields: vendor, total, due date — each worth 30 pts
    # Supporting fields: inv number, tax — each worth 10 pts
    # Penalties: overdue -15, total mismatch -20
    _rs_score = 100
    _rs_deductions = []

    if _rs_missing_vendor:
        _rs_score -= 30
        _rs_deductions.append(("❌", "#ef4444", "Vendor missing", "-30 pts",
                               "Payee identity unknown — critical risk factor."))
    if _rs_missing_total:
        _rs_score -= 30
        _rs_deductions.append(("❌", "#ef4444", "Total Amount missing", "-30 pts",
                               "No confirmed amount — payment cannot be authorised."))
    if _rs_missing_due:
        _rs_score -= 20
        _rs_deductions.append(("⚠️", "#f59e0b", "Payment Due Date missing", "-20 pts",
                               "Cannot assess urgency or overdue status."))
    if _rs_missing_inv_num:
        _rs_score -= 10
        _rs_deductions.append(("⚠️", "#f59e0b", "Invoice Number missing", "-10 pts",
                               "No reference number — tracking and audit trail affected."))
    if _rs_missing_tax:
        _rs_score -= 10
        _rs_deductions.append(("⚠️", "#f59e0b", "Tax details missing", "-10 pts",
                               "Tax cannot be verified — final payable amount uncertain."))
    if _rs_is_overdue:
        _rs_score -= 15
        _rs_deductions.append(("🔴", "#ef4444", "Invoice is OVERDUE", "-15 pts",
                               "Payment deadline has passed — financial/legal risk increases daily."))
    if _rs_total_mismatch:
        _rs_score -= 20
        _rs_deductions.append(("🔴", "#ef4444", "Total vs line items mismatch", "-20 pts",
                               "Calculated subtotals do not match stated total — possible error or fraud."))

    _rs_score = max(_rs_score, 0)

    # ── Risk band ─────────────────────────────────────────────
    if _rs_score >= 75:
        _rs_band       = "🟢 Low Risk"
        _rs_band_color = "#22c55e"
        _rs_band_bg    = "#011a05"
        _rs_band_bdr   = "#22c55e"
        _rs_advice     = "Invoice appears complete and valid. Safe to proceed with payment approval."
    elif _rs_score >= 40:
        _rs_band       = "🟡 Medium Risk"
        _rs_band_color = "#f59e0b"
        _rs_band_bg    = "#1a1000"
        _rs_band_bdr   = "#f59e0b"
        _rs_advice     = "One or more non-critical fields are missing. Review before processing payment."
    else:
        _rs_band       = "🔴 High Risk"
        _rs_band_color = "#ef4444"
        _rs_band_bg    = "#1a0505"
        _rs_band_bdr   = "#ef4444"
        _rs_advice     = "Critical fields missing or data mismatch detected. Do NOT process until resolved."

    # ── Layout ────────────────────────────────────────────────
    _rs_col1, _rs_col2 = st.columns([1, 1])

    with _rs_col1:
        # Score gauge card
        st.markdown(
            f"<div style='background:{_rs_band_bg};border:2px solid {_rs_band_bdr};"
            f"border-radius:14px;padding:24px;text-align:center;margin-bottom:12px'>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
            f"color:#6b7280;letter-spacing:0.08em;text-transform:uppercase'>"
            f"Risk Score</div>"
            f"<div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;"
            f"color:{_rs_band_color};line-height:1.1'>{_rs_score}</div>"
            f"<div style='font-size:0.72rem;color:#6b7280'>out of 100</div>"
            f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;"
            f"color:{_rs_band_color};margin-top:10px'>{_rs_band}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        # Advice card
        st.markdown(
            f"<div style='background:#12151e;border:1px solid {_rs_band_bdr}30;"
            f"border-radius:10px;padding:14px 16px;font-size:0.85rem;color:#cbd5e1'>"
            f"💡 {_rs_advice}</div>",
            unsafe_allow_html=True
        )

        # Field presence summary
        st.markdown(
            "<div class='section-header' style='font-size:0.9rem;margin-top:16px'>"
            "Field Presence Check</div>",
            unsafe_allow_html=True
        )
        for _fld, _present in [
            ("Vendor / From",  not _rs_missing_vendor),
            ("Total Amount",   not _rs_missing_total),
            ("Payment Due",    not _rs_missing_due),
            ("Invoice Number", not _rs_missing_inv_num),
            ("Tax",            not _rs_missing_tax),
        ]:
            _fc = "#22c55e" if _present else "#ef4444"
            _fi = "✅" if _present else "❌"
            st.markdown(
                f"<div class='kp-card'>"
                f"<div class='kp-label'>{_fld}</div>"
                f"<div class='kp-value' style='color:{_fc}'>{_fi} "
                f"{'Present' if _present else 'Missing'}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    with _rs_col2:
        st.markdown(
            "<div class='section-header' style='font-size:0.95rem'>"
            "🔍 Risk Breakdown</div>",
            unsafe_allow_html=True
        )

        if _rs_deductions:
            for _d_icon, _d_color, _d_title, _d_pts, _d_detail in _rs_deductions:
                st.markdown(
                    f"<div style='background:#12151e;border:1px solid #1e2130;"
                    f"border-left:4px solid {_d_color};border-radius:10px;"
                    f"padding:12px 16px;margin:6px 0'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<b style='color:{_d_color}'>{_d_icon} {_d_title}</b>"
                    f"<span style='font-family:DM Mono,monospace;font-size:0.78rem;"
                    f"color:{_d_color};background:{_d_color}20;padding:2px 8px;"
                    f"border-radius:10px'>{_d_pts}</span>"
                    f"</div>"
                    f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:4px'>{_d_detail}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div class='flag-card flag-ok'>"
                "✅ &nbsp; No risk factors detected — invoice fields are complete and consistent."
                "</div>",
                unsafe_allow_html=True
            )

        # Suggested remediation steps
        st.markdown(
            "<div class='section-header' style='font-size:0.9rem;margin-top:16px'>"
            "📌 Remediation Steps</div>",
            unsafe_allow_html=True
        )
        _rs_steps = []
        if _rs_missing_vendor:
            _rs_steps.append("Contact issuer to confirm vendor identity and business registration")
        if _rs_missing_total:
            _rs_steps.append("Request a corrected invoice with the final total clearly stated")
        if _rs_missing_due:
            _rs_steps.append("Ask the vendor to specify a payment due date in writing")
        if _rs_total_mismatch:
            _rs_steps.append("Request line-by-line breakdown from vendor to resolve total discrepancy")
        if _rs_is_overdue:
            _rs_steps.append("Escalate to accounts payable immediately — overdue penalties may apply")
        if _rs_missing_inv_num:
            _rs_steps.append("Request a formal invoice number for audit and record-keeping purposes")
        if _rs_missing_tax:
            _rs_steps.append("Verify applicable tax rate with vendor or finance team")
        if not _rs_steps:
            _rs_steps.append("No remediation needed — proceed to payment authorisation")
            _rs_steps.append("File invoice in accounts payable system with reference number")

        for _rs_s in _rs_steps:
            st.markdown(
                f"<div class='kp-card'><div class='kp-value'>➤ {_rs_s}</div></div>",
                unsafe_allow_html=True
            )

    # ── FIX 3: Risk Breakdown — full-width explanation panel ──
    st.markdown(
        "<div class='section-header' style='margin-top:20px'>📊 Risk Breakdown</div>",
        unsafe_allow_html=True
    )
    st.caption("Detailed explanation of how the risk score was calculated — each factor and its contribution.")

    # Build signed contribution rows
    _rb_rows = []

    # Positive contributions (fields that ARE present — reduce risk)
    if not _rs_missing_vendor:
        _rb_rows.append(("✅", "#22c55e", "Vendor / From present",
                         "−0 pts", "Payee is identified — no risk added."))
    if not _rs_missing_total:
        _rb_rows.append(("✅", "#22c55e", "Total Amount present",
                         "−0 pts", "Payment amount is confirmed — no risk added."))
    if not _rs_missing_due:
        _rb_rows.append(("✅", "#22c55e", "Payment Due Date present",
                         "−0 pts", "Due date is known — urgency can be assessed."))
    if not _rs_missing_inv_num:
        _rb_rows.append(("✅", "#22c55e", "Invoice Number present",
                         "−0 pts", "Reference number available for tracking."))
    if not _rs_missing_tax:
        _rb_rows.append(("✅", "#22c55e", "Tax details present",
                         "−0 pts", "Tax is confirmed — final payable amount is clear."))

    # Negative contributions (issues that RAISED risk)
    if _rs_missing_vendor:
        _rb_rows.append(("❌", "#ef4444", "Vendor / From missing",
                         "+30 risk pts", "Cannot confirm who to pay — highest risk category."))
    if _rs_missing_total:
        _rb_rows.append(("❌", "#ef4444", "Total Amount missing",
                         "+30 risk pts", "No amount to authorise — payment is impossible without this."))
    if _rs_missing_due:
        _rb_rows.append(("⚠️", "#f59e0b", "Payment Due Date missing",
                         "+20 risk pts", "Urgency and overdue status cannot be determined."))
    if _rs_missing_inv_num:
        _rb_rows.append(("⚠️", "#f59e0b", "Invoice Number missing",
                         "+10 risk pts", "No reference for audit trail or dispute resolution."))
    if _rs_missing_tax:
        _rb_rows.append(("⚠️", "#f59e0b", "Tax details missing",
                         "+10 risk pts", "Final payable amount may be incorrect without tax."))
    if _rs_is_overdue:
        _rb_rows.append(("🔴", "#ef4444", "Invoice is OVERDUE",
                         "+15 risk pts", "Payment deadline has passed — penalties or legal risk may apply."))
    if _rs_total_mismatch:
        _rb_rows.append(("🔴", "#ef4444", "Total vs line items mismatch",
                         "+20 risk pts", "Stated total does not match sum of line items — possible error or fraud."))

    if not _rb_rows:
        st.markdown(
            "<div class='flag-card flag-ok'>"
            "✅ &nbsp; All fields verified and no anomalies detected. Risk score: 100/100.</div>",
            unsafe_allow_html=True
        )
    else:
        # Display as a clean 3-column table-style layout using cards
        for _rb_icon, _rb_color, _rb_factor, _rb_pts, _rb_detail in _rb_rows:
            _pts_bg = f"{_rb_color}18"
            st.markdown(
                f"<div style='background:#12151e;border:1px solid #1e2130;"
                f"border-left:4px solid {_rb_color};border-radius:10px;"
                f"padding:11px 16px;margin:5px 0;"
                f"display:flex;align-items:flex-start;gap:12px'>"
                f"<div style='min-width:180px'>"
                f"<b style='color:{_rb_color};font-size:0.88rem'>"
                f"{_rb_icon} {_rb_factor}</b></div>"
                f"<div style='flex:1;color:#94a3b8;font-size:0.8rem'>{_rb_detail}</div>"
                f"<div style='min-width:110px;text-align:right'>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.78rem;"
                f"color:{_rb_color};background:{_pts_bg};"
                f"padding:3px 10px;border-radius:12px;white-space:nowrap'>"
                f"{_rb_pts}</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Score summary footer
        _rb_pts_total = (
            (_rs_missing_vendor  * 30) + (_rs_missing_total  * 30) +
            (_rs_missing_due     * 20) + (_rs_missing_inv_num * 10) +
            (_rs_missing_tax     * 10) + (_rs_is_overdue      * 15) +
            (_rs_total_mismatch  * 20)
        )
        st.markdown(
            f"<div style='background:#0d0f14;border:1px solid #1e2130;"
            f"border-radius:10px;padding:12px 18px;margin-top:10px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='color:#6b7280;font-family:DM Mono,monospace;font-size:0.78rem'>"
            f"TOTAL RISK DEDUCTION</span>"
            f"<span style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;"
            f"color:{_rs_band_color}'>−{_rb_pts_total} pts &nbsp;→&nbsp; "
            f"Score: {_rs_score}/100 &nbsp; {_rs_band}</span>"
            f"</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────
# AGENT C: COMPLAINT — PRIORITY DECISION AGENT
# ─────────────────────────────────────────────────────────────────
if "Complaint" in _doc_type:

    st.markdown("<hr class='ds-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>🚨 Complaint Priority</div>",
        unsafe_allow_html=True
    )

    import re as _re_cp

    # ── Signals ───────────────────────────────────────────────
    _cp_word_count   = len(_flat_text.split())
    _cp_has_subject  = "Subject" in _labels
    _cp_has_ref      = "Reference" in _labels
    _cp_has_sender   = "Submitted By" in _labels

    # Urgency / intensity keywords
    _cp_urgency_kw   = [
        "urgent", "immediately", "asap", "unacceptable", "furious",
        "demand", "legal", "lawsuit", "escalate", "disgusting",
        "fraud", "cheated", "scam", "worst", "terrible", "horrible",
        "threatening", "compensation", "refund", "immediately"
    ]
    _cp_lower        = _flat_text.lower()
    _cp_urgency_hits = [kw for kw in _cp_urgency_kw if kw in _cp_lower]

    # Complaint keyword presence
    _cp_complaint_kw = ["complaint", "issue", "problem", "dissatisfied",
                        "unhappy", "refund", "poor", "failed", "broken",
                        "damaged", "wrong", "error", "delay"]
    _cp_complaint_hits = [kw for kw in _cp_complaint_kw if kw in _cp_lower]

    # ── Priority decision (rule-based) ────────────────────────
    _cp_score = 0
    _cp_score += min(len(_cp_urgency_hits)  * 15, 45)   # max 45 from urgency
    _cp_score += min(len(_cp_complaint_hits) * 5, 25)    # max 25 from complaint kw
    _cp_score += (10 if _cp_has_subject  else 0)         # subject present
    _cp_score += (5  if _cp_word_count > 100 else 0)     # detailed text
    _cp_score += (10 if _cp_word_count  > 200 else 0)    # very detailed
    _cp_score += (5  if _cp_has_sender   else 0)         # sender known

    _cp_score = min(_cp_score, 100)

    if _cp_score >= 60 or len(_cp_urgency_hits) >= 2:
        _cp_priority      = "🔴 High Priority"
        _cp_priority_color = "#ef4444"
        _cp_priority_bg    = "#1a0505"
        _cp_priority_bdr   = "#ef4444"
        _cp_sla            = "Respond within 4 hours"
        _cp_band_label     = "CRITICAL"
    elif _cp_score >= 30 or _cp_word_count >= 50:
        _cp_priority      = "🟡 Medium Priority"
        _cp_priority_color = "#f59e0b"
        _cp_priority_bg    = "#1a1000"
        _cp_priority_bdr   = "#f59e0b"
        _cp_sla            = "Respond within 24 hours"
        _cp_band_label     = "MODERATE"
    else:
        _cp_priority      = "🟢 Low Priority"
        _cp_priority_color = "#22c55e"
        _cp_priority_bg    = "#011a05"
        _cp_priority_bdr   = "#22c55e"
        _cp_sla            = "Respond within 48–72 hours"
        _cp_band_label     = "ROUTINE"

    # ── Layout ────────────────────────────────────────────────
    _cp_col1, _cp_col2 = st.columns([1, 1])

    with _cp_col1:
        # Priority gauge card
        st.markdown(
            f"<div style='background:{_cp_priority_bg};"
            f"border:2px solid {_cp_priority_bdr};"
            f"border-radius:14px;padding:24px;text-align:center;margin-bottom:12px'>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.7rem;"
            f"color:#6b7280;letter-spacing:0.1em;text-transform:uppercase'>"
            f"Priority Level</div>"
            f"<div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;"
            f"color:{_cp_priority_color};margin:8px 0'>{_cp_priority}</div>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
            f"background:{_cp_priority_color}20;color:{_cp_priority_color};"
            f"border-radius:20px;display:inline-block;padding:4px 14px;"
            f"margin-bottom:8px'>{_cp_band_label}</div>"
            f"<div style='font-size:0.82rem;color:#94a3b8;margin-top:8px'>"
            f"⏱ SLA: {_cp_sla}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Signal summary checklist
        st.markdown(
            "<div class='section-header' style='font-size:0.9rem'>Signal Summary</div>",
            unsafe_allow_html=True
        )
        for _sig_lbl, _sig_val, _sig_ok in [
            ("Subject Stated",    f"{'Yes' if _cp_has_subject else 'Not found'}",  _cp_has_subject),
            ("Sender Identified", f"{'Yes' if _cp_has_sender  else 'Not found'}",  _cp_has_sender),
            ("Reference Number",  f"{'Yes' if _cp_has_ref     else 'Not found'}",  _cp_has_ref),
            ("Word Count",        f"{_cp_word_count} words",                       _cp_word_count >= 50),
            ("Urgency Keywords",  f"{len(_cp_urgency_hits)} detected",             len(_cp_urgency_hits) > 0),
            ("Complaint Keywords",f"{len(_cp_complaint_hits)} detected",           len(_cp_complaint_hits) > 0),
        ]:
            _sc = "#22c55e" if _sig_ok else "#6b7280"
            _si = "✅" if _sig_ok else "○"
            st.markdown(
                f"<div class='kp-card'>"
                f"<div class='kp-label'>{_sig_lbl}</div>"
                f"<div class='kp-value' style='color:{_sc}'>{_si} {_sig_val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    with _cp_col2:
        # Why this priority
        st.markdown(
            "<div class='section-header' style='font-size:0.95rem'>"
            "🧠 Why This Priority?</div>",
            unsafe_allow_html=True
        )

        _cp_reasons = []

        if len(_cp_urgency_hits) >= 2:
            _cp_reasons.append(("🔴", "#ef4444",
                                 f"Strong urgency detected ({len(_cp_urgency_hits)} keywords)",
                                 f"Words like '{_cp_urgency_hits[0]}' and '{_cp_urgency_hits[1]}' "
                                 f"signal high emotional intensity or legal threat."))
        elif len(_cp_urgency_hits) == 1:
            _cp_reasons.append(("🟡", "#f59e0b",
                                 f"Urgency keyword present: '{_cp_urgency_hits[0]}'",
                                 "Single urgency signal detected — monitor closely."))

        if len(_cp_complaint_hits) >= 3:
            _cp_reasons.append(("🔴", "#ef4444",
                                 f"{len(_cp_complaint_hits)} complaint indicators found",
                                 "Multiple complaint-related words confirm this is a serious grievance."))
        elif _cp_complaint_hits:
            _cp_reasons.append(("🟡", "#f59e0b",
                                 f"Complaint keywords: {', '.join(_cp_complaint_hits[:3])}",
                                 "Complaint tone confirmed — requires a formal response."))

        if _cp_has_subject:
            _cp_reasons.append(("✅", "#22c55e",
                                 "Subject line is clearly stated",
                                 "A clear subject helps route and prioritise the complaint efficiently."))
        else:
            _cp_reasons.append(("⚠️", "#f59e0b",
                                 "Subject line missing",
                                 "No subject detected — complaint may be harder to categorise and route."))

        if _cp_word_count > 200:
            _cp_reasons.append(("🔴", "#ef4444",
                                 f"Detailed complaint ({_cp_word_count} words)",
                                 "Lengthy complaints typically indicate serious dissatisfaction and more complex issues."))
        elif _cp_word_count > 50:
            _cp_reasons.append(("🟡", "#f59e0b",
                                 f"Moderate detail ({_cp_word_count} words)",
                                 "Some detail present but may need clarification during resolution."))
        else:
            _cp_reasons.append(("⚠️", "#f59e0b",
                                 f"Very short text ({_cp_word_count} words)",
                                 "Complaint is brief — request more details before escalating."))

        if _cp_has_sender:
            _cp_reasons.append(("✅", "#22c55e",
                                 "Complainant identified",
                                 "Sender details were extracted — enables direct follow-up."))
        else:
            _cp_reasons.append(("⚠️", "#f59e0b",
                                 "Complainant not identified",
                                 "No sender/submitter name found — anonymous complaint."))

        if _cp_has_ref:
            _cp_reasons.append(("✅", "#22c55e",
                                 "Reference / Order number present",
                                 "Reference number enables quick case lookup and tracking."))
        else:
            _cp_reasons.append(("ℹ️", "#6b7280",
                                 "No reference number",
                                 "Absence of reference slows case tracking — request from complainant."))

        for _r_icon, _r_color, _r_title, _r_detail in _cp_reasons:
            st.markdown(
                f"<div style='background:#12151e;border:1px solid #1e2130;"
                f"border-left:4px solid {_r_color};border-radius:10px;"
                f"padding:12px 16px;margin:6px 0'>"
                f"<b style='color:{_r_color}'>{_r_icon} {_r_title}</b>"
                f"<div style='color:#94a3b8;font-size:0.8rem;margin-top:4px'>{_r_detail}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Suggested response actions
        st.markdown(
            "<div class='section-header' style='font-size:0.9rem;margin-top:16px'>"
            "📌 Response Actions</div>",
            unsafe_allow_html=True
        )
        _cp_actions = []
        if len(_cp_urgency_hits) >= 2:
            _cp_actions.append("Escalate immediately to senior support team")
        if not _cp_has_subject:
            _cp_actions.append("Classify and tag complaint manually before routing")
        if not _cp_has_sender:
            _cp_actions.append("Attempt to identify complainant via reference or contact info")
        if _cp_word_count < 50:
            _cp_actions.append("Request complainant to provide more detail about the issue")
        if not _cp_has_ref:
            _cp_actions.append("Ask complainant for order/case reference number")
        if len(_cp_urgency_hits) == 0 and _cp_word_count >= 50:
            _cp_actions.append("Acknowledge receipt within SLA window")
            _cp_actions.append("Assign to relevant department for investigation")
        if not _cp_actions:
            _cp_actions.append("Acknowledge complaint and acknowledge within SLA timeframe")
            _cp_actions.append("Assign case to appropriate support agent")

        for _ca in _cp_actions:
            st.markdown(
                f"<div class='kp-card'><div class='kp-value'>➤ {_ca}</div></div>",
                unsafe_allow_html=True
            )

