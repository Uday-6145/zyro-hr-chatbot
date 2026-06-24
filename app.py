import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics — HR Assistant",
    page_icon="https://img.icons8.com/fluency/48/briefcase.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Navy: #0B1F3A  |  Blue accent: #2563EB  |  Slate 900: #0F172A
# Slate 600: #475569  |  Slate 400: #94A3B8  |  Border: #E2E8F0
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stMarkdown, .stChatMessage {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Main layout ── */
.main .block-container {
    padding: 2rem 3rem 2.5rem !important;
    max-width: 860px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.25rem !important;
}

/* Sidebar brand */
.sb-brand {
    font-size: 15px;
    font-weight: 700;
    color: #0b1f3a;
    line-height: 1.2;
    margin-bottom: 2px;
}
.sb-brand-tag {
    font-size: 11px;
    font-weight: 500;
    color: #94a3b8;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* Sidebar section label */
.sb-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
}

/* Policy list items */
.sb-policy {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 400;
    color: #334155;
    margin-bottom: 1px;
    transition: background 0.15s;
}
.sb-policy-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #2563eb;
    flex-shrink: 0;
}

/* Sidebar divider */
.sb-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 18px 0;
}

/* Sidebar notice */
.sb-notice {
    border-left: 3px solid #2563eb;
    background: #eff6ff;
    border-radius: 0 6px 6px 0;
    padding: 10px 12px;
    font-size: 12px;
    color: #1e40af;
    line-height: 1.6;
}

/* ── Page header ── */
.page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 18px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 24px;
}
.page-title {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
    font-weight: 400;
}

/* Animated online badge — signature element */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 999px;
    padding: 5px 13px;
    font-size: 12px;
    font-weight: 500;
    color: #15803d;
    flex-shrink: 0;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1);   }
    50%       { opacity: .35; transform: scale(.8); }
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-size: 14.5px !important;
    line-height: 1.7  !important;
    color: #1e293b    !important;
}

/* ── Source expander ── */
[data-testid="stExpander"] {
    background: #fafafa !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    font-size: 12px  !important;
    color: #64748b   !important;
    font-weight: 500 !important;
}

/* Source items */
.src-item {
    font-size: 12.5px;
    color: #475569;
    padding: 4px 0 4px 10px;
    border-left: 2px solid #2563eb;
    margin: 4px 0;
    line-height: 1.4;
}

/* ── Chat input ── */
[data-testid="stChatInputTextArea"] {
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    color: #1e293b !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    font-size: 13px !important;
    color: #64748b !important;
}
</style>
""", unsafe_allow_html=True)


# ── Backend ───────────────────────────────────────────────────────────────────
CORPUS_PATH = "hr_docs/"


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading HR documents…")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7},
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=get_api_key(),
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a precise HR Help Desk assistant.

CRITICAL: Acrux Dynamics and Zyro Dynamics are the same company. Answer questions about either using the provided documents.
Answer ONLY using information in the context below.

RULES:
1. NO INTROS: Never say "According to the policy" or "Based on the context". Start directly with the facts.
2. NO METADATA: Do not mention document codes, page numbers, or file names.
3. EXACT FIGURES: Always include specific numbers, dates, percentages, and durations verbatim.
4. COMPLETE: If asked two things, answer both.
5. BULLETS: Use bullet points for eligibility conditions or multi-step processes.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a strict binary classifier for an HR chatbot.

Classify as IN_SCOPE if the question is about:
- Leave policies (EL, SL, maternity, paternity)
- Salary, payroll, CTC, compensation, grades, bonuses
- Work from home, hybrid, remote work
- Performance reviews, APR, PIP, ratings
- Code of conduct, ethics, discipline
- POSH, sexual harassment prevention
- Onboarding, probation, offboarding
- Travel reimbursements, expenses
- IT security, device policy
- Company profile, culture, benefits, PF, gratuity, insurance

Classify as OUT_OF_SCOPE if about:
- Individual stock options, ESOP vesting, or personal equity
- Stock prices, financial markets
- Company revenue, financial performance
- Product features, sales, CRM tools
- Recruitment, hiring process, job applications
- Competitor comparisons (Zoho, Freshworks, TCS, etc.)
- Sports, weather, general knowledge, coding, math, science
- Questions about policies at OTHER companies

Focus ONLY on the topic, not the company name.
Respond with ONLY: IN_SCOPE or OUT_OF_SCOPE"""),
        ("human", "{question}"),
    ])

    return (
        retriever,
        RAG_PROMPT | llm | StrOutputParser(),
        OOS_PROMPT | llm | StrOutputParser(),
    )


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


REFUSAL = (
    "That question falls outside the scope of Zyro Dynamics HR documentation. "
    "For further assistance, please reach out to the HR department directly."
)


def ask(question, retriever, prompt_chain, classifier_chain):
    verdict = classifier_chain.invoke({"question": question}).strip().upper()
    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL, []

    docs = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({
        os.path.basename(d.metadata.get("source", ""))
        .replace(".pdf", "").replace("_", " ")
        for d in docs
    })
    answer = prompt_chain.invoke({"context": context, "question": question})
    return answer, sources


# ── Sidebar ───────────────────────────────────────────────────────────────────
POLICIES = [
    "Company Profile",
    "Employee Handbook",
    "Leave Policy",
    "Work From Home Policy",
    "Code of Conduct",
    "Performance Review Policy",
    "Compensation & Benefits",
    "IT & Data Security",
    "Prevention of Sexual Harassment",
    "Onboarding & Separation",
    "Travel & Expense Policy",
]

with st.sidebar:
    st.markdown("""
        <p class="sb-brand">Zyro Dynamics</p>
        <p class="sb-brand-tag">HR Knowledge Base</p>
        <p class="sb-label">Policy Library</p>
    """, unsafe_allow_html=True)

    for policy in POLICIES:
        st.markdown(
            f'<div class="sb-policy">'
            f'<span class="sb-policy-dot"></span>{policy}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sb-notice">
            Responses are sourced exclusively from official Zyro Dynamics HR
            documentation. Contact HR directly for sensitive or complex matters.
        </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div>
        <div class="page-title">HR Policy Assistant</div>
        <div class="page-subtitle">Zyro Dynamics &nbsp;&middot;&nbsp; Internal Use Only</div>
    </div>
    <span class="status-badge">
        <span class="pulse-dot"></span>
        Online
    </span>
</div>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Good day. I'm the Zyro Dynamics HR Assistant. "
            "I can help you with leave policies, compensation, work-from-home "
            "guidelines, performance reviews, and other HR-related queries. "
            "How can I assist you?"
        ),
        "sources": [],
    }]


# ── Render conversation history ───────────────────────────────────────────────
def render_sources(sources):
    if sources:
        with st.expander("Referenced Documents"):
            for s in sources:
                st.markdown(f'<div class="src-item">{s}</div>', unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        render_sources(msg.get("sources", []))


# ── Pipeline ──────────────────────────────────────────────────────────────────
retriever, prompt_chain, classifier_chain = build_pipeline()


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an HR policy question…"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):
            answer, sources = ask(prompt, retriever, prompt_chain, classifier_chain)
        st.write(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
