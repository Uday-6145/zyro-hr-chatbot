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
    page_title="Zyro HR Desk",
    page_icon="https://img.icons8.com/ios-filled/50/2563EB/briefcase.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font override */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Clean up Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar: always dark — enterprise dashboard pattern ── */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
/* Override all sidebar text to light */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #F1F5F9 !important;
}
/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #1E293B !important;
    margin: 1rem 0 !important;
}
/* Sidebar info box */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: rgba(37, 99, 235, 0.12) !important;
    border: 1px solid rgba(37, 99, 235, 0.25) !important;
    border-radius: 6px !important;
    color: #93C5FD !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #93C5FD !important;
    font-size: 0.8rem !important;
}

/* ── Main content area ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
}

/* ── Custom header block ── */
.zd-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
}
.zd-wordmark {
    font-size: 1.1rem;
    font-weight: 700;
    color: #2563EB;
    letter-spacing: -0.02em;
}
.zd-divider {
    width: 1px;
    height: 18px;
    background: rgba(148, 163, 184, 0.3);
}
.zd-subtitle {
    font-size: 0.82rem;
    font-weight: 400;
    color: #64748B;
    letter-spacing: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.6rem !important;
    background: transparent !important;
}
/* User message: subtle blue tint */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #2563EB !important;
    background: rgba(37, 99, 235, 0.04) !important;
}
/* Assistant message: left rule */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid rgba(148, 163, 184, 0.25) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] {
    border-radius: 8px !important;
}

/* ── Source expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 6px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Sidebar section label ── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin: 0 0 0.75rem 0;
    display: block;
}
/* Policy list */
.policy-list {
    list-style: none;
    padding: 0;
    margin: 0 0 1rem 0;
}
.policy-list li {
    padding: 0.35rem 0;
    font-size: 0.83rem;
    color: #94A3B8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    display: flex;
    align-items: center;
    gap: 8px;
}
.policy-list li::before {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #475569;
    flex-shrink: 0;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: rgba(34, 197, 94, 0.1);
    color: #86EFAC;
    border: 1px solid rgba(34, 197, 94, 0.2);
    margin-bottom: 1rem;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
    font-size: 0.82rem !important;
    color: #64748B !important;
}

/* Dark mode adaptations for main area */
@media (prefers-color-scheme: dark) {
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(37, 99, 235, 0.08) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CORPUS_PATH = "hr_docs/"
REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents."

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


# ── API key helper ─────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


# ── Pipeline (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base...")
def build_pipeline():
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        groq_api_key=get_api_key()
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
        ("human", "{question}")
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
        ("human", "{question}")
    ])

    prompt_chain     = RAG_PROMPT | llm | StrOutputParser()
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    return retriever, prompt_chain, classifier_chain


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def ask(question, retriever, prompt_chain, classifier_chain):
    verdict = classifier_chain.invoke({"question": question}).strip().upper()
    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL, []

    docs    = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({
        os.path.basename(d.metadata.get("source", ""))
        .replace(".pdf", "")
        .replace("_", " ")
        for d in docs
    })
    answer = prompt_chain.invoke({"context": context, "question": question})
    return answer, sources


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 0.5rem 0 1.25rem;">
            <div style="font-size:1rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.02em;">
                Zyro Dynamics
            </div>
            <div style="font-size:0.72rem; color:#475569; margin-top:2px; text-transform:uppercase; letter-spacing:0.08em;">
                HR Knowledge Base
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Available Policies</span>', unsafe_allow_html=True)

    policy_items = "".join(f"<li>{p}</li>" for p in POLICIES)
    st.markdown(
        f'<ul class="policy-list">{policy_items}</ul>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size:0.78rem; color:#475569; line-height:1.5;">
            Responses are generated exclusively from official Zyro Dynamics HR documents.
            For decisions affecting your employment, consult the HR team directly.
        </div>
    """, unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="zd-header">
        <span class="zd-wordmark">HR Desk</span>
        <div class="zd-divider"></div>
        <span class="zd-subtitle">Policy Assistant &mdash; Acrux Dynamics</span>
    </div>
""", unsafe_allow_html=True)

# ── Load pipeline ──────────────────────────────────────────────────────────────
retriever, prompt_chain, classifier_chain = build_pipeline()

st.markdown("""
    <div class="status-badge">
        <span class="status-dot"></span>
        Knowledge base ready &mdash; 11 policy documents
    </div>
""", unsafe_allow_html=True)

# ── Chat state ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome. I can answer questions about leave entitlements, "
            "work-from-home arrangements, compensation, performance reviews, "
            "code of conduct, travel reimbursements, and other HR policies. "
            "How can I help you today?"
        ),
        "sources": []
    }]

# ── Render history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(s)

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an HR policy question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            answer, sources = ask(prompt, retriever, prompt_chain, classifier_chain)
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
