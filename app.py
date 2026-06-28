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
    page_title="Zyro HR Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CUSTOM CSS: "TECHNICAL BRUTALISM" THEME ---
st.markdown("""
    <style>
        /* Import Monospace Font */
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');

        /* Global Font Override */
        html, body, [class*="css"] {
            font-family: 'Space Mono', monospace !important;
        }

        /* Hide Streamlit Header & Footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Main Background & Text Color */
        .stApp {
            background-color: #FFFFFF;
            color: #1A1A1A;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F8F9F8 !important;
            border-right: 2px solid #E5E5E5;
        }

        /* The Custom Stacked Logo */
        .sidebar-logo {
            font-size: 3.5rem;
            font-weight: 700;
            color: #2E5D4B; /* Forest Green */
            line-height: 0.9;
            letter-spacing: -2px;
            margin-bottom: 5px;
        }
        .sidebar-subtitle {
            font-size: 0.85rem;
            color: #666666;
            margin-bottom: 30px;
        }

        /* Brutalist Stats Box */
        .stats-box {
            border: 2px solid #2E5D4B;
            padding: 15px;
            text-align: center;
            margin: 20px 0;
            background-color: #FFFFFF;
        }
        .stats-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2E5D4B;
            line-height: 1;
        }
        .stats-text {
            font-size: 0.75rem;
            text-transform: lowercase;
            color: #1A1A1A;
        }

        /* Main Area Big Header */
        .main-header {
            font-size: 2.5rem;
            color: #D3D3D3; /* Light gray */
            font-weight: 700;
            border-bottom: 2px solid #F0F0F0;
            padding-bottom: 10px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .main-header-icon {
            color: #2E5D4B;
            font-size: 2rem;
        }

        /* Subheading */
        .sub-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #2E5D4B;
            margin-bottom: 15px;
            text-transform: lowercase;
        }

        /* Clean up Chat Interface */
        .stChatMessage {
            background-color: transparent !important;
            border-bottom: 1px dashed #E5E5E5;
            padding-bottom: 20px;
            padding-top: 10px;
        }
        
        /* Force chat text to be dark regardless of system theme */
        [data-testid="stChatMessageContent"] p, 
        [data-testid="stChatMessageContent"] div,
        [data-testid="stChatMessageContent"] li {
            color: #1A1A1A !important;
        }
        
        /* FIX FOR INVISIBLE CHAT INPUT TEXT */
        [data-testid="stChatInput"] {
            background-color: transparent !important;
        }
        [data-testid="stChatInput"] > div {
            border: 2px solid #2E5D4B !important;
            border-radius: 0px !important; 
            background-color: #FFFFFF !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #1A1A1A !important;
            background-color: #FFFFFF !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #888888 !important;
        }
        [data-testid="stChatInput"] button {
            color: #2E5D4B !important;
        }
        
        /* Expander / Sources Styling */
        .streamlit-expanderHeader {
            font-family: 'Space Mono', monospace !important;
            font-size: 0.85rem !important;
            color: #2E5D4B !important;
            border: 1px solid #E5E5E5 !important;
            background-color: #F8F9F8 !important;
        }
        
        /* Plain text lists in sidebar */
        .policy-list {
            font-size: 0.8rem;
            line-height: 1.8;
            color: #1A1A1A;
        }
    </style>
""", unsafe_allow_html=True)

CORPUS_PATH = "hr_docs/"

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

@st.cache_resource(show_spinner="Loading HR documents...")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
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
    retriever = vectorstore.as_retriever(
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

    prompt_chain = RAG_PROMPT | llm | StrOutputParser()
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    return retriever, prompt_chain, classifier_chain

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents."

def ask(question, retriever, prompt_chain, classifier_chain):
    verdict = classifier_chain.invoke({"question": question}).strip().upper()
    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL, []

    docs = retriever.invoke(question)
    context = format_docs(docs)
    sources = list({
        os.path.basename(d.metadata.get('source', '')).replace('.pdf', '').replace('_', ' ')
        for d in docs
    })
    answer = prompt_chain.invoke({"context": context, "question": question})
    return answer, sources


# ── UI LAYOUT ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">zyro<br>bot</div>
        <div class="sidebar-subtitle">AI-powered internal HR assistant</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="stats-box">
            <div class="stats-number">11</div>
            <div class="stats-text">policies loaded</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><div style='font-size: 0.85rem; font-weight: bold; color: #2E5D4B;'>indexing:</div>", unsafe_allow_html=True)
    
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct",
        "Performance Review Policy", "Compensation & Benefits",
        "IT & Data Security", "Prevention of Sexual Harassment",
        "Onboarding & Separation", "Travel & Expense Policy"
    ]
    
    policy_html = "<div class='policy-list'>"
    for p in policies:
        policy_html += f"&gt; {p.lower()}<br>"
    policy_html += "</div>"
    st.markdown(policy_html, unsafe_allow_html=True)
    
    st.markdown("<br><br><div style='font-size: 0.7rem; color: #999;'>Powered by LangChain & Groq</div>", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <span>ask me anything</span>
        <span class="main-header-icon">[▶]</span>
    </div>
    <div class="sub-heading">recently answered</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "system initialized. ready to process queries regarding zyro dynamics internal hr policies.",
        "sources": []
    }]

USER_AVATAR = "👤"
BOT_AVATAR = "🟩"

for msg in st.session_state.messages:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("[ view source documents ]"):
                for s in msg["sources"]:
                    st.write(f"- {s.lower()}")

retriever, prompt_chain, classifier_chain = build_pipeline()

if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("processing..."):
            answer, sources = ask(prompt, retriever, prompt_chain, classifier_chain)
        st.write(answer)
        if sources:
            with st.expander("[ view source documents ]"):
                for s in sources:
                    st.write(f"- {s.lower()}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
