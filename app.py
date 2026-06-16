app_code = """
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🏢",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CORPUS_PATH  = "hr_docs/"

# ── Cache: build pipeline once ───────────────────────────
@st.cache_resource(show_spinner="Loading HR documents...")
def build_pipeline():
    # 1. Load
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Vector store + MMR retriever (Expanded net for multi-hop tables)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
    )

    # 5. LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )

    # 6. Prompts (Ultra-Concise for Semantic Scoring)
    RAG_PROMPT = ChatPromptTemplate.from_template(
        \"\"\"You are a precise HR Help Desk assistant. Zyro Dynamics and Acrux Dynamics are the same company.
Answer the user's question using ONLY the provided context.

CRITICAL RULES FOR SEMANTIC SCORING:
1. START IMMEDIATELY: Do not use introductory phrases like "According to the policy", "Based on the context", or "The documents state". Just give the answer.
2. NO METADATA: Do NOT output document codes, names, or page numbers.
3. CONCISE: Be as brief and direct as possible. 
4. EXACT MATH: Copy exact numbers, durations, and percentages.
5. PARTIAL: If you only find half the answer, state what you found. Do not apologize for missing information.

Context:
{context}

Question: {question}
Answer:\"\"\"
    )

    OOS_PROMPT = ChatPromptTemplate.from_template(
        \"\"\"You are a strict binary classifier for an HR chatbot serving Zyro/Acrux Dynamics.

Classify as IN_SCOPE if the question is about:
- Leaves, maternity, sick leave, salary, payroll, CTC ranges, bonus targets, health insurance, WFH rules, performance reviews (PIP, APR), or general HR policies.

Classify as OUT_OF_SCOPE if the question is about:
- Individual stock options, ESOP vesting, or personal equity (like "how many stock options will I receive").
- External recruitment or job applications.
- Company revenue or financial performance.
- Competitor companies (Zoho, Freshworks).
- Product features or CRM tools.

Question: {question}
Respond with exactly ONE word: IN_SCOPE or OUT_OF_SCOPE.\"\"\"
    )

    # 7. Chains
    prompt_chain = RAG_PROMPT | llm | StrOutputParser()
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    return retriever, prompt_chain, classifier_chain

def format_docs(docs):
    # Stripped metadata to prevent the LLM from reading it out loud
    return "\\n\\n---\\n\\n".join(doc.page_content for doc in docs)

REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents."

def ask(question, retriever, prompt_chain, classifier_chain):
    # 1. Classify
    verdict = classifier_chain.invoke({"question": question}).strip().upper()
    if "OUT_OF_SCOPE" in verdict:
        return REFUSAL, []
    
    # 2. Query Enrichment (Ensures Acrux and Zyro docs are both found)
    search_query = question + " Zyro Dynamics Acrux Dynamics"
    docs = retriever.invoke(search_query)
    
    # 3. Format and extract sources for the UI
    context = format_docs(docs)
    sources = list({
        os.path.basename(d.metadata.get('source','')).replace('.pdf','').replace('_',' ')
        for d in docs
    })
    
    # 4. Generate Answer
    answer = prompt_chain.invoke({"context": context, "question": question})
    return answer, sources

# ── UI ───────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Assistant")
st.caption("Ask me anything about Zyro Dynamics HR policies")

with st.sidebar:
    st.header("📚 Available Policies")
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct",
        "Performance Review Policy", "Compensation & Benefits",
        "IT & Data Security", "Prevention of Sexual Harassment",
        "Onboarding & Separation", "Travel & Expense Policy"
    ]
    for p in policies:
        st.write(f"• {p}")
    st.divider()
    st.info("💡 This bot answers only from official Zyro Dynamics HR documents.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm the Zyro Dynamics HR Assistant. Ask me about leave, WFH, salary, benefits, or any other HR policy!",
        "sources": []
    })

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.write(f"• {s}")

# Load pipeline
retriever, prompt_chain, classifier_chain = build_pipeline()

# Input
if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources = ask(prompt, retriever, prompt_chain, classifier_chain)
        st.write(answer)
        if sources:
            with st.expander("📎 Sources"):
                for s in sources:
                    st.write(f"• {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py updated with maximum scoring optimizations.")
