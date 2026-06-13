import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Vector store + MMR retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # 5. LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )

    # 6. Prompts
    RAG_PROMPT = ChatPromptTemplate.from_template(
        """You are the HR assistant for Zyro Dynamics Pvt. Ltd.
Answer using ONLY the HR policy documents below.

Rules:
- Be specific: include exact numbers, durations, procedures
- Cite the policy document name
- Do NOT use outside knowledge
- If not in context, say: I can only answer HR-related questions from Zyro Dynamics policy documents.

Context:
{context}

Question: {question}
Answer:"""
    )

    OOS_PROMPT = ChatPromptTemplate.from_template(
        """You are a classifier. Is this question related to Zyro Dynamics HR policies?
HR topics: leave, salary, compensation, WFH, performance, code of conduct,
POSH, onboarding, offboarding, travel expenses, IT security, benefits.

Question: {question}
Reply with ONLY YES or NO:"""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{os.path.basename(doc.metadata.get('source','Policy')).replace('.pdf','').replace('_',' ')}]\n{doc.page_content}"
            for doc in docs
        )

    def get_sources(docs):
        return list({
            os.path.basename(doc.metadata.get('source','Unknown')).replace('.pdf','').replace('_',' ')
            for doc in docs
        })

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    return retriever, rag_chain, classifier_chain

REFUSAL = "I can only answer HR-related questions from Zyro Dynamics policy documents."

def ask(question, retriever, rag_chain, classifier_chain):
    verdict = classifier_chain.invoke({"question": question}).strip().upper()
    if verdict == "NO":
        return REFUSAL, []
    docs    = retriever.invoke(question)
    sources = list({
        os.path.basename(d.metadata.get('source','')).replace('.pdf','').replace('_',' ')
        for d in docs
    })
    answer  = rag_chain.invoke(question)
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
retriever, rag_chain, classifier_chain = build_pipeline()

# Input
if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources = ask(prompt, retriever, rag_chain, classifier_chain)
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