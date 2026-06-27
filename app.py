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

    page_title="HR Assistant",

    layout="wide"

)



CORPUS_PATH = "hr_docs/"



def get_api_key():

    # Try st.secrets first (Streamlit Cloud), fall back to env var

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

        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7}  # matches best Kaggle config

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





# ── UI ────────────────────────────────────────────────────

st.title("Zyro Dynamics HR Assistant")

st.caption("Ask me anything about Zyro Dynamics HR policies")



with st.sidebar:

    st.header("📚 Available Policies")

    for p in [

        "Company Profile", "Employee Handbook", "Leave Policy",

        "Work From Home Policy", "Code of Conduct",

        "Performance Review Policy", "Compensation & Benefits",

        "IT & Data Security", "Prevention of Sexual Harassment",

        "Onboarding & Separation", "Travel & Expense Policy"

    ]:

        st.write(f"• {p}")

    st.divider()

    st.info("💡 This bot answers only from official Zyro Dynamics HR documents.")



if "messages" not in st.session_state:

    st.session_state.messages = [{

        "role": "assistant",

        "content": "Hello! I'm the Zyro Dynamics HR Assistant. Ask me about leave, WFH, salary, benefits, or any other HR policy!",

        "sources": []

    }]



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if msg.get("sources"):

            with st.expander("📎 Sources"):

                for s in msg["sources"]:

                    st.write(f"• {s}")



retriever, prompt_chain, classifier_chain = build_pipeline()



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
