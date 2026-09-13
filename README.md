# Zyro HR Assistant — RAG-based HR Chatbot

An AI-powered internal HR assistant that answers employee questions using 
Retrieval-Augmented Generation (RAG) over a company's HR policy documents.

## Features
- Ingests and indexes PDF HR policy documents (leave policy, WFH policy, 
  code of conduct, POSH, onboarding, compensation, IT security, and more)
- Full RAG pipeline: PDF loading → recursive text chunking → 
  sentence-transformer embeddings → FAISS vector store
- MMR-based retrieval for diverse, relevant context on every query
- LLM-powered answers via Groq's Llama 3.3 70B model, with a strict system 
  prompt that enforces factual, policy-grounded responses
- Built-in scope guardrail — a binary in-scope/out-of-scope classifier 
  ensures the bot only answers HR-related questions and declines everything else
- Interactive Streamlit chat interface with expandable source-document 
  citations and custom UI styling

## Tech Stack
Python, LangChain, FAISS, HuggingFace Embeddings, Groq API (Llama 3.3 70B), Streamlit

## How It Works
1. On startup, all PDFs in `hr_docs/` are loaded and split into overlapping chunks.
2. Each chunk is embedded with a sentence-transformer model and stored in a FAISS vector index.
3. Every user question is first classified as in-scope or out-of-scope.
4. If in-scope, the most relevant chunks are retrieved (MMR search) and passed as 
   context to the LLM, which generates a grounded, source-cited answer.

## Run Locally
```bash
git clone https://github.com/Uday-6145/zyro-hr-chatbot.git
cd zyro-hr-chatbot
pip install -r requirements.txt

# add your Groq API key as an environment variable
export GROQ_API_KEY=your_key_here

streamlit run app.py
```

## Future Improvements
- Add conversation memory across sessions
- Support additional document formats (docx, plain text)
- Add user authentication for multi-department policy access
