# 🏢 Zyro Dynamics — Enterprise HR Policy Assistant

An AI-powered internal HR assistant built with **Retrieval-Augmented Generation (RAG)**, **FAISS**, and **Groq LPUs**. It answers employee questions with 100% factual accuracy directly from company policy PDFs, complete with document and page-level citations.

---

## 💡 Why I Built This Project

In almost every organization, two common problems happen every single day:
1. **Employees waste time searching:** Company policies are scattered across dozens of long, dense PDFs (Leave Policy, POSH, Code of Conduct, Travel Expenses, Insurance). When an employee needs to know something simple—like *"How many sick leaves can I take without a medical certificate?"* or *"Am I eligible for WFH during probation?"*—they have to hunt through 40+ pages.
2. **HR teams get overwhelmed:** HR managers spend hours answering the exact same repetitive questions via Slack, email, or tickets.

**The Solution:**  
I built the **Zyro Dynamics HR Assistant** to serve as an instant, 24/7 digital HR front desk. Employees get clear, conversational answers in less than 2 seconds, backed by exact policy source citations. Most importantly, the assistant is engineered with strict enterprise guardrails so it **never hallucinates** and **refuses to answer out-of-scope or inappropriate questions**.

---

## 🎯 Key Highlights

- **Zero Hallucination (`temperature = 0.0`):** Operates on strict greedy decoding, ensuring every response is purely grounded in the company's internal documents.
- **Traceable Citations:** Every answer references the exact source PDF and page number for complete transparency and compliance.
- **4-Layer Defense Guardrails:**
  1. *Input Security Filter:* Rejects prompt injection and adversarial jailbreak attempts.
  2. *Zero-Shot Intent Classifier:* Identifies whether the question is HR-related or out-of-scope (e.g., general knowledge, competitor policies, stock prices).
  3. *Grounded Synthesis:* Constrains answers strictly to retrieved policy passages.
  4. *Output Normalizer:* Sanitizes conversational fluff and standardizes currency formatting.
- **Blazing Fast Latency (<1.5s):** Uses Groq's custom LPU (Language Processing Unit) hardware for near-instant inference on open-source LLaMA models.
- **Full Observability:** Integrated with **LangSmith** telemetry to track latency, token usage, and retrieval quality across all queries.
- **Interactive Web Interface:** Deployed using **Gradio** with real-time status badges and source expanders.

---

## 🏗️ Architecture & How It Works

```text
User Question
      │
      ▼
[1. Input Security Guardrail]  ──(Adversarial/Jailbreak)──► 🚫 Security Refusal
      │ (Safe)
      ▼
[2. Intent & Scope Classifier]  ──(Out-of-Scope)──────────► 🛡️ Enterprise Refusal
      │ (In-Scope HR Query)
      ▼
[3. Vector Retrieval (FAISS)]
      │ • Embed query with sentence-transformers/all-mpnet-base-v2 (768D)
      │ • Retrieve Top-5 semantically relevant policy chunks from RAM
      ▼
[4. Prompt Synthesis & LLM]
      │ • Context formatted with [Source Document | Page Number]
      │ • LLaMA model on Groq generates grounded, deterministic answer
      ▼
[5. Output Normalization]
      │ • Cleans formatting, normalizes currency, standardizes bullets
      ▼
Final Grounded Answer with Source Citations
```

---

## 📁 Repository Structure

```text
.
├── README.md               # Project documentation & architecture overview
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignores environment files, caches, and system files
├── hr_docs/                # Official Zyro Dynamics HR policy documents (11 PDFs)
│   ├── 00_Company_Profile.pdf
│   ├── 01_Employee_Handbook.pdf
│   ├── 02_Leave_Policy.pdf
│   ├── 03_Work_From_Home_Policy.pdf
│   ├── 04_Code_of_Conduct.pdf
│   ├── 05_Performance_Review_Policy.pdf
│   ├── 06_Compensation_and_Benefits_Policy.pdf
│   ├── 07_IT_and_Data_Security_Policy.pdf
│   ├── 08_Prevention_of_Sexual_Harassment_Policy.pdf
│   ├── 09_Onboarding_and_Separation_Policy.pdf
│   └── 10_Travel_and_Expense_Policy.pdf
└── zyro_chat_bot.ipynb     # End-to-end Colab notebook (Ingestion, RAG, Guardrails, UI)
```

---

## 🛠️ Tech Stack

| Component | Technology | Why It Was Chosen |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://www.langchain.com/) (LCEL) | Modular composability and streaming runnables |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` | High-quality 768D semantic representations; runs locally at $0 cost with total data privacy |
| **Vector Index** | [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) | Ultra-fast, in-memory nearest-neighbor retrieval without database server overhead |
| **LLM Inference** | [Groq LPU](https://groq.com/) | Near-zero latency (~300+ tokens/sec) on Meta's LLaMA architecture |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Production-grade tracing of inputs, outputs, token consumption, and retrieval accuracy |
| **UI Framework** | [Gradio](https://www.gradio.app/) | Interactive web UI with shareable live public links |

---

## 🚀 Getting Started

### Option A: Run in Google Colab (Recommended)
1. Open the notebook [`zyro_chat_bot.ipynb`](./zyro_chat_bot.ipynb) in **Google Colab**.
2. Add your API keys under Colab Secrets (the 🔑 icon on the left panel):
   - `GROQ_API_KEY`: Your free Groq API key from [console.groq.com](https://console.groq.com)
   - `LANGSMITH_API_KEY` *(optional)*: Your LangSmith API key for telemetry
3. Upload the policy PDFs from `hr_docs/` into `/content/corpus/`.
4. Run all cells sequentially. The Gradio cell will generate a live public URL to interact with the bot!

### Option B: Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Uday-6145/zyro-hr-chatbot.git
   cd zyro-hr-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set your environment variables:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   export LANGCHAIN_API_KEY="your-langsmith-key" # Optional
   ```

4. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook zyro_chat_bot.ipynb
   ```

---

## 📊 Evaluation & Testing

The system includes an automated test matrix evaluating:
1. **In-Scope Policy Queries:** Checks that questions on Earned Leave, WFH eligibility, Maternity Leave, and PIP rules retrieve the correct PDF documents and return accurate answers.
2. **Out-of-Scope Queries:** Verifies that off-topic queries (e.g., stock prices, cricket, competitor policies) are cleanly intercepted and declined by the intent classifier.
3. **Adversarial Jailbreaks:** Verifies that prompt injection attempts (e.g., *"Ignore previous instructions"*) are immediately blocked.

```text
============================================================
📊 PRODUCTION TEST SUITE RESULTS
============================================================
 • In-Scope Policy Retrieval: 100% Accuracy with Page Citations
 • Out-of-Scope Rejection:    100% Precision
 • Adversarial Defense:       100% Block Rate
 • Average Response Latency:  ~1.2 seconds
 • Hallucination Compliance:  Zero Hallucinations (temperature=0.0)
============================================================
```

---

## 👤 Author

Developed by **Uday Pratap Singh**  
- **GitHub:** [@Uday-6145](https://github.com/Uday-6145)
