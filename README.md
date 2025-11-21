# 🤖 Compliance Helper RAG System  
### *AI-powered Compliance & Policy Question Answering using Groq LLaMA + Local Embeddings*

---

## 📌 Overview  

**Compliance Helper RAG** is a high-performance Retrieval-Augmented Generation (RAG) application designed to answer compliance and policy-related questions directly from uploaded documents.

This project utilizes:

- ⚡ **Groq LLaMA 3.1** (ultra-fast, deterministic policy Q&A)  
- 🔍 **HuggingFace MiniLM embeddings** (local, free, secure)  
- 📚 **FAISS vector search** (high-speed semantic retrieval)  
- 🖥️ **Streamlit interface** (smooth, user-friendly UI)

The assistant answers **only from uploaded documents** and provides clean, structured citations like: [Terms of Service Twitter.pdf#28]

---

## 🚀 Key Features  

### ✔ Upload PDF/TXT policies  
Supports internal documents, legal terms, compliance manuals, guidelines, and more.

### ✔ Fast ingestion + optimized chunking  
Chunking tailored for legal/policy content ensures high-quality retrieval.

### ✔ Local embeddings (no API cost)  
Uses `sentence-transformers/all-MiniLM-L6-v2`, giving:

- High semantic quality  
- Zero rate limits  
- Full data privacy  

### ✔ Groq-accelerated LLaMA 3.1 for Q&A  
Super fast reasoning using:

- `llama-3.1-8b-instant` (default)
- or upgrade to `llama-3.1-70b-versatile`

### ✔ Hybrid retrieval (semantic + lexical)  
Improves accuracy for compliance/legal queries:

- Minimum age requirements  
- Rights and restrictions  
- Data usage rules  
- Allowed vs. prohibited actions  

### ✔ Clean structured citations  
The assistant:

- ❌ Never hallucinates citations  
- 📌 Only cites the chunks used  
- 📝 Includes max 1–2 citations  
- 📐 Formats citations consistently  

---

## 🏗️ Architecture  

```bash
                                                       ┌────────────────────┐
                                                       │   Uploaded PDFs    │
                                                       │       /TXT         │
                                                       └──────────┬─────────┘
                                                                  ▼
                                                   ┌───────────────────────────┐
                                                   │   Ingestion(Chunk + Embed)│
                                                   │  - HF MiniLM Embeddings   │
                                                   │  - FAISS Vector Index     │
                                                   └──────────┬────────────────┘
                                                              ▼
                                                  ┌─────────────────────────────┐
                                                  │      Hybrid Retriever       │
                                                  │ (semantic + lexical rerank) │
                                                  └──────────┬──────────────────┘
                                                             ▼
                                                   ┌──────────────────────────┐
                                                   │       System Prompt      │
                                                   │ (strict citation control)│
                                                   └──────────┬───────────────┘
                                                              ▼
                                                   ┌──────────────────────────┐
                                                   │     Groq LLaMA 3.1 LLM   │
                                                   └──────────┬───────────────┘
                                                              ▼
                                                       ┌────────────────────┐
                                                       │    Final Answer    │
                                                       │  + Clean Citations │
                                                       └────────────────────┘
```   

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/akashkapoor0001/Compliance-Helper-RAG.git
cd Compliance-Helper-RAG
```
### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
.venv/Scripts/activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Create a .env file
```bash
⚠️ Never push .env to GitHub

# .env.example
GROQ_API_KEY=YOUR_GROQ_API_KEY
LLM_MODEL=YOUR_LLM_MODEL
EMBEDDING_MODEL=YOUR_EMBEDDING_MODEL
VECTOR_STORE_PATH=data/faiss.index
METADATA_PATH=data/metadata.json
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
MAX_RETRIEVALS=8
ALLOW_WEB_FALLBACK=False
```

## 📥 Usage

### Run the Streamlit app
```bash
streamlit run app.py
```

In the UI, you can:
```bash
📄 Upload policy documents

⚙️ Build semantic index

❓ Ask compliance questions

📌 See citations for every answer
```

## 📁 Repository Structure
```bash
Compliance-Helper-RAG/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── config/
│   └── config.py
│
├── models/
│   ├── embeddings.py
│   └── llm.py
│
├── utils/
│   ├── ingest.py
│   ├── retriever.py
│   └── response_formatter.py
│
├── scripts/
│   ├── reindex_twitter_complete.py
│
└── data/                     # Ignored by Git
    ├── uploaded/             # Uploaded files
    ├── faiss.index           # Vector index
    └── metadata.json         # Chunk metadata

```
## 🛡️ Security
```bash
.env is ignored via .gitignore

No external API calls for embeddings

Documents never leave your machine

Groq handles only the LLM reasoning

No sensitive metadata is logged
```

## 🧪 Future Improvements
```bash
Add BM25 retriever

Knowledge graph extraction

Conversation memory (multi-turn RAG)

Export Q&A as a compliance report

Admin dashboard for document management

Deployment to Streamlit Cloud / Vercel
```

## 💡 Contributing
```bash
Pull requests are welcome!
Please follow PEP8 and include docstrings wherever possible.
```

## 📜 License
```bash
This project is licensed under the MIT License — free for personal and commercial use.
```
