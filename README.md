# 📄 Document Q&A System

A **RAG (Retrieval-Augmented Generation)** application that lets you upload any PDF document and ask questions about it using AI. Built with LangChain, ChromaDB, HuggingFace Embeddings, and Groq (LLaMA 3).

---

## 🚀 Live Demo

> Upload a PDF → Ask questions → Get instant AI-powered answers based strictly on your document.

---

## ✨ Features

- 📤 Upload any PDF document
- 🧠 AI answers based **only** on your document — no hallucinations
- 📖 Shows **source page numbers** for every answer
- 💾 Saves vector store to disk — reload previous documents instantly
- ⚡ Fast responses powered by **Groq inference**
- 🆓 Completely free to run (no paid APIs required)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | LLaMA 3 8B via Groq |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB |
| **RAG Framework** | LangChain (LCEL) |
| **PDF Loader** | PyPDFLoader |

---

## 📁 Project Structure

```
📦 Medical Report Explainer
├── app.py              # Streamlit frontend UI
├── rag_engine.py       # RAG pipeline (backend logic)
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
└── chroma_db/          # Vector store (auto-created on first run)
```

---

## ⚙️ How It Works

```
PDF Upload
    ↓
Extract text page by page (PyPDFLoader)
    ↓
Split into 1000-char overlapping chunks (RecursiveCharacterTextSplitter)
    ↓
Convert chunks to vectors (HuggingFace Embeddings)
    ↓
Store vectors in ChromaDB
    ↓
User asks a question
    ↓
Find top 3 similar chunks from ChromaDB
    ↓
Send chunks + question to LLaMA 3 via Groq
    ↓
Display answer + source page numbers
```

---

## 🔧 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-report-explainer.git
cd medical-report-explainer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

Go to 👉 [https://console.groq.com](https://console.groq.com) and sign up for free.

### 4. Create a `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🖥️ Usage

1. **Upload a PDF** from the sidebar
2. Click **"Process Document"** and wait for it to load
3. **Type your question** in the chat box
4. Get an AI answer with **source page references**
5. Use **"Load Previous Document"** to reload without reprocessing

---

## 📦 Dependencies

```
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-chroma
langchain-groq
chromadb
pypdf
python-dotenv
streamlit
sentence-transformers
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your secret in **App Settings → Secrets**:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy!

> **Note:** ChromaDB is stored locally, so the "Load Previous Document" feature won't persist between deployments on free hosting.

---

## 📄 Good PDFs to Try

- 🏥 Medical reports
- 📚 Research papers
- ⚖️ Legal contracts
- 📊 Company annual reports
- 📖 Textbooks

---

## 🔒 Security

- Never commit your `.env` file
- `.env` and `chroma_db/` are excluded via `.gitignore`
- API keys are loaded securely via `python-dotenv`

---

## 👨‍💻 Author

Built by **Monish**

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
