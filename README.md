# RAG AI Application – Cognizant Annual Report Assistant

This Retrieval-Augmented Generation (RAG) application, built with **LangChain**, **FAISS**, and **Streamlit**, lets you upload, process, and query PDF documents interactively.  
Originally optimized for the **Cognizant 2024 Annual Report**, it now supports **general PDFs**, provides **clickable suggested questions** grouped by category, and offers **answers with source references**.

---

## 🚀 Features

- 📄 **Upload & Process PDFs** – Extracts text and stores vector embeddings locally for fast retrieval.  
- 💬 **Interactive Q&A** – Ask natural language questions and get accurate answers with cited sources.  
- 📑 **Multi-Document Querying** – Choose which PDFs to include in a query.  
- 📌 **Suggested Question Templates** – Pre-defined, categorized sample questions for Finance, Company Overview, Strategic Insights, and ESG.  
- 🔍 **Optimized Chunking** – Improved document splitting for better context retrieval.  
- ⚡ **Configurable LLM Model** – Choose from multiple Google Gemini models at runtime.  
- 🛠 **Local Storage** – Keeps vector stores on disk for persistent performance.  

---

## 📦 Setup Instructions

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended for fast dependency installs)

### Installation

1. **Clone this repository**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Create and activate virtual environment**
```bash
# Install uv if not already installed
pip install uv

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate       # Windows
```

3. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

4. **Set your Google API key**  
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your-google-gemini-api-key
```

---

## ▶ Running the Application

```bash
streamlit run app.py
```

---

## 🖥 Usage Flow

1. **Upload PDFs** in the sidebar (e.g., *Cognizant Annual Report 2024*).  
2. **Process** the document to generate embeddings.  
3. Select one or more **sources** for querying.  
4. Use **Suggested Questions** buttons (grouped by category) or type your own.  
5. View **answers with sources** — includes page numbers, section titles, or file names.  

---

## 📂 Project Structure
```
├── app.py                 # Main Streamlit app with Suggested Questions UI
├── rag_engine.py          # RAG pipeline using LangChain + FAISS
├── utils.py               # PDF handling utilities
├── data/                  # PDF storage
│   └── PRO013686_8_Cognizant_ARS_2024_PR_LR.pdf
├── vector_store/          # Persistent FAISS embeddings
├── requirements.txt       # Dependencies
├── .env                   # API key
└── README.md              # Documentation
```

---

## 📌 Notes
- Default configuration uses **Gemini 2.0 Flash** for fast, cost-efficient responses.  
- Works best with **finance-related PDFs** but supports any text-based PDF.  
- Chunking is tuned for **long context retention** while ensuring quick retrieval.  
- Clickable **Suggested Questions** make it easier for non-technical users to start queries.
