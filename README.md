# 🛡️ Defence RAG System

An intelligent, Retrieval-Augmented Generation (RAG) system built to parse, search, and reason over Indian Naval Regulations and Defence Procurement Policies. 

This project provides a robust backend pipeline and a beautifully designed, transparent Streamlit frontend to interactively query complex regulatory documents with perfect citations.

## ✨ Why We Built This
Navigating massive regulatory documents like the `RegsNavyIV.pdf` is time-consuming and error-prone. This project was developed to:
1. **Automate Document Retrieval:** Quickly find the exact page and section for any query.
2. **Synthesize Complex Rules:** Use advanced LLMs to answer complex questions without hallucinating.
3. **Ensure Transparency:** Provide a UI that explicitly shows *what* the system is doing, *how long* it takes, and *which exact texts* it's relying on.

## 🏗️ Architecture
The system follows a modular, best-practice RAG architecture:
- **Ingestion (`ingest/`)**: Uses `pdfplumber` to extract text from PDFs, and `RecursiveCharacterTextSplitter` to cleanly chunk the text (1000 chars, 200 overlap).
- **Embeddings & Vector Store (`retriever/`)**: Uses Google's `gemini-embedding-001` with exponential backoff for rate limits, storing vectors persistently in `ChromaDB`.
- **Generation (`generator/`)**: Prompts the LLM to strictly rely on retrieved context and generate citations.
- **Evaluation (`evaluator/`)**: Computes Retrieval Precision, MRR, Citation Validity, and Hallucination rates.

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` (or create a new `.env` file) and add your API key:
```env
EMBEDDING_PROVIDER=gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Run the Application
The project includes a beautiful Streamlit UI. Start it using:
```bash
python -m streamlit run app.py
```
*Open your browser and navigate to `http://localhost:8501`.*

### 5. Using the CLI (Backend Only)
You can also run the system entirely from the command line:
```bash
# Ingest new documents from the /source folder
python main.py ingest --data-dir ./source --reset

# Ask a direct question
python main.py query "What are the eligibility criteria for recruitment?" --verbose

# Run the automated evaluation suite
python main.py evaluate --samples
```

## 📊 Features
- **Dynamic UI:** Glassmorphism design, real-time step trackers, and detailed timing metrics.
- **Rate Limit Handlers:** Automatic exponential backoff and batching (20 chunks/batch) to seamlessly work on the Gemini free tier.
- **Strict Grounding:** The model is explicitly trained to refuse unanswerable questions to prevent hallucinations.
