# PDF RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that allows you to upload any PDF document, parse and store its contents in a local vector store, and ask questions about it using Google Gemini's powerful Large Language Model (LLM).

The application is built on top of **Streamlit**, **LangChain**, **ChromaDB**, and the **Google Gemini API**.

---

## 🚀 Key Features

* **PDF Upload & Parsing:** Upload documents directly in the UI and parse them using `pypdf`.
* **Smart Text Chunking:** Employs recursive character text-splitting to retain context structure.
* **Vector Storage:** Generates embeddings using Google Gemini Embeddings and persists them locally with ChromaDB.
* **Intelligent QA Retrieval:** Leverages LangChain to retrieve contextually matching chunks and query the Gemini LLM.
* **Persistent DB State:** Remembers your document across app restarts so you don't have to re-upload.
* **Strict Factuality:** Instructed to answer queries strictly based on document facts; falls back gracefully if answers aren't in the text.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **RAG Orchestration:** LangChain
* **Vector DB:** ChromaDB
* **LLM & Embeddings:** Google Gemini API
* **PDF Reader:** PyPDF
* **Env Config:** python-dotenv

---

## 📋 Project Structure

```text
pdf-rag-chatbot/
│
├── app.py                 # Main Streamlit application entry point
├── requirements.txt       # Project python dependencies
├── README.md              # Setup instructions
├── .env.example           # Template for environment variables
├── data/                  # Folder for temporary local files
├── chroma_db/             # Folder for persistent Chroma database files
│
└── utils/                 # Modular utility functions
    ├── __init__.py
    ├── pdf_loader.py      # PDF text extraction
    ├── text_splitter.py   # Text splitting and chunking
    ├── vector_store.py    # Embedding creation and Chroma setup
    └── rag_chain.py       # Retrieve context and generate response via Gemini
```

---

## ⚙️ Setup Instructions

Follow these step-by-step instructions to get the application running locally:

### Step 1: Check Python Installation
Make sure you have Python 3.8+ and `pip` installed:
```bash
python --version
pip --version
```

### Step 2: Create Project Folder
```bash
mkdir pdf-rag-chatbot
cd pdf-rag-chatbot
```

### Step 3: Create Virtual Environment
Create an isolated Python virtual environment to manage dependencies:
```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment
Activate the virtual environment depending on your operating system:
* **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\activate
  ```
* **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 5: Install Dependencies
Install all required libraries specified in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 6: Create `.env` File
Create a `.env` file to store your API key securely. Copy the example file:
```bash
cp .env.example .env
```
Open the newly created `.env` file and replace the placeholder with your actual Google Gemini API key:
```env
GOOGLE_API_KEY=AIzaSy...your_gemini_api_key...
```
*(You can obtain an API key for free from the [Google AI Studio](https://aistudio.google.com/)).*

### Step 7: Run Application
Start the Streamlit development server:
```bash
streamlit run app.py
```

### Step 8: Open Browser
The application will automatically launch in your default browser. If it doesn't, navigate to:
```text
http://localhost:8501
```

---

## 🛡️ Error Handling Details

The application features graceful handling of the following conditions:
* **Missing API Key:** Warns the user and provides an input field in the sidebar to supply the key dynamically.
* **Corrupt/Encrypted PDFs:** Catches exceptions and returns user-friendly messages detailing why the file cannot be processed.
* **Factual Fallback:** If the answer is unavailable in the PDF context, the chatbot responds with exactly: *"The answer was not found in the uploaded document."*
* **Empty Queries & Actions:** Validates input fields before processing to prevent accidental API requests.
