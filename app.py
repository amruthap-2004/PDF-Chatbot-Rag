import os
import shutil
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local utility modules
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.vector_store import create_vector_store, load_vector_store
from utils.rag_chain import ask_question

# Configuration directories
DB_DIR = "chroma_db"
DATA_DIR = "data"
META_FILE = os.path.join(DB_DIR, "meta.txt")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Helper functions for managing database metadata
def save_meta(filename: str):
    with open(META_FILE, "w", encoding="utf-8") as f:
        f.write(filename)

def load_meta() -> str | None:
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

def clear_meta():
    if os.path.exists(META_FILE):
        try:
            os.remove(META_FILE)
        except Exception:
            pass

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

# Attempt to restore the vector store from disk on startup
if st.session_state.vector_store is None:
    loaded_store = load_vector_store(DB_DIR)
    if loaded_store is not None:
        st.session_state.vector_store = loaded_store
        st.session_state.processed_filename = load_meta() or "Uploaded Document"

# Streamlit Page Configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern, custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;700;800&display=swap');
    
    /* Global font styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient styling */
    .title-gradient {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8383 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #7d8590;
        margin-bottom: 2rem;
    }
    
    /* Styled container card for answers */
    .answer-card {
        border-radius: 12px;
        padding: 24px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Styled context snippet block */
    .source-snippet {
        background-color: rgba(0, 0, 0, 0.25);
        padding: 12px;
        border-left: 4px solid #FF4B4B;
        font-family: 'Inter', monospace;
        font-size: 0.85rem;
        border-radius: 4px;
        margin-top: 8px;
        white-space: pre-wrap;
    }
    
    /* Active file badge */
    .active-file-badge {
        display: inline-block;
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Area
st.markdown("<h1 class='title-gradient'>🤖 PDF RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>A professional Retrieval-Augmented Generation chatbot powered by LangChain, ChromaDB, and Google Gemini.</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/bot.png", width=80)
    st.markdown("### 🛠️ Configuration & Control")
    st.write("Follow these steps to query your documents:")
    
    # 1. API Key Check & Configuration
    api_key = os.getenv("GOOGLE_API_KEY")
    api_key_valid = True
    
    if not api_key:
        st.warning("⚠️ GOOGLE_API_KEY not found in .env file.")
        api_key_input = st.text_input("Enter Gemini API Key:", type="password")
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
            api_key = api_key_input
        else:
            api_key_valid = False
            st.error("Please enter an API key or configure it in the .env file to proceed.")
            
    st.markdown("---")
    
    # 2. PDF Document Upload
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    process_btn = st.button("Process Document", use_container_width=True, disabled=not api_key_valid)
    
    # Action when "Process Document" is clicked
    if process_btn:
        if not uploaded_file:
            st.error("❌ Please upload a PDF file first.")
        else:
            # Temporary file path
            temp_file_path = os.path.join(DATA_DIR, "temp.pdf")
            
            try:
                # Save uploaded file locally
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # RAG Setup Steps with explicit loading spinners
                with st.spinner("Step 1/3: Extracting text from PDF..."):
                    docs = load_pdf(temp_file_path)
                    
                with st.spinner("Step 2/3: Chunking document content..."):
                    chunks = split_documents(docs)
                    
                with st.spinner("Step 3/3: Embedding chunks & generating Chroma DB..."):
                    vector_store = create_vector_store(chunks, DB_DIR)
                    save_meta(uploaded_file.name)
                    
                # Update Session State
                st.session_state.vector_store = vector_store
                st.session_state.processed_filename = uploaded_file.name
                
                # Cleanup local temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
                st.success(f"✅ Successfully processed: {uploaded_file.name}")
                st.rerun()
                
            except ValueError as ve:
                st.error(f"❌ Parsing Error: {str(ve)}")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    st.error("❌ API Rate Limit Exceeded (429 / RESOURCE_EXHAUSTED). Please wait 60 seconds for your free tier quota to reset, then click 'Process Document' again.")
                else:
                    st.error(f"❌ Failed to process PDF: {str(e)}")
                
    st.markdown("---")
    
    # 3. Database State Management & Cleanup
    st.markdown("### 💾 Vector Database Status")
    if st.session_state.vector_store is not None:
        st.success("Database Status: Ready")
        st.markdown(f"**Loaded Document:**\n`{st.session_state.processed_filename}`")
        
        # Reset Database button
        if st.button("Reset / Clear Database", type="secondary", use_container_width=True):
            # Clear metadata and session variables
            clear_meta()
            
            # Delete collection from existing database to free locked SQLite files
            if st.session_state.vector_store is not None:
                try:
                    st.session_state.vector_store.delete_collection()
                except Exception:
                    pass
                    
            st.session_state.vector_store = None
            st.session_state.processed_filename = None
            
            # Delete Chroma DB folder files
            if os.path.exists(DB_DIR):
                try:
                    shutil.rmtree(DB_DIR)
                    os.makedirs(DB_DIR, exist_ok=True)
                except Exception as e:
                    st.warning("Database directory cleared, but some files are locked. They will be cleaned on next restart.")
            
            st.success("Database reset successfully.")
            st.rerun()
    else:
        st.info("Database Status: Empty")

# Main Area Chat & QA Logic
if st.session_state.vector_store is None:
    st.info("👈 Please configure your API key (if needed) and upload a PDF in the sidebar to begin.")
else:
    # Display Badge for active file
    st.markdown(f"<div class='active-file-badge'>📁 Active Document: {st.session_state.processed_filename}</div>", unsafe_allow_html=True)
    
    # User Question Form
    with st.form(key="qa_form"):
        user_query = st.text_input("Ask a question about the document:", placeholder="What is the main topic of the document?")
        submit_btn = st.form_submit_button("Submit Question", type="primary")
        
    if submit_btn:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question before submitting.")
        elif not api_key:
            st.error("❌ Google API Key is missing. Please set it in the sidebar.")
        else:
            try:
                # Query RAG Chain
                with st.spinner("Rummaging through document context and generating answer..."):
                    answer, sources = ask_question(st.session_state.vector_store, user_query)
                    
                # Display Answer in card layout
                st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
                st.markdown("### 🤖 Answer")
                st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display Sources if available and answer wasn't a fallback
                if sources and answer != "The answer was not found in the uploaded document.":
                    with st.expander("📚 View Source Snippets (Top 3 Chunks)"):
                        for idx, doc in enumerate(sources):
                            page_num = doc.metadata.get("page", "Unknown")
                            source_name = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Chunk {idx+1}** (Source: `{source_name}`, Page `{page_num}`):")
                            st.markdown(f"<div class='source-snippet'>{doc.page_content}</div>", unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    st.error(f"❌ API Rate Limit Exceeded (429 / RESOURCE_EXHAUSTED). Details: {str(e)}")
                else:
                    st.error(f"❌ Error occurred during query: {str(e)}")
                    st.info("Tip: Double-check your API key validity and internet connection.")
