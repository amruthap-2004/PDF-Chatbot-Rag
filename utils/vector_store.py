import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings.
    Checks for the GOOGLE_API_KEY environment variable.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

def create_vector_store(documents: list[Document], persist_directory: str) -> Chroma:
    """
    Creates a new Chroma vector store from the provided documents.
    Deletes any existing database collection to start fresh.
    
    Args:
        documents (list[Document]): The chunked documents to store.
        persist_directory (str): Directory where Chroma DB files will be stored.
        
    Returns:
        Chroma: The initialized Chroma vector store instance.
    """
    embeddings = get_embeddings()
    
    # Try to delete the collection first to clear data and metadata in SQLite
    if os.path.exists(persist_directory):
        try:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            db.delete_collection()
        except Exception:
            pass
            
        try:
            shutil.rmtree(persist_directory)
        except Exception:
            pass

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database (automatic in newer versions, but safe to call if available)
    if hasattr(vector_store, "persist"):
        vector_store.persist()
        
    return vector_store

def load_vector_store(persist_directory: str) -> Chroma | None:
    """
    Loads an existing Chroma vector store from the persist_directory.
    
    Args:
        persist_directory (str): Directory where Chroma DB files are stored.
        
    Returns:
        Chroma: The loaded Chroma vector store instance, or None if it doesn't exist.
    """
    # Check if the directory exists and is not empty
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return None
        
    try:
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vector_store
    except Exception:
        # If there's an error loading (e.g. corrupt DB), return None
        return None
