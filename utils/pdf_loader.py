import os
from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF file and extracts text from each page, returning a list of LangChain Documents.
    
    Args:
        file_path (str): The path to the PDF file.
        
    Returns:
        list[Document]: A list of LangChain Document objects.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or is encrypted/corrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
        
    documents = []
    try:
        reader = PdfReader(file_path)
        
        # Check if the PDF is encrypted and attempt decryption
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("The PDF file is encrypted and cannot be processed.")
                
        # Loop through pages and extract text
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Store page number (1-based index) in metadata
                metadata = {"source": os.path.basename(file_path), "page": page_num + 1}
                documents.append(Document(page_content=text, metadata=metadata))
                
        if not documents:
            raise ValueError(
                "No readable text could be extracted from the PDF. "
                "The file may be corrupted, empty, or contain only scanned images without OCR."
            )
            
        return documents
        
    except Exception as e:
        # If it's already a ValueError we raised, propagate it
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Error parsing PDF file: {str(e)}")
