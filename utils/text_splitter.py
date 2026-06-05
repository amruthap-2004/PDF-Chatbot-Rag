from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.
    
    Args:
        documents (list[Document]): List of documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.
        
    Returns:
        list[Document]: A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)
