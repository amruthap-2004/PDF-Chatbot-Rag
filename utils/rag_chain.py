import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# System prompt directing the model to stay strictly within the provided context
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

Instructions:
1. Answer the question using ONLY the context provided below.
2. If the answer cannot be found in the context, or if the context is empty, you MUST reply with EXACTLY this sentence: "The answer was not found in the uploaded document."
3. Do not assume, extrapolate, or use any pre-trained knowledge outside of the context.
4. Keep the answer clear, factual, and concise.

Context:
{context}

Question:
{question}

Answer:"""

def ask_question(vector_store, question: str) -> tuple[str, list]:
    """
    Retrieves the top 3 relevant chunks from the vector store and queries the Gemini LLM
    to generate an answer strictly based on the retrieved context.
    
    Args:
        vector_store: The Chroma vector database instance.
        question (str): The user's query.
        
    Returns:
        tuple[str, list]: A tuple containing the generated answer (str) and
                          the list of retrieved source documents (list).
    """
    # Initialize the retriever to get the top 3 chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Retrieve documents using the modern invoke method
    docs = retriever.invoke(question)
    
    # Handle case where no documents are retrieved
    if not docs:
        return "The answer was not found in the uploaded document.", []
        
    # Format the context from retrieved chunks
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Initialize Gemini LLM with temperature 0.0 for deterministic factual answers
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Please set it in your .env file.")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0.0
    )
    
    # Create the prompt and chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    
    # Run the chain
    answer = chain.invoke({"context": context, "question": question})
    
    return answer.strip(), docs
