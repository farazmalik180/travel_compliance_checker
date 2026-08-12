import os
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.groq import Groq
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
from backend.core.config import settings

def setup_rag():
    if not LLAMA_INDEX_AVAILABLE:
        print("LlamaIndex dependencies missing. RAG is disabled.")
        return None
        
    print("Setting up RAG and Vector DB...")
    # Initialize the LLM and Embedding models
    llm = Groq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Check if the PDF rules document exists
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fia_exit_rules.pdf")
    if not os.path.exists(pdf_path):
        print(f"Warning: {pdf_path} not found. RAG will have no data.")
        return None
        
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index

# Global index instance
_index = setup_rag()

def query_knowledge_base(query: str) -> str:
    
    try:
        query_engine = _index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying RAG: {e}"
