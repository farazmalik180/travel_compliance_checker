import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from backend.core.config import settings

def setup_rag():
    if not settings.GROQ_API_KEY:
        print("Warning: GROQ_API_KEY not set. RAG might fail during synthesis.")
        
    # Setup Groq LLM for synthesis
    Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=settings.GROQ_API_KEY)
    
    # Setup local HuggingFace embeddings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Load document
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    try:
        documents = SimpleDirectoryReader(input_files=[os.path.join(data_dir, "fia_exit_rules.pdf")]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index
    except Exception as e:
        print(f"Error loading RAG index: {e}")
        return None

# Initialize index globally so it's only loaded once when server starts
print("Initializing RAG Index...")
rag_index = setup_rag()

def query_knowledge_base(query: str) -> str:
    """Queries the LlamaIndex engine for relevant rules."""
    if not rag_index:
        return "RAG engine not initialized or missing fia_exit_rules.pdf."
    
    try:
        query_engine = rag_index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error querying RAG: {e}"
