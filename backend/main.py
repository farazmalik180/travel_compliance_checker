from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.core.config import settings
from backend.services.rag_service import query_knowledge_base
from backend.agents.graph import compliance_graph

app = FastAPI(
    title="Travel Compliance Checker API",
    description="Backend API for FIA departure rules compliance checking.",
    version="0.1.0"
)

class CheckComplianceRequest(BaseModel):
    nationality: str
    destination: str
    visa_category: str
    purpose: str
    passport_history: str
    profession: str = "N/A"
    bank_funds: str = "N/A"
    has_return_ticket: bool = False
    has_hotel_booking: bool = False
    has_financial_proof: bool = False
    has_protector_stamp: bool = False
    documents: List[dict]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class CheckComplianceResponse(BaseModel):
    status: str
    compliance_score: str
    verified_items: List[str]
    missing_or_incomplete_requirements: List[str]
    fia_rule_reference: str
    error: str = None

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.3"}

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    async def generate():
        if not settings.GROQ_API_KEY:
            yield "data: {\"error\": \"Missing Groq API Key\"}\n\n"
            return
            
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0.5, streaming=True)
        
        # Build history
        history = []
        user_latest = ""
        for msg in req.messages:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
                user_latest = msg.content
            elif msg.role == "assistant":
                history.append(AIMessage(content=msg.content))
                
        # Query RAG for context based on latest user message
        rag_context = query_knowledge_base(user_latest)
        
        sys_msg = SystemMessage(content=f"You are a helpful travel and immigration AI assistant for Pakistan's FIA. Use the following context from the FIA rulebook to answer questions if relevant. If it's a general travel question, you can answer it naturally.\n\nFIA RAG Context:\n{rag_context}")
        
        messages = [sys_msg] + history
        
        try:
            for chunk in llm.stream(messages):
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
            yield "data: {\"done\": true}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/check-compliance", response_model=CheckComplianceResponse)
def check_compliance(req: CheckComplianceRequest):
    # Non-streaming legacy endpoint
    state = {
        "nationality": req.nationality,
        "destination": req.destination,
        "visa_category": req.visa_category,
        "purpose": req.purpose,
        "passport_history": req.passport_history,
        "documents": req.documents
    }
    
    try:
        final_state = compliance_graph.invoke(state)
        return CheckComplianceResponse(
            status=final_state.get("status", "UNKNOWN"),
            compliance_score=final_state.get("compliance_score", "N/A"),
            verified_items=final_state.get("verified_items", []),
            missing_or_incomplete_requirements=final_state.get("missing_or_incomplete_requirements", []),
            fia_rule_reference=final_state.get("fia_rule_reference", "N/A")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/check-compliance-stream")
async def check_compliance_stream(req: CheckComplianceRequest):
    def event_generator():
        state = {
            "nationality": req.nationality,
            "destination": req.destination,
            "visa_category": req.visa_category,
            "purpose": req.purpose,
            "passport_history": req.passport_history,
            "profession": req.profession,
            "bank_funds": req.bank_funds,
            "has_return_ticket": req.has_return_ticket,
            "has_hotel_booking": req.has_hotel_booking,
            "has_financial_proof": req.has_financial_proof,
            "has_protector_stamp": req.has_protector_stamp,
            "documents": req.documents
        }
        
        try:
            final_node_state = {}
            for output in compliance_graph.stream(state):
                # LangGraph yields a dict with a single key (the node name)
                for node_name, node_state in output.items():
                    final_node_state = node_state
                    yield f"data: {json.dumps({'node': node_name})}\n\n"
            
            # After stream finishes, yield the final result
            yield f"data: {json.dumps({'final': final_node_state})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
