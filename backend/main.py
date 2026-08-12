from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import List
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
    documents: List[dict]

class CheckComplianceResponse(BaseModel):
    status: str
    compliance_score: str
    verified_items: List[str]
    missing_or_incomplete_requirements: List[str]
    fia_rule_reference: str
    error: str = None

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
