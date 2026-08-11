from fastapi import FastAPI
from pydantic import BaseModel
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
    documents: List[dict]

class CheckComplianceResponse(BaseModel):
    status: str
    compliance_score: str
    verified_items: List[str]
    missing_or_incomplete_requirements: List[str]
    fia_rule_reference: str
    error: str = None

@app.post("/api/check-compliance", response_model=CheckComplianceResponse)
async def check_compliance(req: CheckComplianceRequest):
    initial_state = {
        "nationality": req.nationality,
        "destination": req.destination,
        "visa_category": req.visa_category,
        "purpose": req.purpose,
        "passport_history": req.passport_history,
        "documents": req.documents
    }
    
    # Invoke the LangGraph workflow
    result_state = compliance_graph.invoke(initial_state)
    
    return CheckComplianceResponse(
        status=result_state.get("status", "ERROR"),
        compliance_score=result_state.get("compliance_score", "N/A"),
        verified_items=result_state.get("verified_items", []),
        missing_or_incomplete_requirements=result_state.get("missing_or_incomplete_requirements", []),
        fia_rule_reference=result_state.get("fia_rule_reference", ""),
        error=result_state.get("error")
    )
