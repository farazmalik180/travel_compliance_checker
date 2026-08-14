from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    """
    The state of the multi-agent workflow.
    """
    # Inputs
    nationality: str
    destination: str
    visa_category: str
    purpose: str
    passport_history: str # e.g. "Fresh", "Experienced"
    profession: str
    bank_funds: str
    has_return_ticket: bool
    has_hotel_booking: bool
    has_financial_proof: bool
    has_protector_stamp: bool
    documents: List[dict] # payload containing base64 images
    
    # Internal state passed between agents
    extracted_data: Optional[Dict[str, Any]]
    retrieved_rules: Optional[str]
    enhanced_scrutiny_flags: Optional[List[str]]
    compliance_evaluation: Optional[Dict[str, Any]]
    
    # Final outputs
    status: Optional[str]
    compliance_score: Optional[str]
    verified_items: Optional[List[str]]
    missing_or_incomplete_requirements: Optional[List[str]]
    fia_rule_reference: Optional[str]
    
    # Track errors if any
    error: Optional[str]
