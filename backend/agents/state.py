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
    documents: List[str] # could be paths or base64 text for prototype
    
    # Internal state passed between agents
    extracted_data: Optional[Dict[str, Any]]
    retrieved_rules: Optional[str]
    compliance_evaluation: Optional[Dict[str, Any]]
    
    # Final outputs
    status: Optional[str]
    compliance_score: Optional[str]
    verified_items: Optional[List[str]]
    missing_or_incomplete_requirements: Optional[List[str]]
    fia_rule_reference: Optional[str]
    
    # Track errors if any
    error: Optional[str]
