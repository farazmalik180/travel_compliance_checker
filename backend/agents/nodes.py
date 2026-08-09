import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from backend.agents.state import AgentState
from backend.core.config import settings
from backend.services.rag_service import query_knowledge_base

FIA_COMPLIANCE_AGENT_SYSTEM_PROMPT = """
You are an expert AI Immigration Compliance Agent specialized in Pakistan's Federal Investigation Agency (FIA) departure rules and Integrated Border Management System (IBMS) guidelines. 

Your objective is to evaluate a traveler's uploaded documents and profile data against strict official FIA requirements before they head to the airport, preventing wrongful offloading or catching discrepancies early.

Use the provided Retrieved Rules from the FIA knowledge base to evaluate the specific constraints for this passenger based on their Destination and Visa Category.

OUTPUT FORMAT:
Return a structured JSON response containing:
- "status": "GREENLIGHT" or "ACTION_REQUIRED"
- "compliance_score": percentage or rating
- "verified_items": list of checks that passed successfully.
- "missing_or_incomplete_requirements": itemized checklist of specific documents or proofs the traveler must fix or acquire.
- "fia_rule_reference": brief explanation citing the official FIA guideline context.
"""

def extract_document_info(state: AgentState) -> AgentState:
    print("---EXTRACTING DOCUMENT INFO---")
    extracted = {
        "passport_validity": "> 6 months",
        "visa_status": "Valid",
        "passport_history": state.get("passport_history", "Experienced"),
        "has_protector_stamp": True if state.get("visa_category") == "WORK" else False,
        "has_return_ticket": True if state.get("visa_category") == "VISIT" else False,
        "has_hotel_booking": True if state.get("visa_category") == "VISIT" else False,
        "financial_proof": "Moderate", 
        "watchlist_flag": "Clear"
    }
    return {"extracted_data": extracted}

def retrieve_rules(state: AgentState) -> AgentState:
    print("---RETRIEVING RULES (RAG)---")
    destination = state.get("destination", "Unknown")
    visa_cat = state.get("visa_category", "Unknown")
    
    query = f"What are the FIA departure exit rules and requirements for a {visa_cat} visa to {destination}?"
    rules = query_knowledge_base(query)
    
    return {"retrieved_rules": rules}

def enhanced_scrutiny_check(state: AgentState) -> AgentState:
    print("---ENHANCED SCRUTINY FOR FRESH PASSPORT---")
    # This node is triggered only if passport history is "Fresh"
    # We mock the stricter checks. In a real system, this would evaluate specific documents.
    flags = [
        "Local ties proof not robust enough.",
        "Corporate employment verification pending."
    ]
    return {"enhanced_scrutiny_flags": flags}

def verify_compliance(state: AgentState) -> AgentState:
    print("---VERIFYING COMPLIANCE---")
    
    if not settings.GROQ_API_KEY:
        print("No Groq API key found, using mock compliance response.")
        return {"compliance_evaluation": {
            "status": "ACTION_REQUIRED",
            "compliance_score": "50%",
            "verified_items": ["Valid Passport"],
            "missing_or_incomplete_requirements": ["Provide API key for full check"],
            "fia_rule_reference": "N/A"
        }}
        
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", FIA_COMPLIANCE_AGENT_SYSTEM_PROMPT),
        ("human", "Passenger details:\nNationality: {nationality}\nDestination: {destination}\nVisa Category: {visa_category}\nPurpose: {purpose}.\n\nExtracted Doc Data: {extracted_data}\n\nEnhanced Scrutiny Flags (if any): {enhanced_flags}\n\nRetrieved Rules (RAG Context):\n{retrieved_rules}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "nationality": state.get("nationality"),
            "destination": state.get("destination"),
            "visa_category": state.get("visa_category"),
            "purpose": state.get("purpose"),
            "extracted_data": json.dumps(state.get("extracted_data", {})),
            "enhanced_flags": json.dumps(state.get("enhanced_scrutiny_flags", [])),
            "retrieved_rules": state.get("retrieved_rules", "")
        })
        evaluation = json.loads(response.content)
    except Exception as e:
        print(f"LLM Error: {e}")
        evaluation = {
            "status": "ERROR",
            "compliance_score": "0%",
            "verified_items": [],
            "missing_or_incomplete_requirements": [f"System error: {str(e)}"],
            "fia_rule_reference": ""
        }
        
    return {"compliance_evaluation": evaluation}

def audit_feedback(state: AgentState) -> AgentState:
    print("---AUDIT FEEDBACK---")
    eval_data = state.get("compliance_evaluation", {})
    
    return {
        "status": eval_data.get("status", "UNKNOWN"),
        "compliance_score": eval_data.get("compliance_score", "N/A"),
        "verified_items": eval_data.get("verified_items", []),
        "missing_or_incomplete_requirements": eval_data.get("missing_or_incomplete_requirements", []),
        "fia_rule_reference": eval_data.get("fia_rule_reference", "No reference available.")
    }
