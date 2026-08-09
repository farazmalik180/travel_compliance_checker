import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from backend.agents.state import AgentState
from backend.core.config import settings

FIA_COMPLIANCE_AGENT_SYSTEM_PROMPT = """
You are an expert AI Immigration Compliance Agent specialized in Pakistan's Federal Investigation Agency (FIA) departure rules and Integrated Border Management System (IBMS) guidelines. 

Your objective is to evaluate a traveler's uploaded documents and profile data against strict official FIA requirements before they head to the airport, preventing wrongful offloading or catching discrepancies early.

Evaluate the passenger based strictly on their declared Visa Category:

1. IF WORK VISA:
- Verify presence of a Valid Passport.
- Verify presence of a Valid Work Visa.
- Verify presence of the Protector Stamp (Bureau of Emigration).
- Check if a valid Work Permit is provided for the destination.

2. IF VISIT / TOURIST VISA:
- Verify presence of a Valid Passport.
- Verify a confirmed Return Ticket.
- Verify a Confirmed Hotel Booking with advance payment proof.
- Verify Sufficient Financial Proof (e.g., bank statements or valid credit cards showing adequate liquidity, e.g., equivalent to €2000+ for standard tourist hubs).
- Evaluate "Sound Profile" and return-intent indicators (employment ties, family roots in Pakistan).

3. IF GOVERNMENT SERVANT:
- Explicitly check for an original NOC (No Objection Certificate) issued by their parent government department.

4. SYSTEM & WATCHLIST CHECK SIMULATION:
- Simulate a check against travel restrictions (ECL/PNIL/BL flags).

OUTPUT FORMAT:
Return a structured JSON response containing:
- "status": "GREENLIGHT" or "ACTION_REQUIRED"
- "compliance_score": percentage or rating
- "verified_items": list of checks that passed successfully according to FIA rules.
- "missing_or_incomplete_requirements": itemized checklist of specific documents or proofs the traveler must fix or acquire to clear FIA airport counter screening safely.
- "fia_rule_reference": brief explanation citing the official FIA guideline context.
"""

def extract_document_info(state: AgentState) -> AgentState:
    """
    Agent 1: Extracts structured data from uploaded documents.
    In a real scenario, this would use OCR (like PyMuPDF or cloud vision).
    For the prototype, we simply assume some mock extracted data based on the provided inputs.
    """
    print("---EXTRACTING DOCUMENT INFO---")
    # Mocking extraction based on input for prototype purposes.
    # We assume if the user says "WORK", they uploaded work docs, etc.
    # In real app, this reads state['documents']
    extracted = {
        "passport_validity": "> 6 months",
        "visa_status": "Valid",
        "has_protector_stamp": True if state.get("visa_category") == "WORK" else False,
        "has_return_ticket": True if state.get("visa_category") == "VISIT" else False,
        "has_hotel_booking": True if state.get("visa_category") == "VISIT" else False,
        "financial_proof": "Moderate", 
        "watchlist_flag": "Clear"
    }
    return {"extracted_data": extracted}

def retrieve_rules(state: AgentState) -> AgentState:
    """
    Agent 2: RAG lookup for local immigration laws using LlamaIndex.
    For this prototype, we simulate a retrieved rule string.
    """
    print("---RETRIEVING RULES (RAG)---")
    destination = state.get("destination", "Unknown")
    # Mock retrieval
    rules = f"Standard FIA rules apply for departure to {destination}. Ensure protector stamp for work, or return ticket/hotel for visit."
    return {"retrieved_rules": rules}

def verify_compliance(state: AgentState) -> AgentState:
    """
    Agent 3: Evaluates compliance against rules using the LLM.
    """
    print("---VERIFYING COMPLIANCE---")
    
    # In a full app without OPENAI_API_KEY set, we'd mock the response to avoid crash.
    # For now, we will use a ChatOpenAI model, ensuring the API key is expected.
    # If the user doesn't have an API key, we will fall back to a mock for the prototype to keep it runnable.
    if not settings.GROQ_API_KEY:
        print("No Groq API key found, using mock compliance response.")
        mock_response = {
            "status": "GREENLIGHT" if state.get("visa_category") in ["WORK", "VISIT"] else "ACTION_REQUIRED",
            "compliance_score": "90%",
            "verified_items": ["Valid Passport", "Valid Visa"],
            "missing_or_incomplete_requirements": ["Confirm financial proof"] if state.get("visa_category") == "VISIT" else [],
            "fia_rule_reference": "FIA Standard guidelines section 4(b)."
        }
        return {"compliance_evaluation": mock_response}
        
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", FIA_COMPLIANCE_AGENT_SYSTEM_PROMPT),
        ("human", "Passenger details: Nationality: {nationality}, Destination: {destination}, Visa Category: {visa_category}, Purpose: {purpose}.\n\nExtracted Doc Data: {extracted_data}\n\nRetrieved Rules: {retrieved_rules}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "nationality": state.get("nationality"),
            "destination": state.get("destination"),
            "visa_category": state.get("visa_category"),
            "purpose": state.get("purpose"),
            "extracted_data": json.dumps(state.get("extracted_data", {})),
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
    """
    Agent 4: Parses the evaluation and sets the final state for the UI.
    """
    print("---AUDIT FEEDBACK---")
    eval_data = state.get("compliance_evaluation", {})
    
    return {
        "status": eval_data.get("status", "UNKNOWN"),
        "compliance_score": eval_data.get("compliance_score", "N/A"),
        "verified_items": eval_data.get("verified_items", []),
        "missing_or_incomplete_requirements": eval_data.get("missing_or_incomplete_requirements", []),
        "fia_rule_reference": eval_data.get("fia_rule_reference", "No reference available.")
    }
