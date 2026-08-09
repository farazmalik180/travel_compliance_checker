from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.nodes import (
    extract_document_info,
    retrieve_rules,
    verify_compliance,
    audit_feedback
)

def create_compliance_graph():
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract", extract_document_info)
    workflow.add_node("retrieve", retrieve_rules)
    workflow.add_node("verify", verify_compliance)
    workflow.add_node("audit", audit_feedback)
    
    # Define edges (straight linear workflow)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_edge("verify", "audit")
    workflow.add_edge("audit", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

compliance_graph = create_compliance_graph()
