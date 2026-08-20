from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.nodes import (
    extract_document_info,
    retrieve_rules,
    enhanced_scrutiny_check,
    verify_compliance,
    advocate_critic_node,
    audit_feedback
)

def should_scrutinize(state: AgentState):
    history = state.get("passport_history", "Experienced")
    if history == "Fresh":
        return "enhanced_scrutiny"
    return "verify"

def create_compliance_graph():
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract", extract_document_info)
    workflow.add_node("retrieve", retrieve_rules)
    workflow.add_node("enhanced_scrutiny", enhanced_scrutiny_check)
    workflow.add_node("verify", verify_compliance)
    workflow.add_node("advocate_critic", advocate_critic_node)
    workflow.add_node("audit", audit_feedback)
    
    # Define edges 
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "retrieve")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "retrieve",
        should_scrutinize,
        {
            "enhanced_scrutiny": "enhanced_scrutiny",
            "verify": "verify"
        }
    )
    
    workflow.add_edge("enhanced_scrutiny", "verify")
    workflow.add_edge("verify", "advocate_critic")
    workflow.add_edge("advocate_critic", "audit")
    workflow.add_edge("audit", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

compliance_graph = create_compliance_graph()
