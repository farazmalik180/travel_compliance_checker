# PIM - Pakistan Immigration Manager Frontend
import streamlit as st
from backend.agents.graph import create_compliance_graph
from backend.services.rag_service import query_knowledge_base
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.core.config import settings

import json
import base64
import os
import sys
import subprocess
import socket
import time
from dotenv import load_dotenv

load_dotenv()

APP_VERSION = "1.0.4"


st.set_page_config(page_title="PIM (Pakistan Immigration Manager)", page_icon="🛂", layout="centered")

st.title("🛂 PIM - Pakistan Immigration Manager")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to PIM. To begin, what is your **Passport History**?"}
    ]
if "step" not in st.session_state:
    st.session_state.step = "passport_history"
if "profile" not in st.session_state:
    st.session_state.profile = {}

if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = [
        {"role": "assistant", "content": "Hello! I am the PIM FIA Assistant. How can I help you today?"}
    ]

with st.sidebar:
    st.title("Controls")
    if st.button("🔄 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

tab1, tab2 = st.tabs(["📋 Compliance Wizard", "💬 Direct AI Chat"])

with tab1:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Step 1: Passport History
    if st.session_state.step == "passport_history":
        st.write("Please select an option:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Fresh Passport", use_container_width=True):
                st.session_state.profile["passport_history"] = "Fresh"
                st.session_state.messages.append({"role": "user", "content": "Fresh Passport"})
                st.session_state.messages.append({"role": "assistant", "content": "Please specify the **Purpose of your visit**."})
                st.session_state.step = "travel_purpose"
                st.rerun()
        with col2:
            if st.button("Experienced Passport", use_container_width=True):
                st.session_state.profile["passport_history"] = "Experienced"
                st.session_state.messages.append({"role": "user", "content": "Experienced Passport"})
                st.session_state.messages.append({"role": "assistant", "content": "Please specify the **Purpose of your visit**."})
                st.session_state.step = "travel_purpose"
                st.rerun()
    
    # Step 2: Purpose of Visit
    elif st.session_state.step == "travel_purpose":
        purpose = st.radio("Purpose of Visit", ["Work", "Visit", "Student", "Government Servant"], horizontal=True)
        if st.button("Next"):
            st.session_state.profile["purpose"] = purpose
            st.session_state.profile["visa_category"] = purpose.upper() # Align with backend models
            st.session_state.messages.append({"role": "user", "content": f"Purpose: {purpose}"})
            st.session_state.messages.append({"role": "assistant", "content": "Thank you. Now please complete the mandatory pre-flight checks below."})
            st.session_state.step = "mandatory_checks"
            st.rerun()
    
    # Step 3: Mandatory Pre-flight Checks
    elif st.session_state.step == "mandatory_checks":
        st.write("**Mandatory Prerequisites for Departure**")
        
        is_student = st.session_state.profile.get("purpose") == "Student"
        is_fresh = st.session_state.profile.get("passport_history") == "Fresh"
        
        # Student visa holders do not require a return ticket
        if is_student:
            t1 = True
        else:
            t1 = st.checkbox("Do you have a confirmed return ticket with the same PNR?")
            
        t2, t3 = True, True
        if is_fresh:
            # Student visa holders do not require a hotel booking
            if is_student:
                t2 = True
            else:
                t2 = st.checkbox("Do you have a confirmed hotel booking for your entire staying period?")
            t3 = st.checkbox("Do you have at least 1000 USD 'show money' (or equivalent)?")
        
        # Dynamically add Protector Stamp check if Work
        t4 = True
        if st.session_state.profile.get("purpose") == "Work":
            t4 = st.checkbox("Do you have a Protector Stamp on your passport?")
        
        if st.button("Submit Checks"):
            summary_parts = []
            if not is_student:
                summary_parts.append(f"Return Ticket: {'Yes' if t1 else 'No'}")
            if is_fresh:
                if not is_student:
                    summary_parts.append(f"Hotel: {'Yes' if t2 else 'No'}")
                summary_parts.append(f"Show Money: {'Yes' if t3 else 'No'}")
            if st.session_state.profile.get("purpose") == "Work":
                summary_parts.append(f"Protector Stamp: {'Yes' if t4 else 'No'}")
            
            summary = "\n".join(summary_parts) if summary_parts else "No additional checks needed."
                
            st.session_state.messages.append({"role": "user", "content": summary})
            
            if t1 and t2 and t3 and t4:
                st.session_state.profile["has_return_ticket"] = t1
                st.session_state.profile["has_hotel_booking"] = t2
                st.session_state.profile["has_financial_proof"] = t3
                st.session_state.profile["has_protector_stamp"] = t4 if st.session_state.profile.get("purpose") == "Work" else True # Set True for non-work to pass LLM
                
                st.session_state.messages.append({"role": "assistant", "content": "All mandatory checks passed. Where are you traveling to?"})
                st.session_state.step = "destination"
            else:
                st.session_state.messages.append({"role": "assistant", "content": "❌ **OFF-LOADING WARNING**: You do not meet the mandatory prerequisites for departure. You cannot proceed."})
                st.session_state.step = "offload_warning"
                
            st.rerun()
    
    # Step 4: Destination
    elif st.session_state.step == "destination":
        dest = st.text_input("Destination Country (e.g. Cambodia, UAE)")
        if st.button("Next "):
            if dest:
                st.session_state.profile["destination"] = dest
                st.session_state.messages.append({"role": "user", "content": dest})
                
                # Blacklist check
                blacklisted_destinations = ["cambodia", "uae", "myanmar", "laos", "thailand", "malaysia", "iraq"]
                is_fresh = st.session_state.profile.get("passport_history") == "Fresh"
                dest_lower = dest.lower().strip()
                
                if dest_lower in blacklisted_destinations:
                    if is_fresh:
                        st.session_state.messages.append({"role": "assistant", "content": f"❌ **OFF-LOADING WARNING**: Fresh Passport holders are strictly restricted from traveling to high-risk destinations like {dest.title()} under current FIA directives. You cannot proceed."})
                        st.session_state.step = "offload_warning"
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ You are traveling to a high-risk destination ({dest.title()}). Please provide your current bank balance (PKR)."})
                        st.session_state.step = "experienced_bank_check"
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload a scan of your Passport and Visa (Images or PDFs)."})
                    st.session_state.step = "upload"
                    
                st.rerun()
            else:
                st.warning("Please enter a destination.")
                
    # Step 4.5: Experienced Bank Check
    elif st.session_state.step == "experienced_bank_check":
        balance = st.number_input("Current Bank Balance (PKR)", min_value=0, step=100000)
        if st.button("Verify Balance"):
            st.session_state.profile["bank_funds"] = f"{balance} PKR"
            st.session_state.messages.append({"role": "user", "content": f"Bank Balance: {balance:,.0f} PKR"})
            
            if balance > 1000000:
                st.session_state.messages.append({"role": "assistant", "content": "✅ Balance verified. Please upload a scan of your Passport and Visa (Images or PDFs)."})
                st.session_state.step = "upload"
            else:
                st.session_state.messages.append({"role": "assistant", "content": "❌ **OFF-LOADING WARNING**: For high-risk destinations, a minimum bank balance of > 1,000,000 PKR is required. You cannot proceed."})
                st.session_state.step = "offload_warning"
                
            st.rerun()
    
    # Step 5: Upload Document & Trigger Stream
    elif st.session_state.step == "upload":
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])
        if st.button("Run Compliance Check", type="primary"):
            st.session_state.messages.append({"role": "user", "content": "Uploaded documents. Please check compliance."})
            # Process files
            docs_payload = []
            if uploaded_files:
                for f in uploaded_files:
                    base64_str = base64.b64encode(f.read()).decode('utf-8')
                    docs_payload.append({
                        "filename": f.name,
                        "content_type": f.type,
                        "content": base64_str
                    })
            st.session_state.profile["documents"] = docs_payload
            st.session_state.profile["nationality"] = "Pakistani" # hardcoded default
            st.session_state.step = "processing"
            st.rerun()
    
    elif st.session_state.step == "processing":
        with st.chat_message("assistant"):
            status_container = st.status("Initializing Agents...", expanded=True)
            
            try:
                compliance_graph = create_compliance_graph()
                for output in compliance_graph.stream(st.session_state.profile):
                    for node_name, node_state in output.items():
                        data = {"node": node_name}
                        
                        if node_name == "extract":
                            status_container.write("⚙️ **Vision Agent** is extracting OCR from your documents...")
                        elif node_name == "retrieve":
                            status_container.write("📚 **RAG Agent** is querying FIA rule databases...")
                        elif node_name == "enhanced_scrutiny":
                            status_container.write("🔎 **Scrutiny Agent** is triggering enhanced checks for Fresh passports...")
                        elif node_name == "verify":
                            status_container.write("🛡️ **Verification Agent** is validating rules...")
                        elif node_name == "advocate_critic":
                            status_container.write("⚖️ **Advocate/Critic Agent** is searching for mitigating factors...")
                        elif node_name == "audit":
                            status_container.write("✅ **Auditor Agent** is synthesizing the final report...")
                            data["final"] = node_state
                        
                        if "final" in data:
                            final_result = data["final"]
                            status_container.update(label="Workflow Complete!", state="complete", expanded=False)
                            
                            status = final_result.get("status")
                            score = final_result.get("compliance_score")
                            
                            output_md = f"### Final Decision: {status}\n**Compliance Score:** {score}\n\n"
                            output_md += "**Verified:**\n"
                            for i in final_result.get("verified_items", []):
                                output_md += f"- ✅ {i}\n"
                            output_md += "\n**Missing / Incomplete:**\n"
                            for i in final_result.get("missing_or_incomplete_requirements", []):
                                output_md += f"- ❌ {i}\n"
                                
                            output_md += f"\n*FIA Reference:* {final_result.get('fia_rule_reference')}"
                            
                            st.markdown(output_md)
                            st.session_state.messages.append({"role": "assistant", "content": output_md})
                            st.session_state.step = "done"
            except Exception as e:
                status_container.error(f"Execution failed: {e}")

    
    elif st.session_state.step in ["done", "offload_warning"]:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

with tab2:
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask me about FIA rules or travel guidelines..."):
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.5, streaming=True)
                
                history = []
                user_latest = ""
                for msg in st.session_state.ai_messages:
                    if msg["role"] == "user":
                        history.append(HumanMessage(content=msg["content"]))
                        user_latest = msg["content"]
                    elif msg["role"] == "assistant":
                        history.append(AIMessage(content=msg["content"]))
                        
                rag_context = query_knowledge_base(user_latest)
                sys_msg = SystemMessage(content=f"You are a helpful travel and immigration AI assistant for Pakistan's FIA. Use the following context from the FIA rulebook to answer questions if relevant. If it's a general travel question, you can answer it naturally.\n\nFIA RAG Context:\n{rag_context}")
                
                messages = [sys_msg] + history
                
                for chunk in llm.stream(messages):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                    
                message_placeholder.markdown(full_response)
                st.session_state.ai_messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Chat error: {e}")
