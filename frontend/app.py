# PIM - Pakistan Immigration Manager Frontend
import streamlit as st
import requests
import json
import base64
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL_STREAM = "http://127.0.0.1:8000/api/check-compliance-stream"
BACKEND_URL_CHAT = "http://127.0.0.1:8000/api/chat"

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
        t1 = st.checkbox("Do you have a confirmed return ticket with the same PNR?")
        
        is_fresh = st.session_state.profile.get("passport_history") == "Fresh"
        t2, t3 = True, True
        if is_fresh:
            t2 = st.checkbox("Do you have a confirmed hotel booking for your entire staying period?")
            t3 = st.checkbox("Do you have at least 1000 USD 'show money' (or equivalent)?")
        
        # Dynamically add Protector Stamp check if Work
        t4 = True
        if st.session_state.profile.get("purpose") == "Work":
            t4 = st.checkbox("Do you have a Protector Stamp on your passport?")
        
        if st.button("Submit Checks"):
            summary = f"Return Ticket: {'Yes' if t1 else 'No'}"
            if is_fresh:
                summary += f"\nHotel: {'Yes' if t2 else 'No'}\nShow Money: {'Yes' if t3 else 'No'}"
            if st.session_state.profile.get("purpose") == "Work":
                summary += f"\nProtector Stamp: {'Yes' if t4 else 'No'}"
                
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
                
                # Blacklist check for Fresh Passports
                blacklisted_destinations = ["cambodia", "uae", "myanmar", "laos", "thailand", "malaysia", "iraq"]
                is_fresh = st.session_state.profile.get("passport_history") == "Fresh"
                
                if is_fresh and dest.lower().strip() in blacklisted_destinations:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ **OFF-LOADING WARNING**: Fresh Passport holders are strictly restricted from traveling to high-risk destinations like {dest.title()} under current FIA directives. You cannot proceed."})
                    st.session_state.step = "offload_warning"
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload a scan of your Passport and Visa (Images or PDFs)."})
                    st.session_state.step = "upload"
                    
                st.rerun()
            else:
                st.warning("Please enter a destination.")
    
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
            st.session_state.profile["purpose"] = "Tourism" # hardcoded default
            st.session_state.step = "processing"
            st.rerun()
    
    elif st.session_state.step == "processing":
        with st.chat_message("assistant"):
            status_container = st.status("Initializing Agents...", expanded=True)
            
            try:
                response = requests.post(BACKEND_URL_STREAM, json=st.session_state.profile, stream=True)
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data = json.loads(decoded_line.removeprefix("data: "))
                            
                            if "node" in data:
                                node = data["node"]
                                if node == "extract":
                                    status_container.write("⚙️ **Vision Agent** is extracting OCR from your documents...")
                                elif node == "retrieve":
                                    status_container.write("📚 **RAG Agent** is querying FIA rule databases...")
                                elif node == "enhanced_scrutiny":
                                    status_container.write("🔎 **Scrutiny Agent** is triggering enhanced checks for Fresh passports...")
                                elif node == "verify":
                                    status_container.write("🛡️ **Verification Agent** is validating rules...")
                                elif node == "audit":
                                    status_container.write("✅ **Auditor Agent** is synthesizing the final report...")
                            
                            elif "final" in data:
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
                                
                            elif "error" in data:
                                status_container.error(data["error"])
                                
            except Exception as e:
                status_container.error(f"Connection failed: {e}")
    
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
                payload = {"messages": st.session_state.ai_messages}
                response = requests.post(BACKEND_URL_CHAT, json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line.removeprefix("data: ")
                            data = json.loads(data_str)
                            if "content" in data:
                                full_response += data["content"]
                                message_placeholder.markdown(full_response + "▌")
                            elif "error" in data:
                                st.error(data["error"])
                message_placeholder.markdown(full_response)
                st.session_state.ai_messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Chat error: {e}")
