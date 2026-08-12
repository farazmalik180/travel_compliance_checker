import streamlit as st
import requests
import json
import base64
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL_STREAM = "http://127.0.0.1:8000/api/check-compliance-stream"

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
            st.session_state.messages.append({"role": "assistant", "content": "Since you hold a Fresh Passport, please provide your **Current Profession** and **Bank Account Balance** (e.g. 500,000 PKR)."})
            st.session_state.step = "fresh_profiling"
            st.rerun()
    with col2:
        if st.button("Experienced Passport", use_container_width=True):
            st.session_state.profile["passport_history"] = "Experienced"
            st.session_state.messages.append({"role": "user", "content": "Experienced Passport"})
            st.session_state.messages.append({"role": "assistant", "content": "Got it. Please complete the mandatory pre-flight checks below."})
            st.session_state.step = "mandatory_checks"
            st.rerun()

# Step 2 (Fresh Only): Profession & Funds
elif st.session_state.step == "fresh_profiling":
    prof = st.text_input("Current Profession")
    funds = st.text_input("Bank Account Balance")
    if st.button("Next"):
        if prof and funds:
            st.session_state.profile["profession"] = prof
            st.session_state.profile["bank_funds"] = funds
            st.session_state.messages.append({"role": "user", "content": f"Profession: {prof}, Funds: {funds}"})
            st.session_state.messages.append({"role": "assistant", "content": "Thank you. Now please complete the mandatory pre-flight checks below."})
            st.session_state.step = "mandatory_checks"
            st.rerun()
        else:
            st.warning("Please fill out both fields.")

# Step 3: Mandatory Pre-flight Checks
elif st.session_state.step == "mandatory_checks":
    st.write("**Mandatory Prerequisites for Departure**")
    t1 = st.checkbox("Do you have a confirmed return ticket with the same PNR?")
    t2 = st.checkbox("Do you have a confirmed hotel booking for your entire staying period?")
    t3 = st.checkbox("Do you have at least 1000 USD 'show money' (or equivalent)?")
    
    if st.button("Submit Checks"):
        st.session_state.messages.append({"role": "user", "content": f"Return Ticket: {'Yes' if t1 else 'No'}\nHotel: {'Yes' if t2 else 'No'}\nShow Money: {'Yes' if t3 else 'No'}"})
        
        if t1 and t2 and t3:
            st.session_state.messages.append({"role": "assistant", "content": "All mandatory checks passed. Where are you traveling to?"})
            st.session_state.step = "destination"
        else:
            st.session_state.messages.append({"role": "assistant", "content": "❌ **OFF-LOADING WARNING**: You do not meet the mandatory prerequisites for departure (Return Ticket, Hotel Booking, and Show Money are strictly required). You cannot proceed."})
            st.session_state.step = "offload_warning"
            
        st.rerun()

# Step 4: Destination
elif st.session_state.step == "destination":
    dest = st.text_input("Destination Country (e.g. Cambodia, UAE)")
    if st.button("Next "):
        if dest:
            st.session_state.profile["destination"] = dest
            st.session_state.messages.append({"role": "user", "content": dest})
            st.session_state.messages.append({"role": "assistant", "content": "What type of visa do you hold?"})
            st.session_state.step = "visa_category"
            st.rerun()
        else:
            st.warning("Please enter a destination.")

# Step 5: Visa Category
elif st.session_state.step == "visa_category":
    cat = st.radio("Select Visa Category", ["WORK", "VISIT", "GOVERNMENT_SERVANT"], horizontal=True)
    if st.button("Next  "):
        st.session_state.profile["visa_category"] = cat
        st.session_state.messages.append({"role": "user", "content": cat})
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a scan of your Passport and Visa (Images or PDFs)."})
        st.session_state.step = "upload"
        st.rerun()

# Step 6: Upload Document & Trigger Stream
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
