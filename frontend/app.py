import streamlit as st
import requests
import json
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = "http://127.0.0.1:8000/api/check-compliance"

st.set_page_config(page_title="FIA Departure Compliance", page_icon="🛂", layout="wide")

st.title("🛂 Automated FIA Travel Compliance Checker")
st.markdown("Ensure your documents are complete before heading to the airport.")

with st.sidebar:
    st.header("Traveler Profile")
    nationality = st.text_input("Nationality", value="Pakistani")
    destination = st.text_input("Destination Country", value="UAE")
    visa_category = st.selectbox("Visa Category", ["WORK", "VISIT", "GOVERNMENT_SERVANT"])
    purpose = st.text_input("Purpose of Travel", value="Tourism")
    
st.header("Document Upload")
uploaded_files = st.file_uploader("Upload Passport, Visa, and Supporting Documents", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])

if st.button("Check Compliance", type="primary"):
    with st.spinner("Analyzing documents and profiling passenger..."):
        # For the prototype, we simply pass dummy file names if any are uploaded.
        doc_names = [f.name for f in uploaded_files] if uploaded_files else ["dummy_document.pdf"]
        
        payload = {
            "nationality": nationality,
            "destination": destination,
            "visa_category": visa_category,
            "purpose": purpose,
            "documents": doc_names
        }
        
        try:
            response = requests.post(BACKEND_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                
                # Render results
                status = result.get("status")
                
                if status == "GREENLIGHT":
                    st.success("✅ GREENLIGHT: Passenger is cleared for travel.")
                elif status == "ACTION_REQUIRED":
                    st.error("🚨 ACTION REQUIRED: Passenger risks being offloaded.")
                else:
                    st.warning(f"⚠️ STATUS UNKNOWN: {status}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Verified Items")
                    for item in result.get("verified_items", []):
                        st.markdown(f"- ✅ {item}")
                        
                with col2:
                    st.subheader("Missing / Incomplete")
                    for item in result.get("missing_or_incomplete_requirements", []):
                        st.markdown(f"- ❌ {item}")
                        
                st.info(f"**Compliance Score:** {result.get('compliance_score')}")
                st.caption(f"**FIA Rule Reference:** {result.get('fia_rule_reference')}")
                
            else:
                st.error(f"Backend error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
