<p align="center">
  <img src="assets/pim_logo.jpg" width="150" alt="PIM Logo">
</p>

#  Travel Document Compliance Checker Prototype (PIM)

<p align="center">
  <a href="https://pimpak.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App"></a>
  <a href="https://huggingface.co/spaces/MF180/travel-compliance-checker"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=flat" alt="Hugging Face Spaces"></a>
  <a href="https://uplfb4vsok54ao32lqrbagm56a0wtcdh.lambda-url.us-east-1.on.aws/health"><img src="https://img.shields.io/badge/AWS-Lambda%20Backend-FF9900?logo=amazon-aws&logoColor=white" alt="AWS Lambda"></a>
  <a href="https://groq.com"><img src="https://img.shields.io/badge/Inference-Groq%20LPU-f55036" alt="Groq LPU"></a>
</p>

An automated AI-driven prototype for evaluating passenger documents against FIA (Federal Investigation Agency) departure rules and Integrated Border Management System (IBMS) guidelines. 
The system features a multi-agent architecture combining strict immigration compliance verification with a traveler advocate/critic loop to catch discrepancies early and prevent wrongful offloading.

---

## 🌐 Live Demos & Deployments

| Component | Platform / Host | Description | Live Access Link |
| :--- | :--- | :--- | :--- |
| **Interactive Web UI** | **Hugging Face Spaces** | Multi-agent compliance screening UI & interactive FIA AI assistant (100% Free Tier) | [🚀 Launch Hugging Face Space](https://huggingface.co/spaces/MF180/travel-compliance-checker)<br>*(Direct App: [mf180-travel-compliance-checker.static.hf.space](https://mf180-travel-compliance-checker.static.hf.space))* |
| **Streamlit Dashboard** | **Streamlit Community Cloud** | Original Python Streamlit prototype with interactive controls and PDF generation | [⚡ Open Streamlit App](https://pimpak.streamlit.app/) |
| **Serverless Agent Backend** | **AWS Lambda** (us-east-1) | Always-free Lambda Function URL exposing `/api/check-compliance`, `/api/chat`, and `/health` | [🔗 Live API Endpoint](https://uplfb4vsok54ao32lqrbagm56a0wtcdh.lambda-url.us-east-1.on.aws/health) |

---

## Project Structure

- `backend/`: FastAPI application containing the LangGraph state machine.
- `frontend/`: Streamlit web application. 

## Prerequisites

- Python 3.9+
- A Groq API Key (if you want the LLM agent to evaluate compliance dynamically)

## Setup & Run Instructions

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:

```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Set your Groq API key in the environment (or create a `.env` file in the `backend` directory):
```bash
# Optional but recommended for full AI evaluation
set GROQ_API_KEY=your_sk_key_here
```

Run the backend server:
```bash
uvicorn main:app --reload --port 8000
```
The FastAPI backend will now be available at `http://localhost:8000`. You can view the swagger UI at `http://localhost:8000/docs`.

### 2. Frontend Setup

Open a **new** terminal and navigate to the `frontend` directory:

```bash
cd frontend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Run the Streamlit application:
```bash
streamlit run app.py
```

The Streamlit interface will open in your default browser at `http://localhost:8501`.

## How It Works

1. **Document Extraction Agent**: (Mocked) Simulates extracting data from uploaded documents.
2. **Rule Retrieval Agent (RAG)**: (Mocked) Simulates fetching relevant local rules using LlamaIndex.
3. **Compliance Verification Agent**: Uses OpenAI (via Langchain) configured with the strict FIA compliance system prompt to score the extracted features.
4. **Audit Feedback Agent**: Parses the results and delivers a definitive GREENLIGHT or ACTION_REQUIRED response with specific itemized checklists.
5. **PDF Report Generator**: Generates and downloads an itemized passenger compliance report.

---
*Note: This project is a prototype designed for demonstration purposes.*

<!-- daily check-in: 2026-10-29 -->
