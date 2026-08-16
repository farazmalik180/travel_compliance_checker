# Travel Document Compliance Checker Prototype

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/farazmalik180/travel_compliance_checker/master/frontend/app.py)

An automated AI-driven prototype for evaluating passenger documents against FIA (Federal Investigation Agency) departure rules. 
The system uses **FastAPI** for the backend, **LangGraph** for multi-agent workflow orchestration, and **Streamlit** for a minimal UI.

## Project Structure

- `backend/`: FastAPI application containing the LangGraph state machine.
- `frontend/`: Streamlit web application.

## Prerequisites

- Python 3.9+
- An Groq API Key (if you want the LLM agent to evaluate compliance dynamically)

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
