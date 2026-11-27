"""AWS Lambda Function URL handler for Pakistan Immigration Manager (PIM) Travel Compliance Agent.

Features:
- Full CORS support for browser access from Hugging Face Spaces & custom domains
- Evaluates travel documents against official FIA departure guidelines & IBMS rules
- Multi-agent reasoning: Strict Compliance Inspector + Traveler Advocate & Critic
- Interactive FIA Immigration AI Assistant for passenger Q&A via Groq API
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Official FIA Departure Rules Reference
FIA_RULES_TEXT = """
FIA DEPARTURE GUIDELINES & EXIT REGULATIONS (Integrated Border Management System - IBMS)

1. WORK VISAS (Emigration Ordinance 1979):
- Any passenger traveling on a Work Visa / Employment Visa must present:
  * Valid Passport (> 6 months validity)
  * Valid Employment Visa / Work Permit for destination country
  * Valid Protector Stamp issued by the Protector of Emigrants (Bureau of Emigration & Overseas Employment)
  * Note: Protector Stamp is mandatory for Work Visas. It is NOT required for Tourist, Visit, or Student Visas.

2. TOURIST & VISIT VISAS:
- Must possess:
  * Valid Passport (> 6 months validity)
  * Valid Visit/Tourist Visa or eVisa
  * Confirmed Return Air Ticket (preferably on confirmed PNR)
  * Confirmed Hotel Booking or verified accommodation / invitation letter with host details
  * Sufficient Financial Proof: Minimum equivalent of 2,000 EUR (or valid credit card with international limit)
  * Clear travel purpose demonstrating intent to return to Pakistan.

3. HIGH-SCRUTINY WATCHLIST DESTINATIONS (15 Monitored Countries):
- Travelers to any of the following 15 monitored countries require heightened risk scrutiny:
  Saudi Arabia, Iran, Iraq, Türkiye, Qatar, Azerbaijan, Kuwait, Kyrgyzstan, Russia, Egypt, Libya, Ethiopia, Senegal, Mauritania, and Kenya.
- For these destinations, airport immigration officers verify bank statements, local ties, and verified return tickets.

4. CAMBODIA SPECIFIC REQUIREMENTS:
- Strict FIA departure directives for Cambodia:
  * Confirmed return air ticket on the exact same PNR as outbound flight
  * Confirmed advance-paid hotel booking
  * Minimum 1,000 USD in cash (show money)
  * Authentic visa documentation and employment/profession proof in Pakistan.

5. GOVERNMENT SERVANTS:
- Must produce an original No Objection Certificate (NOC) from their parent department / ministry.
"""

FIA_COMPLIANCE_SYSTEM_PROMPT = f"""
You are an expert AI Immigration Compliance Officer specialized in Pakistan's Federal Investigation Agency (FIA) departure rules and IBMS guidelines.

Your objective is to evaluate a traveler's uploaded profile and documents against strict official FIA requirements to prevent wrongful offloading or catch discrepancies before heading to the airport.

{FIA_RULES_TEXT}

EVALUATION PROTOCOL:
- Status MUST be either "CLEARED" or "OFF-LOAD".
- If the passenger satisfies the requirements for their visa category and destination, status must be "CLEARED".
- Protector Stamp is ONLY required for WORK visas. Do NOT flag protector stamp for Visit, Tourist, Student, or Business visas.
- Return ticket and hotel booking are mandatory for Visit/Tourist visas.
- Give a numerical compliance score (e.g. "90%").
- List verified items that satisfy FIA regulations.
- List any missing or incomplete requirements clearly with practical advice.
- Provide official FIA rule references citing relevant sections.

OUTPUT FORMAT (JSON object only):
{{
  "status": "CLEARED" | "OFF-LOAD",
  "compliance_score": "percentage rating",
  "verified_items": ["item 1", "item 2", ...],
  "missing_or_incomplete_requirements": ["missing item 1", ...],
  "fia_rule_reference": "explanation citing FIA guidelines and legal provisions"
}}
"""

ADVOCATE_SYSTEM_PROMPT = """
You are the Traveler Advocate and Critic Agent. Your role is to protect genuine travelers from wrongful offloading by searching for mitigating socio-economic factors (stable profession, business ownership, established savings, return ticket) that counteract automated strict risk flags.

If the passenger is a genuine traveler with sufficient intent to return, provide an advocacy argument and set strong_case to true.
If the passenger has a fatal statutory violation (e.g. missing Protector Stamp on a Work Visa, or traveling to Cambodia without return ticket/cash), set strong_case to false.

OUTPUT FORMAT (JSON object only):
{
  "mitigating_factors": "detailed advocacy argument",
  "strong_case": true | false
}
"""


def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        "Content-Type": "application/json",
    }


def _call_groq(messages: List[Dict[str, str]], json_mode: bool = False, temperature: float = 0.2) -> str:
    """Call Groq API using standard library urllib."""
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured on AWS Lambda.")

    model = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]


def handle_check_compliance(body: Dict[str, Any]) -> Dict[str, Any]:
    nationality = body.get("nationality", "Pakistani")
    destination = body.get("destination", "Unknown")
    visa_cat = (body.get("visa_category") or "VISIT").upper()
    purpose = body.get("purpose", "Tourism / Visit")
    passport_history = body.get("passport_history", "Experienced")
    profession = body.get("profession", "Private Employee")
    bank_funds = body.get("bank_funds", "N/A")
    has_return = bool(body.get("has_return_ticket", False))
    has_hotel = bool(body.get("has_hotel_booking", False))
    has_funds = bool(body.get("has_financial_proof", False))
    has_protector = bool(body.get("has_protector_stamp", False))

    user_prompt = f"""
Passenger Profile:
- Nationality: {nationality}
- Destination: {destination}
- Visa Category: {visa_cat}
- Purpose of Travel: {purpose}
- Passport Travel History: {passport_history}
- Declared Profession: {profession}
- Bank Funds / Financial Proof: {bank_funds}
- Return Ticket Present: {has_return}
- Hotel Booking / Accommodation Proof: {has_hotel}
- Sufficient Funds Proof (Bank Statement / Credit Card): {has_funds}
- Protector Stamp Present (if Work Visa): {has_protector}
"""

    # Step 1: Strict Inspection Node
    strict_messages = [
        {"role": "system", "content": FIA_COMPLIANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_eval = _call_groq(strict_messages, json_mode=True, temperature=0.1)
        evaluation = json.loads(raw_eval)
    except Exception as exc:
        return {
            "status": "ERROR",
            "compliance_score": "0%",
            "verified_items": [],
            "missing_or_incomplete_requirements": [f"API Error evaluating compliance: {str(exc)}"],
            "fia_rule_reference": "System error contacting inference engine.",
            "advocate_notes": "Unable to invoke debate agent due to upstream error.",
        }

    # Step 2: Traveler Advocate & Critic Node
    advocate_notes = "Traveler fully cleared by immigration officer."
    if evaluation.get("status") != "CLEARED":
        advocate_messages = [
            {"role": "system", "content": ADVOCATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
{user_prompt}

Strict Model Findings:
- Status: {evaluation.get('status')}
- Missing Requirements: {json.dumps(evaluation.get('missing_or_incomplete_requirements', []))}
- Reference: {evaluation.get('fia_rule_reference', '')}

Evaluate whether mitigating socio-economic factors warrant clearance or if statutory violations prevent travel.
""",
            },
        ]
        try:
            raw_adv = _call_groq(advocate_messages, json_mode=True, temperature=0.5)
            adv_res = json.loads(raw_adv)
            advocate_notes = adv_res.get("mitigating_factors", "Advocate reviewed case.")
            if adv_res.get("strong_case", False):
                # Advocate successfully mitigates non-statutory flags
                evaluation["status"] = "CLEARED"
                evaluation["verified_items"].append("Cleared via Advocate review: socio-economic ties verified.")
                evaluation["missing_or_incomplete_requirements"] = [
                    f"(ADVOCATE MITIGATED) {item}" for item in evaluation.get("missing_or_incomplete_requirements", [])
                ]
        except Exception as e:
            advocate_notes = f"Advocate review note: {str(e)}"

    evaluation["advocate_notes"] = advocate_notes
    return evaluation


def handle_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = body.get("messages", [])
    if not messages:
        return {"reply": "Hello! I am your PIM FIA Travel Compliance AI assistant. How can I help you today?"}

    chat_history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": f"You are an authoritative, helpful FIA Immigration Compliance Assistant for Pakistani travelers.\nAnswer accurately based on official departure guidelines:\n{FIA_RULES_TEXT}\nBe polite, concise, and highlight critical requirements like Protector Stamp, return tickets, and show money.",
        }
    ]

    for m in messages[-8:]:
        role = "user" if m.get("role") == "user" else "assistant"
        chat_history.append({"role": role, "content": str(m.get("content", ""))})

    try:
        reply = _call_groq(chat_history, json_mode=False, temperature=0.6)
        return {"reply": reply}
    except Exception as exc:
        return {"reply": f"Sorry, I encountered an error connecting to the compliance service: {str(exc)}"}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main AWS Lambda handler supporting Function URLs and API Gateway."""
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()

    # Pre-flight CORS request
    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": json.dumps({"status": "cors_ready"}),
        }

    raw_path = (
        event.get("rawPath")
        or event.get("path")
        or "/"
    ).lower()

    # Parse JSON body if present
    body: Dict[str, Any] = {}
    raw_body = event.get("body")
    if raw_body:
        try:
            body = json.loads(raw_body)
        except Exception:
            body = {}

    headers = _cors_headers()

    # Routing
    if raw_path in ("/health", "/api/health") or (raw_path == "/" and method == "GET"):
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({
                "status": "healthy",
                "service": "Pakistan Immigration Manager (PIM) - Travel Compliance Agent",
                "version": "1.1.0",
                "provider": "AWS Lambda Free Tier + Groq",
                "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            }),
        }

    if "check-compliance" in raw_path and method == "POST":
        result = handle_check_compliance(body)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(result),
        }

    if "chat" in raw_path and method == "POST":
        result = handle_chat(body)
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(result),
        }

    return {
        "statusCode": 404,
        "headers": headers,
        "body": json.dumps({"error": f"Path not found: {raw_path}"}),
    }
