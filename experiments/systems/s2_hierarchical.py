"""
S2: Hierarchical Consensus Pipeline implementation.
Configured with max_tokens=4096, reasoning_effort="medium", and missing-value failure handling.
"""
import os
import json
from typing import Dict, Any
from experiments.systems.base import LLMClient
from experiments.data.profile_builder import build_agent_profile, serialize_profile_for_prompt

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

COMMON_RULES = load_prompt("fia_rules_common.txt")
DOC_TEMPLATE = load_prompt("s2_document_integrity.txt").format(common_rules=COMMON_RULES)
SOCIO_TEMPLATE = load_prompt("s2_socio_economic.txt").format(common_rules=COMMON_RULES)
SCORER_TEMPLATE = load_prompt("s2_risk_scorer.txt").format(common_rules=COMMON_RULES)

class HierarchicalSystem:
    def __init__(self, llm_client: LLMClient = None):
        self.system_id = "S2"
        self.llm_client = llm_client or LLMClient()

    def evaluate_profile(self, profile: Dict[str, Any], include_demographics: bool = False) -> Dict[str, Any]:
        agent_profile = build_agent_profile(profile, include_demographics=include_demographics)
        serialized_profile = serialize_profile_for_prompt(agent_profile)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost_usd = 0.0
        total_latency_ms = 0.0
        parse_statuses = []
        retries_list = []
        raw_traces = {}

        # 1. Document Integrity Assessment
        msg_doc = [
            {"role": "system", "content": DOC_TEMPLATE},
            {"role": "user", "content": f"PASSENGER PROFILE:\n{serialized_profile}\n\nPerform procedural and statutory document integrity audit. Return valid JSON."}
        ]
        res_doc = self.llm_client.invoke_with_retry(msg_doc, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
        total_prompt_tokens += res_doc["tokens"]["prompt_tokens"]
        total_completion_tokens += res_doc["tokens"]["completion_tokens"]
        total_reasoning_tokens += res_doc["tokens"]["reasoning_tokens"]
        total_cost_usd += res_doc["cost_usd"]
        total_latency_ms += res_doc["latency_ms"]
        parse_statuses.append(res_doc["parse_status"])
        retries_list.append(res_doc["retries"])
        raw_traces["document_integrity"] = res_doc["raw_text"]
        doc_eval = res_doc["parsed_json"]

        # 2. Socio-Economic Context Assessment
        msg_socio = [
            {"role": "system", "content": SOCIO_TEMPLATE},
            {"role": "user", "content": f"PASSENGER PROFILE:\n{serialized_profile}\n\nPerform livelihood and economic anchorage analysis. Return valid JSON."}
        ]
        res_socio = self.llm_client.invoke_with_retry(msg_socio, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
        total_prompt_tokens += res_socio["tokens"]["prompt_tokens"]
        total_completion_tokens += res_socio["tokens"]["completion_tokens"]
        total_reasoning_tokens += res_socio["tokens"]["reasoning_tokens"]
        total_cost_usd += res_socio["cost_usd"]
        total_latency_ms += res_socio["latency_ms"]
        parse_statuses.append(res_socio["parse_status"])
        retries_list.append(res_socio["retries"])
        raw_traces["socio_economic"] = res_socio["raw_text"]
        socio_eval = res_socio["parsed_json"]

        # 3. Consensus Scorer & Adjudicator
        msg_scorer = [
            {"role": "system", "content": SCORER_TEMPLATE},
            {"role": "user", "content": f"PASSENGER PROFILE:\n{serialized_profile}\n\n"
                                        f"SPECIALIST FINDINGS:\n"
                                        f"1. Document Integrity Audit: {json.dumps(doc_eval)}\n"
                                        f"2. Socio-Economic Context Audit: {json.dumps(socio_eval)}\n\n"
                                        f"Synthesize final clearance decision and continuous risk score strictly in the specified JSON format."}
        ]
        res_scorer = self.llm_client.invoke_with_retry(msg_scorer, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
        total_prompt_tokens += res_scorer["tokens"]["prompt_tokens"]
        total_completion_tokens += res_scorer["tokens"]["completion_tokens"]
        total_reasoning_tokens += res_scorer["tokens"]["reasoning_tokens"]
        total_cost_usd += res_scorer["cost_usd"]
        total_latency_ms += res_scorer["latency_ms"]
        parse_statuses.append(res_scorer["parse_status"])
        retries_list.append(res_scorer["retries"])
        raw_traces["consensus_scorer"] = res_scorer["raw_text"]
        final_eval = res_scorer["parsed_json"]

        has_exhaustion = any(s == "EXHAUSTED" for s in parse_statuses)
        dec_candidate = final_eval.get("decision")

        if has_exhaustion or dec_candidate not in ["CLEARED", "OFF-LOAD"]:
            decision = None
            risk_score = None
            confidence = None
            is_parse_failure = True
            rationale = str(final_eval.get("rationale", "Hierarchical consensus pipeline failure."))
        else:
            decision = dec_candidate
            try:
                risk_score = float(final_eval.get("risk_score", 50.0))
                risk_score = max(0.0, min(100.0, risk_score))
            except (ValueError, TypeError):
                risk_score = 50.0
            try:
                confidence = float(final_eval.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            is_parse_failure = False
            rationale = str(final_eval.get("rationale", ""))

        overall_parse_status = "SUCCESS" if all(s == "SUCCESS" for s in parse_statuses) else (
            "EXHAUSTED" if any(s == "EXHAUSTED" for s in parse_statuses) else "RETRY"
        )

        return {
            "system_id": self.system_id,
            "profile_id": profile.get("profile_id", "UNKNOWN"),
            "decision": decision,
            "risk_score": risk_score,
            "confidence": confidence,
            "rationale": rationale,
            "is_parse_failure": is_parse_failure,
            "parse_status": overall_parse_status,
            "retries": sum(retries_list),
            "tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            },
            "cost_usd": total_cost_usd,
            "latency_ms": total_latency_ms,
            "calls_count": 3,
            "specialist_outputs": {
                "document_integrity": doc_eval,
                "socio_economic": socio_eval
            },
            "raw_response": json.dumps(raw_traces)
        }
