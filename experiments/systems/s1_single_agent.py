"""
S1 (Single-Agent Baseline) and S1b (Single-Agent + Mitigating Context) implementations.
Configured with max_tokens=4096, reasoning_effort="medium", and missing-value failure handling.
"""
import os
from typing import Dict, Any
from experiments.systems.base import LLMClient
from experiments.data.profile_builder import build_agent_profile, serialize_profile_for_prompt

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

COMMON_RULES = load_prompt("fia_rules_common.txt")
S1_TEMPLATE = load_prompt("s1_single_agent.txt")
S1B_TEMPLATE = load_prompt("s1b_single_agent_mitigating.txt")

class SingleAgentSystem:
    def __init__(self, system_id: str = "S1", llm_client: LLMClient = None):
        self.system_id = system_id
        self.llm_client = llm_client or LLMClient()
        if system_id == "S1":
            self.system_prompt = S1_TEMPLATE.format(common_rules=COMMON_RULES)
        elif system_id == "S1b":
            self.system_prompt = S1B_TEMPLATE.format(common_rules=COMMON_RULES)
        else:
            raise ValueError(f"Unknown system_id for SingleAgent: {system_id}")

    def evaluate_profile(self, profile: Dict[str, Any], include_demographics: bool = False) -> Dict[str, Any]:
        agent_profile = build_agent_profile(profile, include_demographics=include_demographics)
        serialized_profile = serialize_profile_for_prompt(agent_profile)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"PASSENGER PROFILE FOR DEPARTURE EVALUATION:\n{serialized_profile}\n\nEvaluate compliance and return your assessment strictly according to the specified JSON schema."}
        ]

        res = self.llm_client.invoke_with_retry(
            messages=messages,
            temperature=0.0,
            reasoning_effort="medium",
            max_retries=2,
            max_tokens=4096
        )

        parsed = res["parsed_json"]
        if res["parse_status"] == "EXHAUSTED" or parsed.get("decision") is None:
            decision = None
            risk_score = None
            confidence = None
            rationale = str(parsed.get("rationale", ""))
            is_parse_failure = True
        else:
            dec_candidate = parsed.get("decision")
            if dec_candidate in ["CLEARED", "OFF-LOAD"]:
                decision = dec_candidate
                try:
                    risk_score = float(parsed.get("risk_score", 50.0))
                    risk_score = max(0.0, min(100.0, risk_score))
                except (ValueError, TypeError):
                    risk_score = 50.0
                try:
                    confidence = float(parsed.get("confidence", 0.5))
                except (ValueError, TypeError):
                    confidence = 0.5
                is_parse_failure = False
            else:
                decision = None
                risk_score = None
                confidence = None
                is_parse_failure = True
            rationale = str(parsed.get("rationale", ""))

        return {
            "system_id": self.system_id,
            "profile_id": profile.get("profile_id", "UNKNOWN"),
            "decision": decision,
            "risk_score": risk_score,
            "confidence": confidence,
            "rationale": rationale,
            "is_parse_failure": is_parse_failure,
            "parse_status": res["parse_status"],
            "retries": res["retries"],
            "tokens": res["tokens"],
            "cost_usd": res["cost_usd"],
            "latency_ms": res["latency_ms"],
            "calls_count": 1,
            "raw_response": res["raw_text"]
        }
