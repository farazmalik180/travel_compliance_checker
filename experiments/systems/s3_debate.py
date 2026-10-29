"""
S3: Peer-to-Peer Adversarial Debate Implementation.
Configured with max_tokens=4096, reasoning_effort="medium", and missing-value failure handling.
"""
import os
import json
from typing import Dict, Any, List
from experiments.systems.base import LLMClient
from experiments.data.profile_builder import build_agent_profile, serialize_profile_for_prompt

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

COMMON_RULES = load_prompt("fia_rules_common.txt")
PROSECUTOR_TEMPLATE = load_prompt("s3_risk_scorer.txt").format(common_rules=COMMON_RULES)
ADVOCATE_TEMPLATE = load_prompt("s3_traveler_advocate.txt").format(common_rules=COMMON_RULES)
ADJUDICATOR_TEMPLATE = load_prompt("s3_adjudicator.txt").format(common_rules=COMMON_RULES)

class AdversarialDebateSystem:
    def __init__(self, llm_client: LLMClient = None):
        self.system_id = "S3"
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
        transcript: List[Dict[str, Any]] = []

        rounds_completed = 0
        max_rounds = 3

        for round_idx in range(1, max_rounds + 1):
            rounds_completed = round_idx
            # 1. Prosecutor Turn
            if round_idx == 1:
                prosecutor_input = f"PASSENGER PROFILE:\n{serialized_profile}\n\nPresent opening prosecution allegations and flight risk assessment."
            else:
                transcript_summary = json.dumps([t for t in transcript if t["round"] < round_idx])
                last_defense = transcript[-1]["content"]
                prosecutor_input = (
                    f"PASSENGER PROFILE:\n{serialized_profile}\n\n"
                    f"PRIOR DEBATE HISTORY: {transcript_summary}\n\n"
                    f"LATEST DEFENSE ARGUMENT: {json.dumps(last_defense)}\n\n"
                    f"Rebut the Advocate's defense points directly and update your prosecution recommendation."
                )

            msg_pros = [
                {"role": "system", "content": PROSECUTOR_TEMPLATE},
                {"role": "user", "content": prosecutor_input}
            ]
            res_pros = self.llm_client.invoke_with_retry(msg_pros, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
            total_prompt_tokens += res_pros["tokens"]["prompt_tokens"]
            total_completion_tokens += res_pros["tokens"]["completion_tokens"]
            total_reasoning_tokens += res_pros["tokens"]["reasoning_tokens"]
            total_cost_usd += res_pros["cost_usd"]
            total_latency_ms += res_pros["latency_ms"]
            parse_statuses.append(res_pros["parse_status"])
            retries_list.append(res_pros["retries"])
            pros_data = res_pros["parsed_json"]

            transcript.append({
                "round": round_idx,
                "speaker": "PROSECUTOR",
                "content": pros_data
            })

            # 2. Advocate Turn
            if round_idx == 1:
                advocate_input = (
                    f"PASSENGER PROFILE:\n{serialized_profile}\n\n"
                    f"PROSECUTION ALLEGATIONS:\n{json.dumps(pros_data)}\n\n"
                    f"Present mitigating socio-economic context and counter-arguments defending departure clearance."
                )
            else:
                advocate_input = (
                    f"PASSENGER PROFILE:\n{serialized_profile}\n\n"
                    f"PROSECUTOR'S REBUTTAL:\n{json.dumps(pros_data)}\n\n"
                    f"Address the prosecutor's rebuttal and make your counter-plea."
                )

            msg_adv = [
                {"role": "system", "content": ADVOCATE_TEMPLATE},
                {"role": "user", "content": advocate_input}
            ]
            res_adv = self.llm_client.invoke_with_retry(msg_adv, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
            total_prompt_tokens += res_adv["tokens"]["prompt_tokens"]
            total_completion_tokens += res_adv["tokens"]["completion_tokens"]
            total_reasoning_tokens += res_adv["tokens"]["reasoning_tokens"]
            total_cost_usd += res_adv["cost_usd"]
            total_latency_ms += res_adv["latency_ms"]
            parse_statuses.append(res_adv["parse_status"])
            retries_list.append(res_adv["retries"])
            adv_data = res_adv["parsed_json"]

            transcript.append({
                "round": round_idx,
                "speaker": "ADVOCATE",
                "content": adv_data
            })

            # Convergence Check
            pros_action = pros_data.get("proposed_action", "OFF-LOAD")
            adv_plea = adv_data.get("defense_plea", "CLEAR_PASSENGER")

            if (pros_action == "CLEARED" and adv_plea == "CLEAR_PASSENGER") or (pros_action == "OFF-LOAD" and adv_plea == "CONCEDE_OFFLOAD"):
                break

        # 3. Final Adjudicator Turn
        adjudicator_input = (
            f"PASSENGER PROFILE:\n{serialized_profile}\n\n"
            f"FULL ADVERSARIAL DEBATE TRANSCRIPT ({rounds_completed} ROUNDS COMPLETED):\n"
            f"{json.dumps(transcript, indent=2)}\n\n"
            f"Weigh both sides and deliver the definitive departure decision and calibrated risk score."
        )

        msg_adj = [
            {"role": "system", "content": ADJUDICATOR_TEMPLATE},
            {"role": "user", "content": adjudicator_input}
        ]
        res_adj = self.llm_client.invoke_with_retry(msg_adj, temperature=0.0, reasoning_effort="medium", max_tokens=4096)
        total_prompt_tokens += res_adj["tokens"]["prompt_tokens"]
        total_completion_tokens += res_adj["tokens"]["completion_tokens"]
        total_reasoning_tokens += res_adj["tokens"]["reasoning_tokens"]
        total_cost_usd += res_adj["cost_usd"]
        total_latency_ms += res_adj["latency_ms"]
        parse_statuses.append(res_adj["parse_status"])
        retries_list.append(res_adj["retries"])
        final_verdict = res_adj["parsed_json"]

        has_exhaustion = any(s == "EXHAUSTED" for s in parse_statuses)
        dec_candidate = final_verdict.get("decision")

        if has_exhaustion or dec_candidate not in ["CLEARED", "OFF-LOAD"]:
            decision = None
            risk_score = None
            confidence = None
            is_parse_failure = True
            rationale = str(final_verdict.get("rationale", "Adversarial debate tribunal parse failure."))
        else:
            decision = dec_candidate
            try:
                risk_score = float(final_verdict.get("risk_score", 50.0))
                risk_score = max(0.0, min(100.0, risk_score))
            except (ValueError, TypeError):
                risk_score = 50.0
            try:
                confidence = float(final_verdict.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            is_parse_failure = False
            rationale = str(final_verdict.get("rationale", ""))

        overall_parse_status = "SUCCESS" if all(s == "SUCCESS" for s in parse_statuses) else (
            "EXHAUSTED" if any(s == "EXHAUSTED" for s in parse_statuses) else "RETRY"
        )

        calls_count = (rounds_completed * 2) + 1

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
            "rounds_completed": rounds_completed,
            "tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            },
            "cost_usd": total_cost_usd,
            "latency_ms": total_latency_ms,
            "calls_count": calls_count,
            "debate_transcript": transcript,
            "raw_response": json.dumps(final_verdict)
        }
