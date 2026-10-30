"""
Base LLM execution wrapper for Groq with retry logic, token usage tracking,
thread-safe expenditure tracking, TPM rate-limiting, and missing-value error handling.
"""
import os
import json
import time
import re
import threading
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

# Pricing for openai/gpt-oss-120b on Groq
INPUT_PRICE_PER_MILLION = 0.15   # $0.15 per 1M input tokens
OUTPUT_PRICE_PER_MILLION = 0.75  # $0.75 per 1M output tokens

class SpendCapExceededError(Exception):
    """Raised when total run expenditure exceeds the hard spend cap."""
    pass

class DailyTokenQuotaExceededError(Exception):
    """Raised when Groq daily token quota (TPD) is reached."""
    pass

class LLMClient:
    def __init__(self, spend_cap_usd: float = 2.00, model: str = "openai/gpt-oss-120b", max_tpm: int = 7000):
        self.model = model
        self.spend_cap_usd = spend_cap_usd
        self.max_tpm = max_tpm
        self.total_spend_usd = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_calls = 0
        self.lock = threading.Lock()
        self.token_history: List[Tuple[float, int]] = []
        self.client = Groq(timeout=45.0)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_in = (prompt_tokens / 1_000_000.0) * INPUT_PRICE_PER_MILLION
        cost_out = (completion_tokens / 1_000_000.0) * OUTPUT_PRICE_PER_MILLION
        return cost_in + cost_out

    def _wait_for_token_bucket(self, estimated_tokens: int = 1500):
        """Thread-safe sliding window rate limiter ensuring calls stay under max_tpm."""
        while True:
            with self.lock:
                now = time.time()
                self.token_history = [(ts, tok) for ts, tok in self.token_history if now - ts < 60.0]
                current_tpm = sum(tok for ts, tok in self.token_history)
                if current_tpm + estimated_tokens <= self.max_tpm or not self.token_history:
                    return
                oldest_ts = self.token_history[0][0]
                sleep_needed = max(1.0, 60.0 - (now - oldest_ts) + 0.5)
            time.sleep(min(sleep_needed, 5.0))

    def invoke_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        max_retries: int = 2,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Invokes Groq API with up to max_retries on network or JSON parsing failure.
        Enforces spend cap. On exhaustion, records missing value (None) instead of OFF-LOAD.
        """
        with self.lock:
            if self.total_spend_usd >= self.spend_cap_usd:
                raise SpendCapExceededError(
                    f"Hard spend cap of ${self.spend_cap_usd:.2f} reached. Total spend: ${self.total_spend_usd:.4f}"
                )

        attempt = 0
        rate_limit_retries = 0
        last_error = ""
        raw_text = ""

        while attempt <= max_retries:
            self._wait_for_token_bucket(estimated_tokens=1500)
            t0 = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    timeout=45.0
                )
                latency_ms = (time.time() - t0) * 1000.0
                
                usage = response.usage
                p_tok = usage.prompt_tokens or 0
                c_tok = usage.completion_tokens or 0
                r_tok = 0
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    r_tok = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                
                cost = self.calculate_cost(p_tok, c_tok)
                
                with self.lock:
                    self.total_calls += 1
                    self.total_spend_usd += cost
                    self.total_prompt_tokens += p_tok
                    self.total_completion_tokens += c_tok
                    self.total_reasoning_tokens += r_tok
                    self.token_history.append((time.time(), p_tok + c_tok))

                raw_text = response.choices[0].message.content or ""
                parsed_json = json.loads(raw_text)

                status_label = "SUCCESS" if attempt == 0 else f"RETRY_{attempt}"
                return {
                    "parsed_json": parsed_json,
                    "raw_text": raw_text,
                    "parse_status": status_label,
                    "retries": attempt,
                    "tokens": {
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "reasoning_tokens": r_tok,
                        "total_tokens": p_tok + c_tok
                    },
                    "cost_usd": cost,
                    "latency_ms": latency_ms,
                    "error": None
                }

            except RateLimitError as rle:
                err_msg = str(rle)
                if "tokens per day (TPD)" in err_msg:
                    raise DailyTokenQuotaExceededError(err_msg)
                
                rate_limit_retries += 1
                if rate_limit_retries > 10:
                    raise rle
                
                wait_sec = 6.0
                m = re.search(r"try again in (\d+(\.\d+)?)s", err_msg)
                if m:
                    wait_sec = float(m.group(1)) + 1.0
                print(f"[RateLimit TPM] Waiting {wait_sec:.1f}s before retry...", flush=True)
                time.sleep(wait_sec)
                continue

            except json.JSONDecodeError as e:
                last_error = f"JSONDecodeError: {str(e)}"
                attempt += 1
                time.sleep(1.0 * attempt)
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                attempt += 1
                time.sleep(2.0 * attempt)

        # Exhausted retries -> record as missing value (None), NOT OFF-LOAD
        return {
            "parsed_json": {
                "decision": None,
                "risk_score": None,
                "confidence": None,
                "rationale": f"JSON parse/execution failure after {max_retries} retries: {last_error}"
            },
            "raw_text": raw_text,
            "parse_status": "EXHAUSTED",
            "retries": max_retries,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "error": last_error
        }
