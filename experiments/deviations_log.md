# Experiments Deviations Log

This document records all methodological, configuration, and execution deviations from initial pre-registration specifications, including root causes, impact assessments, and corrective actions.

---

### Deviation 1: Dev Pilot Run 1 JSON Truncation and Fallback Inflation
* **Date:** 2026-10-29
* **Affected Artifacts:** `experiments/results/pilot_raw_checkpoint_run1_truncated.jsonl`, `run_manifest_run1_truncated.json`, `pilot_sampled_traces_run1_truncated.json`
* **Root Cause:**
  The LLM runner was configured with `max_tokens = 512`. Because `openai/gpt-oss-120b` generates hidden reasoning tokens (averaging 160–580 tokens) before outputting completion text, generation was truncated mid-sentence before closing the JSON string and braces. Groq's edge validator returned `json_validate_failed` (HTTP 400). Per the initial fallback rule, all unparseable responses defaulted to `"OFF-LOAD"` with `confidence = 0.0` and `risk_score = 100.0`. This artificially inflated the offload rates (88–95%) and caused high parse-failure counts across all systems.
* **Status of Run 1 Data:**
  Archived with suffix `_run1_truncated`. Performance numbers from Run 1 are strictly quarantined and invalidated.
* **Corrective Actions Implemented for Run 2:**
  1. `max_tokens` set to `4096` across all systems to provide ample headroom for full reasoning traces and comprehensive rationales.
  2. Fixed `reasoning_effort = "medium"` across all systems.
  3. Modified error handling protocol: A call that still fails to parse after retries is recorded as a missing value (`decision = None` / `"PARSE_FAILURE"`, `risk_score = None`, `is_parse_failure = True`) instead of defaulting to `"OFF-LOAD"`. Metrics will be reported both with failed profiles excluded and with failures counted as `"OFF-LOAD"`.
  4. Added concurrency to the runner respecting Groq paid-tier limits (1,000 RPM, 250k TPM).

