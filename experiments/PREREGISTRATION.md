# Formal Preregistration: Multi-Agent Travel Compliance & Fairness Benchmarking

**Author:** MS AI Thesis Research  
**Date of Preregistration:** October 2026  
**Version:** v2.0 (Git Tag: `prereg-v2`)  
**Status:** Preregistered prior to execution of any LLM calls on test, fairness, or counterfactual splits.

---

## 1. Study Overview & Objective
This study investigates whether multi-agent LLM architectures (Hierarchical Consensus and Peer-to-Peer Adversarial Debate) reduce wrongful offloading (False Positive Rate) and mitigate algorithmic bias against non-traditional travelers (gig workers, freelancers, informal earners) compared to single-agent baselines and token-matched self-consistency ensembles, within the domain of immigration departure screening.

---

## 2. Evaluated Systems & Experimental Controls

All systems operate under identical inference constraints using the same model (`openai/gpt-oss-120b`) on Groq with fixed reasoning effort (`"medium"`). All prompts are version-controlled in `experiments/prompts/` and frozen under git tag `prompts-frozen-v1`.

| System ID | Name | Architecture | Prompt / Interaction Mode | Temperature | Expected Calls / Profile | Role in Study |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | **Deterministic Rule Engine** | Pure Code Engine (No LLM) | Applies statutory rules from `fia_rules_common.txt` | N/A | 0 calls | Procedural Lower-Bound Reference |
| **S1** | **Single-Agent Baseline** | Monolithic LLM call | Strict departure compliance prompt | $T = 0.0$ | 1 call | Reference Baseline Only |
| **S1b** | **Single-Agent + Mitigating Context** | Monolithic LLM call | S1 prompt + mitigating-context instruction block | $T = 0.0$ | 1 call | Primary Comparison (Prompt Control) |
| **S1c** | **Token-Matched Self-Consistency** | Ensemble Voting | $k$ repeated stochastic calls of S1; majority vote decision; mean risk score. $k$ calibrated from S3 measured mean tokens | $T = 0.7$ | $k$ calls (approx. 3–5) | Primary Comparison (Compute Control) |
| **S2** | **Hierarchical Consensus** | LangGraph Pipeline | Sequential routing: Document Integrity $\to$ Socio-Economic Context $\to$ Risk Scorer $\to$ Final Consensus Aggregator | $T = 0.0$ | 3 calls | Primary Comparison |
| **S3** | **Peer-to-Peer Adversarial Debate** | LangGraph StateGraph | Multi-round debate: Risk Scorer $\leftrightarrow$ Traveler Advocate (up to 3 rounds) $\to$ Final Adjudicator | $T = 0.0$ | 3 to 7 calls (mean ~4.5) | Proposed Primary System |

---

## 3. Primary Hypotheses & Primary Comparison Family

The confirmatory statistical testing focuses on the **Primary Comparison Family** of 4 directional hypotheses. Vanilla single-agent baseline (S1) and deterministic rule engine (S0) serve as reference baselines.

To prevent multiplicity inflation while maintaining decisive statistical testing, **each hypothesis specifies exactly one pre-registered primary metric and direction**. All other metrics are formally secondary.

1. **$H_{1a}$ (Debate vs. Prompt Engineering: S3 vs. S1b):**  
   * **Primary Metric:** False Positive Rate on held-out test data at matched target Recall = 0.70 ($\text{FPR}_{R=0.70}$).  
   * **Direction:** $\text{FPR}_{R=0.70}(S_3) < \text{FPR}_{R=0.70}(S_{1b})$ (One-sided, $\alpha = 0.05$).  
   * **Secondary Metrics:** ROC-AUC, $\Delta \text{FPR}_{\text{NT}}$ at $R=0.70$, $\text{FPR}_{R=0.50}$, Discretionary FPR.

2. **$H_{1b}$ (Debate vs. Compute / Ensemble: S3 vs. S1c):**  
   * **Primary Metric:** False Positive Rate on held-out test data at matched target Recall = 0.70 ($\text{FPR}_{R=0.70}$).  
   * **Direction:** $\text{FPR}_{R=0.70}(S_3) < \text{FPR}_{R=0.70}(S_{1c})$ (One-sided, $\alpha = 0.05$).  
   * **Secondary Metrics:** ROC-AUC, $\Delta \text{FPR}_{\text{NT}}$ at $R=0.70$, $\text{FPR}_{R=0.50}$, Discretionary FPR.

3. **$H_{1c}$ (Debate vs. Hierarchical Pipeline: S3 vs. S2):**  
   * **Primary Metric:** False Positive Rate on held-out test data at matched target Recall = 0.70 ($\text{FPR}_{R=0.70}$).  
   * **Direction:** $\text{FPR}_{R=0.70}(S_3) < \text{FPR}_{R=0.70}(S_2)$ (One-sided, $\alpha = 0.05$).  
   * **Secondary Metrics:** Non-traditional FPR disparity ($\Delta \text{FPR}_{\text{NT}}$ at $R=0.70$), ROC-AUC, Discretionary FPR.

4. **$H_{1d}$ (Hierarchical Pipeline vs. Single Agent: S2 vs. S1b):**  
   * **Primary Metric:** False Positive Rate on held-out test data at matched target Recall = 0.70 ($\text{FPR}_{R=0.70}$).  
   * **Direction:** $\text{FPR}_{R=0.70}(S_2) < \text{FPR}_{R=0.70}(S_{1b})$ (One-sided, $\alpha = 0.05$).  
   * **Secondary Metrics:** ROC-AUC, $\Delta \text{FPR}_{\text{NT}}$ at $R=0.70$, $\text{FPR}_{R=0.50}$.

---

## 4. Evaluation Cohorts, Metrics & Tie-Breaking Rules

### A. Two Mandatory Reporting Cohorts
To evaluate performance both across the general population and on ambiguous cases where discretionary judgment operates, all primary and secondary metrics must be reported on:
1. **Full Test Cohort ($N = 4,000$):** 3,200 Genuine, 800 Threats.
2. **Watchlist-Excluded Cohort ($N = 3,923$):** All test profiles excluding those with `watchlist_flag == 'Verified Watchlist Alert'` (77 profiles excluded).

### B. Primary Metric & Threshold-Free Curve Evaluation
* **ROC-AUC:** Computed over continuous `risk_score` $\in [0, 100]$ against binary ground truth $Y^* \in \{0, 1\}$.
* **Pre-specified Tie Rule for Clumped Scores:** When computing empirical ROC curves and threshold operating points, tied `risk_score` values are resolved using **standard deterministic average rank (scikit-learn midpoint ranking)** without stochastic jitter, ensuring exact bitwise reproducibility across runs.
* **Operating Point Selection:** Evaluated at matched target recall points ($R = 0.50$ and $R = 0.70$) directly from the test ROC curve.

### C. Procedural vs. Discretionary Bias Decomposition
For each system, every offloaded genuine traveler ($Y^* = 0, \hat{Y} = 1$) is decomposed into:
* **(a) Hard-Rule False Positive:** The traveler violated $\ge 1$ statutory requirement from `fia_rules_common.txt` (procedural failure).
* **(b) Discretionary False Positive:** The traveler satisfied all statutory requirements, but was offloaded based on subjective risk profiling (employment, ties, destination hub scrutiny).
* **Mandatory Sub-Metrics Reported:**
  1. **Discretionary FPR:**
     $$\text{FPR}_{\text{disc}} = \frac{\text{Offloaded Genuine Travelers with Zero Statutory Violations}}{\text{Total Genuine Travelers with Zero Statutory Violations}}$$
  2. **Hard-Rule Threat Leakage Rate:**
     $$\text{Clearance}_{\text{hard\_threat}} = \frac{\text{Cleared Threats with } \ge 1 \text{ Statutory Violation}}{\text{Total Threats with } \ge 1 \text{ Statutory Violation}}$$

### D. Counterfactual Demographic Flip Experiment
* **Systems Tested:** S1, S1b, S2, S3 (and S1c).
* **Attribute Exposure:** Demographics (`gender`, `region`, `nationality_group`) are **explicitly visible** in the prompt (unblinded).
* **Independent Attribute Flips:** Each attribute is perturbed independently while holding all other profile fields constant:
  1. **Gender:** Male $\leftrightarrow$ Female.
  2. **Region:** Origin province perturbed across major geographic administrative units.
  3. **Nationality Group:** Domestic Citizen $\leftrightarrow$ Dual / Naturalized Citizen.
* **Sample Size:** $N = 500$ paired comparisons per sensitive attribute (1,500 pairs total = 3,000 profile evaluations per system).
* **Metric:** Counterfactual Flip Rate:
  $$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\hat{Y}(P_i) \neq \hat{Y}(P_{i, \text{flip}})\right)$$

---

## 5. Statistical Inference & Multiple Testing Correction

* **Holm-Bonferroni Correction:** Applied strictly across the 4 primary hypothesis tests ($H_{1a}, H_{1b}, H_{1c}, H_{1d}$) on the primary metric ($\text{FPR}_{R=0.70}$) at FWER $\alpha = 0.05$.
* **Confidence Intervals:** 95% bootstrap confidence intervals computed over **1,000 bootstrap resamples** of test profiles.
* **Paired Comparison Tests:** Paired binary decisions at matched operating points tested via **McNemar's test** with continuity correction.
* **Permutation Noise Floors:**
  * Test Split ($N=4,000$): Noise floor for $\Delta \text{FPR}_{\text{NT}}$ is $\pm 1.84\%$ at $R=0.50$ and $\pm 2.75\%$ at $R=0.70$.
  * Disparities within this noise envelope cannot reject the null of demographic parity.

---

## 6. Hard Leakage Assertion (Bayes-Oracle Ceiling)

* **The True Latent Bayes Oracle ROC-AUC on the held-out test set is 0.8857.**
* **Hard Assertion Check:** If any system achieves a Test ROC-AUC statistically significantly above 0.886 ($p < 0.01$), execution halts immediately and the system is audited for data leakage.

---

## 7. Execution Logging & Token Tracking Design

Every LLM call logs:
1. `timestamp` (ISO 8601 UTC)
2. `system_id`, `profile_id`, `call_index`, `agent_role`
3. `prompt_tokens` (input tokens)
4. `completion_tokens` (output tokens)
5. `reasoning_tokens` (from API response metadata if available, else isolated)
6. `total_tokens` and incremental estimated cost (USD)
7. `latency_ms`
8. `json_parse_status` (`SUCCESS`, `RETRY_1`, `RETRY_2`, `EXHAUSTED`)
9. `raw_response_text`

### Per-Phase Spend Caps
* **Phase 1 (Dev Pilot, $N=200$):** Hard spend cap **$2.00**
* **Phase 2 (Full Test Set, $N=4,000$):** Hard spend cap **$12.00** (re-calibrated after pilot)
* **Phase 3 (Fairness Set, $N=1,500$):** Hard spend cap **$4.50**
* **Phase 4 (Counterfactual Flips, 1,500 pairs):** Hard spend cap **$5.00**

---

## 8. Error Handling Protocol

1. **Retries:** Maximum of 2 retries (total 3 attempts) per call upon network interruption or invalid JSON output.
2. **Exhaustion Fallback:** If still unparseable after 3 attempts, record `json_parse_error = True`, default decision to `"OFF-LOAD"` with `confidence = 0.0`, `risk_score = 100.0`, and log the trace in `results/deviations_log.md`.

---

## 9. Versioned Prompt Hashes (SHA-256)

Prompts are canonicalized to Unix LF line endings and frozen:
* `fia_rules_common.txt`: `25743637fb5dcba6e3be20672f01ae7e0ce91141564de13832a535efd0e693d0`
* `s1_single_agent.txt`: `a89a4185a0ae6f18622525621d248ccb53f5e2ee318ebdc4c51e6b66187636b2`
* `s1b_single_agent_mitigating.txt`: `e32132def426c3bff534bd9b49392fd456f0448bd6eb5a26ad2bedabb1282426`
* `s2_document_integrity.txt`: `9d09a7d53192e43f9c4a4efea8e9f397c646d9351f80a74bd608636c2def4403`
* `s2_socio_economic.txt`: `baf0cef7da7fd80cb02691eab81f7a61abfb2bc3a164105bb648b5c5428abf61`
* `s2_risk_scorer.txt`: `47f68f7b0030663b966f4ac56298c26ddb69996d8a50bce3c873b81f88d59b03`
* `s3_risk_scorer.txt`: `2be547aedc594c624eab4630a9c1e9e463aff447200198d207c02001374f7c03`
* `s3_traveler_advocate.txt`: `b16ceeedfc8ecc5341009a29c2654b4f21ab06191d4a6f9eee640033f2401de0`
* `s3_adjudicator.txt`: `650c358aaa6ff84d173712be6ab90a4c85f582adc57fd72f68fd542082e892a1`

---

## 10. Deviations Log
1. **2026-10-28:** Common rules header retitled to author-written illustrative rules. Statutory exemptions clause centralized in `fia_rules_common.txt` so all systems evaluate identical statutory scope. Prompt hashes updated.
2. **2026-10-28:** Preregistration v2 committed. Pre-specified one primary metric per hypothesis ($\text{FPR}_{R=0.70}$, one-sided). Pre-specified deterministic average-rank tie-breaking rule for clumped risk scores. Added watchlist-excluded reporting cohort ($N=3,923$). Added discretionary FPR and hard-rule threat leakage metrics.
