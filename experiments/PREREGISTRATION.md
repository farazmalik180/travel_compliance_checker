# Formal Preregistration: Multi-Agent Travel Compliance & Fairness Benchmarking

**Author:** MS AI Thesis Research  
**Date of Preregistration:** October 2026  
**Status:** Preregistered prior to execution of any LLM calls on evaluation splits.

---

## 1. Study Overview & Objective
This study investigates whether multi-agent LLM architectures (Hierarchical Consensus and Peer-to-Peer Adversarial Debate) reduce wrongful offloading (False Positive Rate) and mitigate algorithmic bias against non-traditional travelers (gig workers, freelancers, informal earners) compared to single-agent baselines and token-matched self-consistency ensembles, within the domain of immigration departure screening.

---

## 2. Evaluated Systems & Experimental Controls

All systems operate under identical inference constraints using the same model (`openai/gpt-oss-120b`) on Groq with fixed reasoning effort (`"medium"`). All prompts are version-controlled in `experiments/prompts/` and frozen prior to test execution.

| System ID | Name | Architecture | Prompt / Interaction Mode | Temperature | Expected Calls / Profile | Role in Study |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | **Single-Agent Baseline** | Monolithic LLM call | Strict departure compliance prompt | $T = 0.0$ | 1 call | Reference Baseline Only |
| **S1b** | **Single-Agent + Mitigating Context** | Monolithic LLM call | S1 prompt + mitigating-context instruction block | $T = 0.0$ | 1 call | Primary Comparison (Prompt Control) |
| **S1c** | **Token-Matched Self-Consistency** | Ensemble Voting | $k$ repeated stochastic calls of S1; majority vote decision; mean risk score. $k$ calibrated from S3 measured mean tokens | $T = 0.7$ | $k$ calls (approx. 3–5) | Primary Comparison (Compute Control) |
| **S2** | **Hierarchical Consensus** | LangGraph Pipeline | Sequential routing: Document Integrity $\to$ Socio-Economic Context $\to$ Risk Scorer $\to$ Final Consensus Aggregator | $T = 0.0$ | 3 calls | Primary Comparison |
| **S3** | **Peer-to-Peer Adversarial Debate** | LangGraph StateGraph | Multi-round debate: Risk Scorer $\leftrightarrow$ Traveler Advocate (up to 3 rounds) $\to$ Final Adjudicator | $T = 0.0$ | 3 to 7 calls (mean ~4.5) | Proposed Primary System |

---

## 3. Primary Hypotheses & Primary Comparison Family

The confirmatory statistical testing focuses on the **Primary Comparison Family** of 4 directional hypotheses. The vanilla single-agent baseline (S1) serves as an uncontrolled reference baseline.

1. **$H_{1a}$ (Debate vs. Prompt Engineering: S3 vs. S1b):**  
   Adversarial Debate (S3) achieves lower False Positive Rate (FPR) on held-out test data at matched recall ($R = 0.50$ and $R = 0.70$) and lower non-traditional FPR disparity ($\Delta \text{FPR}_{\text{NT}}$) than the single-agent prompt-control (S1b).
2. **$H_{1b}$ (Debate vs. Compute / Ensemble: S3 vs. S1c):**  
   Adversarial Debate (S3) achieves higher ROC-AUC and lower FPR at matched recall than Token-Matched Self-Consistency (S1c), proving that conversational role-specialization yields qualitative reasoning advantages beyond simple stochastic sample voting.
3. **$H_{1c}$ (Debate vs. Hierarchical Pipeline: S3 vs. S2):**  
   Dialectic debate between adversarial peers (S3) achieves a lower False Positive Rate on ambiguous non-traditional profiles than a sequential hierarchical routing pipeline (S2).
4. **$H_{1d}$ (Hierarchical Pipeline vs. Single Agent: S2 vs. S1b):**  
   Hierarchical decomposition (S2) achieves superior ROC-AUC and lower FPR disparity than the single-agent prompt-control (S1b).

---

## 4. Primary Metrics & Threshold Design

### A. Primary Evaluation (Threshold-Free & Matched Recall)
1. **Test ROC-AUC:** Computed over continuous `risk_score` $\in [0, 100]$ against binary ground truth $Y^* \in \{0, 1\}$.
2. **Test FPR at Matched Recall (ROC Curve Points):**  
   Evaluated directly from the Test empirical ROC curve at Recall = 0.50 and Recall = 0.70. Reported with 1,000 bootstrap 95% confidence intervals.
3. **Non-Traditional FPR Disparity ($\Delta \text{FPR}_{\text{NT}}$):**
   $$\Delta \text{FPR}_{\text{NT}} = \Pr(\hat{Y}=1 \mid \text{NT}=1, Y^*=0) - \Pr(\hat{Y}=1 \mid \text{NT}=0, Y^*=0)$$

### B. Secondary Evaluation (Native Categorical Decisions)
* Native Accuracy, Recall, Overall FPR, and $\Delta \text{FPR}_{\text{NT}}$ evaluated on each system's discrete output `decision` (`"CLEARED"` vs `"OFF-LOAD"`).

### C. Demographic Counterfactual Flip Rate (Fairness Set)
* Evaluated on paired counterfactual profiles where sensitive attributes (`gender`, `region`, `nationality_group`) are swapped while holding all compliance variables constant:
  $$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\hat{Y}(P_i) \neq \hat{Y}(P_{i, \text{flip}})\right)$$

---

## 5. Statistical Testing & Multiple Testing Correction

* **Primary Family Correction:** The **Holm-Bonferroni step-down procedure** is applied across the 4 primary hypothesis comparisons (S3 vs S1b, S3 vs S1c, S3 vs S2, S2 vs S1b) at Family-Wise Error Rate $\alpha = 0.05$.
* **Paired Binary Error Tests:** Differences in paired binary classifications on matched test samples are tested using **McNemar's test** with continuity correction.
* **Confidence Intervals:** 95% bootstrap confidence intervals for all metrics are estimated via **1,000 bootstrap resamples** over test profiles.
* **Permutation Noise Floor Thresholds:**
  * Test Set ($N=4,000$): Permutation noise floor for $\Delta \text{FPR}_{\text{NT}}$ is $\pm 1.84\%$ at $R=0.50$ and $\pm 2.75\%$ at $R=0.70$.
  * Fairness Set ($N=1,500$): Permutation noise floor is $\pm 2.83\%$ at $R=0.50$ and $\pm 4.17\%$ at $R=0.70$.
  * Any observed disparity within these bounds cannot reject the null hypothesis of equal subgroup treatment.

---

## 6. Hard Leakage Sanity Check (Bayes-Oracle Ceiling)

* **The True Latent Bayes Oracle ROC-AUC on the held-out test set is 0.8857.**
* **Leakage Assertion Check:** Because the Bayes Oracle operates on the true unobserved sufficient statistic $Z$, no system operating solely on noisy surface observables can legitimately outperform 0.8857.
* **Protocol:** If any evaluated system achieves a Test ROC-AUC statistically significantly above 0.886 ($p < 0.01$), the run is immediately aborted and audited for feature leakage.

---

## 7. Derivation of Continuous Risk Score

All systems output a continuous `risk_score` $\in [0.0, 100.0]$:
* **S1 & S1b:** Directly generated by the single agent via calibrated probability output.
* **S1c (Self-Consistency):** Computed as the arithmetic mean of the $k$ individual stochastic risk scores:
  $$\text{risk\_score}_{\text{S1c}} = \frac{1}{k} \sum_{i=1}^k \text{risk\_score}_i$$
* **S2 (Hierarchical Consensus):** Directly calibrated by the Consensus Risk Scorer in the final synthesis node after reviewing both specialist reports.
* **S3 (Adversarial Debate):** Calibrated by the Final Impartial Adjudicator after weighing the Prosecutor's proposed risk score against the Advocate's mitigating arguments.

---

## 8. Error Handling & JSON Parse Failure Protocol

1. **Retry Limit:** Each individual LLM call has a maximum of **2 retries (total 3 attempts)** with exponential backoff on network errors or JSON syntax failures.
2. **Exhaustion Handling:** If a response fails to parse as valid JSON after 3 attempts:
   * The incident is logged in `results/raw_predictions.csv` with `json_parse_error = True` and raw response text recorded.
   * For pipeline safety, the fallback decision defaults to `"OFF-LOAD"` with `confidence = 0.0` and `risk_score = 100.0`.
   * All parse failures are tabulated and reported in the final deviations and robustness log.

---

## 9. Versioned Prompt Hashes (SHA-256)

All prompts are version-controlled in `experiments/prompts/` and frozen:
* `fia_rules_common.txt`: `b56b3d7ef298f5fbb0c1acef5068b3dfd35028863fb13c39a0258e70f602b243`
* `s1_single_agent.txt`: `a89a4185a0ae6f18622525621d248ccb53f5e2ee318ebdc4c51e6b66187636b2`
* `s1b_single_agent_mitigating.txt`: `e32132def426c3bff534bd9b49392fd456f0448bd6eb5a26ad2bedabb1282426`
* `s2_document_integrity.txt`: `9d09a7d53192e43f9c4a4efea8e9f397c646d9351f80a74bd608636c2def4403`
* `s2_socio_economic.txt`: `baf0cef7da7fd80cb02691eab81f7a61abfb2bc3a164105bb648b5c5428abf61`
* `s2_risk_scorer.txt`: `47f68f7b0030663b966f4ac56298c26ddb69996d8a50bce3c873b81f88d59b03`
* `s3_risk_scorer.txt`: `2be547aedc594c624eab4630a9c1e9e463aff447200198d207c02001374f7c03`
* `s3_traveler_advocate.txt`: `83c4e01b00c4be41434849ef61a1482749a44cca606ca5a81d9d7edbad1646c8`
* `s3_adjudicator.txt`: `650c358aaa6ff84d173712be6ab90a4c85f582adc57fd72f68fd542082e892a1`

---

## 10. Deviations Log
*(Any post-preregistration deviations during execution must be logged below with rationale and timestamp).*
* *None to date (Preregistration baseline).*
