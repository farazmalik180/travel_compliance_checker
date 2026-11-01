# Formal Preregistration: Multi-Agent Travel Compliance & Demographic Fairness Benchmarking

**Author:** MS AI Thesis Research  
**Date of Preregistration:** November 2026  
**Protocol Version:** v5.0-Free (Estimation-Focused Free-Tier Execution Protocol)  
**Status:** Preregistered specification prior to execution of benchmark calls on held-out test splits.

---

## 1. Study Overview & Core Research Question
This study investigates whether multi-agent LLM architectures (Hierarchical Consensus and Peer-to-Peer Adversarial Debate) reduce wrongful offloading (False Positive Rate) and mitigate demographic/livelihood bias against non-traditional travelers (gig workers, freelancers, informal earners) compared to single-agent baselines and token-matched self-consistency ensembles, within the high-stakes domain of immigration departure screening.

To operate strictly within the verifiable constraints of the **Groq Free Tier (8,000 TPM, 200,000 TPD)** with zero paid billing, Version 5.0-Free restructures the experimental design into an **estimation-focused benchmarking framework** supported by bootstrap confidence intervals, nested test cohorts, and automated daily token-budget scheduling.

---

## 2. Evaluated Systems & Inference Controls

All LLM systems evaluate profiles using `openai/gpt-oss-120b` on Groq with frozen prompts (`prompts-frozen-v1`), fixed reasoning effort (`"medium"`), and `max_tokens = 4096`. Code harness and rate-limiters will be locked under git tag `code-frozen-v1` immediately before test execution.

| System ID | Name | Architecture | Prompt / Interaction Mode | Temperature | Expected Calls / Profile | Role in Study |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | **Deterministic Rule Engine** | Pure Code Engine (No LLM) | Applies statutory rules from `fia_rules_common.txt` | N/A | 0 calls | Procedural Lower-Bound Reference (evaluated on $N=4,000$, $N=200$, $N=100$) |
| **ML-Base** | **Supervised ML Classifiers** | Scikit-Learn / Gradient Boosting | Tabular Logistic Regression & HistGradientBoosting | N/A | 0 calls | Non-LLM Tabular Baselines (evaluated on $N=4,000$, $N=200$, $N=100$) |
| **S1** | **Single-Agent Baseline** | Monolithic LLM call | Strict departure compliance prompt | $T = 0.0$ | 1 call | Reference Baseline Only |
| **S1b** | **Single-Agent + Context** | Monolithic LLM call | S1 prompt + mitigating-context instruction block | $T = 0.0$ | 1 call | Primary Comparison (Prompt Control) |
| **S1c** | **Self-Consistency Ensemble** | Majority Voting Ensemble | $k$ repeated stochastic calls of S1; majority vote decision; mean risk score. $k$ calibrated from dev pilot | $T = 0.7$ | $k$ calls ($k = 6$, evaluated on $N=100$) | Primary Comparison (Compute Control) |
| **S2** | **Hierarchical Consensus** | Sequential Multi-Agent Pipeline | Routing: Document Integrity Specialist $\to$ Socio-Economic Specialist $\to$ Consensus Risk Adjudicator | $T = 0.0$ | 3 calls | Primary Comparison |
| **S3** | **Adversarial Debate** | Multi-Turn Debate Pipeline | Multi-round debate: Risk Scorer (Prosecutor) $\leftrightarrow$ Traveler Advocate (up to 3 rounds) $\to$ Final Adjudicator (Judge) | $T = 0.0$ | 3 to 7 calls (mean ~3.57 calls) | Proposed Primary System |

---

## 3. Evaluation Cohorts & Free-Tier Sampling Architecture

To balance rigorous comparative measurement against the 190,000 TPD safety budget, the study implements nested evaluation cohorts:

```
Full Test Distribution (N = 4,000 profiles, SHA-256: dataset_test.csv)
  └─ Evaluated: S0 Deterministic Rules & ML Baselines (0 API calls, $0.00 compute)
  │
  └── Core Stratified Test Subset (N = 200 profiles, SHA-256: 2ec4419b23af...)
        ├─ Strata: 160 Genuine (93 Trad, 67 Non-Trad), 40 Threats (24 Trad, 16 Non-Trad)
        ├─ Evaluated: S1, S1b, S2, S3
        │
        └── Nested Compute-Control Subset (N = 100 profiles, SHA-256: 862dd90f4eda...)
              ├─ Strata: 80 Genuine (47 Trad, 33 Non-Trad), 20 Threats (12 Trad, 8 Non-Trad)
              └─ Evaluated: S1c (k = 6 calls = 600 LLM invocations)
```

### Cohort Definitions & Scope Adjustments:
1. **Cohort A: Full Test Set ($N = 4,000$ profiles):**  
   Evaluates S0 and supervised ML baselines across the complete held-out test set at zero API cost, establishing the broad population performance bounds.
2. **Cohort B: Core Test Subset ($N = 200$ profiles):**  
   Drawn from the seeded 1,000-ID subset using `StratifiedShuffleSplit` (seed `2026`, jointly stratified by `ground_truth_label` $\times$ `is_non_traditional`). Evaluates $S_1, S_{1b}, S_2, S_3$.
3. **Cohort C: Nested Compute-Control Subset ($N = 100$ profiles):**  
   Drawn directly from inside the 200-ID subset using seed `2026`. Evaluates $S_{1c}$ ($k = 6$) to verify whether compute scaling accounts for multi-agent gains.
4. **Fairness Set Adjustment:**  
   The standalone 500-profile fairness set is **dropped** to conserve token budget; non-traditional fairness and subgroup disparities are evaluated directly across the 200-profile core test cohort.
5. **Cohort D: Counterfactual Gender Flips ($N = 60$ base profiles $\times$ 3 copies):**  
   Focuses strictly on the primary comparison of interest ($S_{1b}$ Prompt Control vs $S_3$ Adversarial Debate). Profiles are evaluated in triplicate (Null Control 1, Null Control 2, Flipped Profile) on Gender (Male $\leftrightarrow$ Female) to measure net sensitivity beyond stochastic variance:
   $$\text{Flip Rate}_{\text{net}} = \max\left(0, \text{Flip Rate}_{\text{flipped}} - \text{Flip Rate}_{\text{null\_control}}\right)$$

---

## 4. Analytical Framework: Estimation with Bootstrap Confidence Intervals

Given $N = 200$, statistical power simulations establish that the sample is well-powered to detect moderate-to-large FPR reductions ($\text{MDE}_{80\%} = 5.5\%$ unadjusted, $6.5\%$ Holm-adjusted), but underpowered for subtle bias gap differences ($\text{MDE}_{80\%} = 16.1\%$). 

Consequently, the confirmatory hypothesis-testing family ($H_{1a} - H_{1d}$) is designated as **exploratory / direction-testing**, and the primary analytical framework is re-centered on **precision parameter estimation**:
1. **Primary Reporting Standard:** Every metric difference ($\Delta \text{FPR}$, $\Delta \text{ROC-AUC}$, $\Delta \text{FPR}_{\text{disc}}$, $\Delta \text{Gap}_{\text{NT}}$) is reported with **10,000-resample paired bootstrap 95% Confidence Intervals** (bias-corrected and accelerated, BCa).
2. **Permutation Noise Floor Auditing:** Empirical differences are benchmarked against the 95th percentile permutation noise floor under the sharp null hypothesis $H_0$ ($4.38\%$ for $\Delta \text{FPR}$, $8.36\%$ for $\Delta \text{Gap}_{\text{NT}}$).
3. **Dual Error-Handling Reporting:**
   - Complete-Case Analysis (Primary): Unparseable outputs dropped from denominator.
   - Sensitivity Worst-Case Analysis (Secondary): Unparseable outputs imputed as `OFF-LOAD`.

---

## 5. Calibration of Compute Control ($S_{1c}$)
* Parameter $k$ is calibrated from dev-pilot token means:
  $$k = \text{round}\left(\frac{\bar{T}_{S3}}{\bar{T}_{S1}}\right)$$
* Empirical measurements across dev profiles indicate $\bar{T}_{S3} = 7,046.7$ tokens and $\bar{T}_{S1} = 1,262.1$ tokens, yielding an exact ratio of **$5.583$ (SE = $0.867$, 95% CI: [3.88, 7.28])**.
* Locked integer setting: **$k = 6$ stochastic calls** ($T = 0.7$).

---

## 6. Daily Token Guard & Automated Unattended Schedule

### A. Guardrail Settings
* **TPM Bucket Limit:** 7,000 tokens/min sliding window (enforced by `_wait_for_token_bucket` in `base.py`).
* **Daily Token Safety Budget:** **190,000 tokens/day** (95% of Groq's 200k TPD cap, halting calls before HTTP 429).
* **Auto-Resume Mechanism:** The unattended runner logs progress, preserves `.jsonl` checkpoints, sleeps until `00:00:05 UTC` (`05:00:05 PKT`), resets the daily counter, and resumes execution automatically.

### B. Day-by-Day Free-Tier Execution Schedule

| Day | Target Workload | Evaluated Systems & Profiles | Expected Tokens | Daily Safety Budget | Cumulative Progress |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Day 1** | Dev Pilot Extension | $S_1, S_{1b}, S_2, S_3$ on 40 dev profiles | ~180k tokens | 190,000 tokens | Dev pilot & claim audit completed |
| **Day 2** | Test Cohort (Part 1) + Baselines | $S_1, S_{1b}, S_2$ on 200 profiles + $S_0$ & ML on 4,000 | ~140k tokens | 190,000 tokens | Baselines & single/hierarchical test completed |
| **Day 3** | Test Cohort (Part 2) | $S_3$ Adversarial Debate on 200 test profiles | ~180k tokens | 190,000 tokens | All primary multi-agent test runs completed |
| **Day 4** | Compute Control | $S_{1c}$ ($k=6$, 600 calls) on 100 nested test profiles | ~150k tokens | 190,000 tokens | S1c compute benchmark completed |
| **Day 5** | Counterfactual Flips | $S_{1b}$ and $S_3$ on 60 triplicate gender profiles | ~160k tokens | 190,000 tokens | Counterfactual sensitivity completed |
| **Day 6** | Synthesis & Reporting | Bootstrap CIs (10k resamples), ROC curves, tables | 0 tokens (local) | 190,000 tokens | Thesis chapter empirical tables compiled |

---

## 7. Changes from Version 4.0 Specification

| Specification Dimension | Version 4.0 (Pre-Registration) | Version 5.0-Free (Current Protocol) | Methodological Rationale |
| :--- | :--- | :--- | :--- |
| **Test Cohort Size** | $N = 4,000$ profiles across all LLM systems | $N = 200$ for LLMs; $N = 4,000$ for $S_0$ and ML | Eliminates 95% of token demand while preserving full-sample bounds on deterministic rules and ML. |
| **Analytical Mode** | Strict NHST confirmatory hypothesis testing | Estimation-focused with 10k bootstrap 95% CIs | Avoids binary over-claiming on underpowered sample sizes; focuses on magnitude and precision of effect estimates. |
| **Self-Consistency ($S_{1c}$)** | Evaluated on full test set | Evaluated on nested $N = 100$ subset with $k=6$ | Maintains direct compute-matching against S3 while capping ensemble calls to 600. |
| **Fairness Cohort** | 500 standalone profiles | Dropped as separate cohort; analyzed in 200-ID test set | Disparity metrics ($\Delta \text{FPR}_{\text{NT}}$) computed directly over 67 non-traditional test profiles. |
| **Counterfactual Flips** | 500 pairs $\times$ 3 attributes $\times$ 5 systems | 60 triplicate base profiles, Gender only, on $S_{1b}$ & $S_3$ | Preserves core comparison (Prompt Control vs Debate) with rigorous null controls under daily cap. |
| **Rate Limit Guard** | Unbounded paid-tier concurrency (250k TPM) | 7,000 TPM bucket, 190k TPD cap, unattended auto-sleep | Guarantees 100% compliance with Groq free-tier quotas with zero risk of HTTP 429 corruption. |
| **Advocate Claim Audit** | Informal post-hoc review | Deterministic non-LLM auditor with 100% injected recall | Validates factual grounding without consuming API tokens or introducing LLM grading bias. |

