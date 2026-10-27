# Synthetic Dataset: Data Generating Process & Fairness Architecture

## 1. Overview & Research Objective
This dataset contains 5,000 synthetic passenger profiles designed to benchmark Single-Agent vs. Hierarchical vs. Peer-to-Peer Multi-Agent LLM architectures on border immigration compliance checking and demographic fairness.

The profiles are partitioned into:
* **`dataset_dev.csv` (1,000 profiles, 20%):** Designated exclusively for prompt development, error analysis, and pilot calibration.
* **`dataset_test.csv` (4,000 profiles, 80%):** Held-out test set strictly frozen before prompt finalization to prevent data leakage and hyper-parameter tuning on evaluation data.

---

## 2. Circularity Status: Reduced, Not Eliminated

In synthetic AI evaluation, **circularity** occurs when the synthetic data generator embeds the same heuristic rules that evaluated agents are instructed to check. When this happens, multi-agent systems simply mimic the generator's rulebook, inflating apparent benchmark accuracy.

### Mechanisms Implemented to Reduce Circularity:
1. **Prior Latent Ground Truth:** The ground-truth label $Y^* \in \{0, 1\}$ (0 = Genuine Traveler, 1 = Genuine Threat) is drawn first from a Bernoulli prior ($\Pr(Y^*=1) = 0.20$), independent of any observable attributes or official FIA rules.
2. **Smooth Latent Driver ($Z$):** Each traveler receives an underlying continuous latent intent score $Z \sim \text{Beta}(\alpha, \beta)$ conditioned solely on $Y^*$. Every observable feature (authenticity, ties, funds, tickets, stamps) is a continuous smooth function or logit-link probability of $Z$ rather than a deterministic discrete heuristic.
3. **Agent Isolation:** No agent prompt, system message, or LangGraph state ever has access to $Y^*$ or $Z$. The agents must infer compliance and flight risk purely from noisy, surface-level observable features.

### Acknowledged Rule-Alignment Threat:
**Circularity is reduced, but NOT completely eliminated.** 
The functional relationships linking $Z$ to observables (e.g., lower protector stamp probability or higher visit visa prevalence for higher $Z$) reflect the author's conceptual model of immigration non-compliance. Because the agent prompts are also instructed on official departure rules, there exists an inherent **rule-alignment threat**: the generative model and the agent prompts share an underlying ontology of what constitutes "risk." Consequently, findings must be interpreted as demonstrating relative multi-agent reasoning dynamics within this modeled universe rather than real-world predictive validity.

---

## 3. Deliberate Overlap & Realistic Hard Cases

The data generating process intentionally models two real-world failure modes to avoid trivial separability:

### A. Non-Traditional Genuine Travelers (Measuring False Positives)
* **Archetype:** Freelance software engineers, ride-hailing drivers, small artisans, remote contractors, and first-time young travelers.
* **Independence Verification:** `is_non_traditional` is generated conditional strictly on `age_band` and is **statistically independent of `ground_truth_label`** (confirmed via Pearson Chi-Square test: $\chi^2 = 0.0354, p = 0.8508$).
* **Surface Appearance:** Irregular monthly income, moderate bank balance, fresh passport, and unconventional local ties.
* **Benchmark Role:** Tests whether multi-agent debate (Traveler Advocate) reduces wrongful offloading compared to a rigid single-agent rule-checker.

### B. Sophisticated Threat Camouflage (Measuring False Negatives)
* **Archetype:** Irregular migrants or human trafficking recruits coached by unauthorized recruitment agencies.
* **Surface Appearance:** Confirmed return tickets (72%), temporary borrowed "show money" in bank accounts (~1.2M PKR), and confirmed hotel bookings.
* **Benchmark Role:** Prevents trivial heuristic keyword matching.

---

## 4. Complete Parameter Manifest & Source Attributions

All values below reflect author-specified modeling choices except where statutory regulations are cited.

| Feature / Parameter | Generative Formula / Probability | Source / Justification |
| :--- | :--- | :--- |
| **`p_threat` (Base Rate)** | $\Pr(Y^* = 1) = 0.20$ | **Author-specified assumption** (ensures statistical power for recall). |
| **$Z \mid Y^*=0$ (Genuine)** | $\text{Beta}(2.0, 4.5)$ ($\mu \approx 0.31, \sigma \approx 0.16$) | **Author-specified assumption** (smooth low-risk distribution). |
| **$Z \mid Y^*=1$ (Threat)** | $\text{Beta}(4.2, 2.8)$ ($\mu \approx 0.60, \sigma \approx 0.17$) | **Author-specified assumption** (smooth high-risk distribution with overlap). |
| **`gender`** | Male: 62%, Female: 36%, Non-binary: 2% | **Author-specified assumption** (reflecting South Asian departure proportions). |
| **`age_band`** | 18-25: 25%, 26-35: 42%, 36-50: 23%, 51+: 10% | **Author-specified assumption** (skewed toward working-age adults). |
| **`region`** | Punjab: 52%, Sindh: 22%, KP: 15%, Balochistan: 6%, Isb/AJK/GB: 5% | **Author-specified assumption** (census-informed provincial distribution). |
| **`is_non_traditional`** | $P(\text{NT} \mid \text{age}) \in \{0.55, 0.45, 0.30, 0.15\}$ | **Author-specified assumption** (younger cohorts have higher freelance rates). |
| **`local_ties_score`** | $\text{clip}(82.0 - 45.0 \cdot Z - 12.0 \cdot \text{NT} + \mathcal{N}(0, 10), 5, 98)$ | **Author-specified assumption** (smooth continuous decay with $Z$). |
| **`bank_funds_pkr`** | $\text{lognormal}(14.2 - 1.6 Z - 0.6 \text{NT}, 0.65) + \text{ShowMoney}$ | **Author-specified assumption** (lognormal distribution + show money boost). |
| **`has_return_ticket`** | $\text{logit}^{-1}(3.2 - 2.6 Z)$ (Visit) / $\text{logit}^{-1}(-1.2 - 0.6 Z)$ (Work) | **Author-specified assumption** (smooth decay). |
| **`has_hotel_booking`** | $\text{logit}^{-1}(2.9 - 2.4 Z)$ (Visit) / $\text{logit}^{-1}(0.8 - 1.2 Z)$ (Work) | **Author-specified assumption** (smooth decay). |
| **`has_financial_proof`** | $\text{logit}^{-1}(2.6 - 2.2 Z)$ | **Author-specified assumption** (smooth decay). |
| **`has_protector_stamp`** | $\text{logit}^{-1}(2.4 - 4.2 Z)$ (Work visa only) | **Statutory Requirement:** Emigration Ordinance 1979 / Bureau of Emigration guidelines. Failure rate linked to $Z$ via **author assumption**. |
| **`high_scrutiny_dests`** | Monitored list (Cambodia, UAE, Thailand, Iraq, etc.) | **Author-specified assumption** (Heuristic compilation based on publicly reported high-scrutiny hubs). |
| **`doc_authenticity`** | $\text{clip}(94.0 - 32.0 \cdot Z + \mathcal{N}(0, 8), 15, 99)$ | **Author-specified assumption** (forensic security feature score). |
| **`watchlist_flag`** | $P(\text{Alert}) = \max(0, (Z - 0.70) \cdot 0.40)$; $P(\text{Match}) = 0.05 + 0.12 Z$ | **Author-specified assumption** (simulates IBMS hit and common-name noise). |

---

## 4.1 Provenance of Repository FIA Rules
The departure guidelines embedded in this repository (e.g. in `backend/scripts/generate_rules_pdf.py` and `backend/agents/nodes.py`) were created by the repository author as an experimental mock representation of Pakistani departure screening. 
* **Statutory Elements:** The mandatory Protector Stamp for labor migrants departing on work visas directly reflects statutory provisions of Pakistan's *Emigration Ordinance 1979* and Bureau of Emigration regulations.
* **Administrative Guidelines:** Requirements regarding hotel reservations, return tickets on visit visas, and specific scrutiny for transit destinations (e.g. show money for Cambodia) represent author-synthesized summaries of publicly circulated travel agency advisories and news reporting. 
* **Institutional Clarification:** These rules do **not** originate from an official, live API or verified administrative database of Pakistan's Federal Investigation Agency (FIA).

---

## 5. Machine Learning Ceiling Check

To guarantee that observable features do not allow trivial separability (AUC > 0.95), models were trained on `dataset_dev.csv` (N=1,000) and evaluated on `dataset_test.csv` (N=4,000):
* **True Latent Bayes Oracle:** Theoretical ceiling ROC-AUC = **0.8857** (using true Beta priors over $Z$).
* **Logistic Regression:** Test ROC-AUC = **0.8077**, Accuracy = **82.38%**
* **HistGradientBoosting:** Test ROC-AUC = **0.7834**, Accuracy = **81.53%**
* **Always-Clear Reference Baseline:** Accuracy = **80.00%**, Recall = **0.00%**, FPR = **0.00%**
* **Always-Offload Reference Baseline:** Accuracy = **20.00%**, Recall = **100.00%**, FPR = **100.00%**

These metrics confirm that the classification problem possesses realistic epistemic uncertainty (empirical AUC in the 0.78–0.81 range), well below the 0.95 ceiling threshold.

---

## 6. Dataset Limitations
1. **Model Parameterization:** All probability link slopes and baseline means reflect author assumptions rather than direct empirical administrative data from border gates.
2. **Discrete Document Proxies:** Physical passports, visas, and airline tickets are represented as structured indicators rather than raw image pixels.
3. **Prevalence Inflation:** The 20% threat prevalence was chosen to provide sufficient statistical sample size for True Positive Rate / Recall calculations across demographic slices.
