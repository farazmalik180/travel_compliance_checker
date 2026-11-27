"""
Statistical Power Simulation and Permutation Noise Floor Calculation
for 1,000-profile stratified test subset (800 Genuine, 200 Threats).
"""
import numpy as np
import pandas as pd
from scipy import stats

def run_fpr_power_simulation(
    n_genuine=800,
    base_fpr=0.15,
    deltas=np.linspace(0.01, 0.06, 11),
    n_sims=1000,
    concordance=0.85,
    alpha=0.05,
    seed=2026
):
    np.random.seed(seed)
    power_results = []
    
    for delta in deltas:
        p_a = base_fpr
        p_b = base_fpr - delta
        
        # Paired contingency table probabilities under concordance rho
        # System A: marginal P(Y_A=1) = p_a
        # System B: marginal P(Y_B=1) = p_b
        # Discordant pairs:
        # P(Y_A=1, Y_B=0) = b
        # P(Y_A=0, Y_B=1) = c
        # b - c = p_a - p_b = delta
        # Concordance: P(Y_A == Y_B) = concordance -> b + c = 1 - concordance
        discordant_total = 1.0 - concordance
        # b - c = delta => 2b = discordant_total + delta => b = (discordant_total + delta) / 2
        p_10 = (discordant_total + delta) / 2.0
        p_01 = (discordant_total - delta) / 2.0
        p_11 = p_a - p_10
        p_00 = 1.0 - (p_11 + p_10 + p_01)
        
        probs = [p_00, p_01, p_10, p_11]
        probs = np.clip(probs, 0, 1)
        probs = probs / np.sum(probs)
        
        significant_count = 0
        for _ in range(n_sims):
            counts = np.random.multinomial(n_genuine, probs)
            # counts: [00, 01, 10, 11]
            n_01 = counts[1] # A=0, B=1
            n_10 = counts[2] # A=1, B=0
            
            # McNemar exact / normal approximation test for A > B (one-sided)
            # Test statistic z = (n_10 - n_01) / sqrt(n_10 + n_01)
            disc_sum = n_10 + n_01
            if disc_sum > 0:
                z = (n_10 - n_01 - 1.0) / np.sqrt(disc_sum) # with continuity correction
                p_val = 1.0 - stats.norm.cdf(z)
                if p_val < alpha:
                    significant_count += 1
                    
        power = significant_count / n_sims
        power_results.append({"delta_fpr": delta, "power": power})
        
    return pd.DataFrame(power_results)

def run_gap_power_simulation(
    n_trad=465,
    n_nt=335,
    base_gap=0.12, # A: FPR_NT = 0.22, FPR_Trad = 0.10
    deltas=np.linspace(0.02, 0.10, 9),
    n_sims=1000,
    concordance=0.85,
    alpha=0.05,
    seed=2026
):
    np.random.seed(seed)
    power_results = []
    
    for delta in deltas:
        # System B gap = base_gap - delta
        # We test whether Delta_Gap = Gap_A - Gap_B > 0
        significant_count = 0
        for _ in range(n_sims):
            # Traditional travelers (n_trad = 465)
            # A has fpr 0.10, B has fpr 0.10 (difference 0)
            p_10_tr = (1 - concordance) / 2.0
            p_01_tr = p_10_tr
            p_11_tr = 0.10 - p_10_tr
            p_00_tr = 1.0 - (p_11_tr + p_10_tr + p_01_tr)
            pr_tr = np.clip([p_00_tr, p_01_tr, p_10_tr, p_11_tr], 0, 1)
            pr_tr /= pr_tr.sum()
            c_tr = np.random.multinomial(n_trad, pr_tr)
            fp_A_tr = c_tr[2] + c_tr[3]
            fp_B_tr = c_tr[1] + c_tr[3]
            fpr_A_tr = fp_A_tr / n_trad
            fpr_B_tr = fp_B_tr / n_trad
            
            # Non-traditional travelers (n_nt = 335)
            # A has fpr 0.22, B has fpr 0.22 - delta
            p_10_nt = ((1 - concordance) + delta) / 2.0
            p_01_nt = ((1 - concordance) - delta) / 2.0
            p_11_nt = 0.22 - p_10_nt
            p_00_nt = 1.0 - (p_11_nt + p_10_nt + p_01_nt)
            pr_nt = np.clip([p_00_nt, p_01_nt, p_10_nt, p_11_nt], 0, 1)
            pr_nt /= pr_nt.sum()
            c_nt = np.random.multinomial(n_nt, pr_nt)
            fp_A_nt = c_nt[2] + c_nt[3]
            fp_B_nt = c_nt[1] + c_nt[3]
            fpr_A_nt = fp_A_nt / n_nt
            fpr_B_nt = fp_B_nt / n_nt
            
            gap_A = fpr_A_nt - fpr_A_tr
            gap_B = fpr_B_nt - fpr_B_tr
            diff_gap = gap_A - gap_B
            
            # Variance of difference in gaps
            # Var(gap_A - gap_B) = Var((fpr_A_nt - fpr_B_nt) - (fpr_A_tr - fpr_B_tr))
            var_diff_nt = (c_nt[1] + c_nt[2] - (c_nt[2] - c_nt[1])**2 / n_nt) / (n_nt**2)
            var_diff_tr = (c_tr[1] + c_tr[2] - (c_tr[2] - c_tr[1])**2 / n_trad) / (n_trad**2)
            se_diff = np.sqrt(var_diff_nt + var_diff_tr)
            
            if se_diff > 0:
                z = diff_gap / se_diff
                if stats.norm.cdf(z) > (1.0 - alpha):
                    significant_count += 1
                    
        power = significant_count / n_sims
        power_results.append({"delta_gap": delta, "power": power})
        
    return pd.DataFrame(power_results)

def calculate_permutation_noise_floors(
    n_genuine=800,
    n_trad=465,
    n_nt=335,
    base_fpr=0.15,
    n_permutations=10000,
    seed=2026
):
    np.random.seed(seed)
    # Under null hypothesis H0, system outputs are exchangeable
    # Simulate paired predictions under H0 with base_fpr
    p_disc = 0.15 # discordant rate under typical paired LLM agreement
    # Each discordant pair has 50% probability of being +1 or -1
    diff_fprs = []
    diff_gaps = []
    
    for _ in range(n_permutations):
        # Overall FPR difference across 800 genuine profiles
        # Under H0, number of discordant pairs ~ Binomial(800, p_disc)
        k_disc = np.random.binomial(n_genuine, p_disc)
        # Difference in positives is Binomial(k_disc, 0.5) - (k_disc - Binomial)
        k_pos = np.random.binomial(k_disc, 0.5)
        diff_count = 2 * k_pos - k_disc
        diff_fpr = diff_count / n_genuine
        diff_fprs.append(abs(diff_fpr))
        
        # Gap difference
        k_disc_nt = np.random.binomial(n_nt, p_disc)
        k_pos_nt = np.random.binomial(k_disc_nt, 0.5)
        diff_nt = (2 * k_pos_nt - k_disc_nt) / n_nt
        
        k_disc_tr = np.random.binomial(n_trad, p_disc)
        k_pos_tr = np.random.binomial(k_disc_tr, 0.5)
        diff_tr = (2 * k_pos_tr - k_disc_tr) / n_trad
        
        diff_gap = diff_nt - diff_tr
        diff_gaps.append(abs(diff_gap))
        
    diff_fprs = np.array(diff_fprs)
    diff_gaps = np.array(diff_gaps)
    
    noise_floors = {
        "FPR_Diff": {
            "SD": float(np.std(diff_fprs)),
            "Mean_Abs": float(np.mean(diff_fprs)),
            "P90": float(np.percentile(diff_fprs, 90)),
            "P95": float(np.percentile(diff_fprs, 95)),
            "P99": float(np.percentile(diff_fprs, 99))
        },
        "Gap_NT_Diff": {
            "SD": float(np.std(diff_gaps)),
            "Mean_Abs": float(np.mean(diff_gaps)),
            "P90": float(np.percentile(diff_gaps, 90)),
            "P95": float(np.percentile(diff_gaps, 95)),
            "P99": float(np.percentile(diff_gaps, 99))
        }
    }
    return noise_floors

if __name__ == "__main__":
    print("--- FPR Power Simulation (N=1,000 Subset, 800 Genuine) ---")
    df_fpr = run_fpr_power_simulation()
    print(df_fpr.to_string(index=False))
    
    print("\n--- Non-Traditional FPR Gap Power Simulation (N=465 Trad, N=335 NT) ---")
    df_gap = run_gap_power_simulation()
    print(df_gap.to_string(index=False))
    
    print("\n--- Permutation Noise Floors (10,000 Resamples under H0) ---")
    floors = calculate_permutation_noise_floors()
    import json
    print(json.dumps(floors, indent=2))

