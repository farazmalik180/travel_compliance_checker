"""
Statistical Power Simulation and Permutation Noise Floor Calculation
for 200-profile test subset (160 Genuine, 40 Threats; 93 Trad, 67 Non-Trad).
Includes empirical concordance (rho = 0.93) and Holm-Bonferroni correction (alpha = 0.0125 to 0.05).
"""
import numpy as np
import pandas as pd
from scipy import stats
import json

def run_fpr_power_200(
    n_genuine=160,
    base_fpr=0.15,
    deltas=np.linspace(0.02, 0.14, 13),
    n_sims=2000,
    concordance=0.93,
    alpha=0.05,
    seed=2026
):
    np.random.seed(seed)
    power_results = []
    
    for delta in deltas:
        p_a = base_fpr
        discordant_total = 1.0 - concordance
        p_10 = (discordant_total + delta) / 2.0
        p_01 = (discordant_total - delta) / 2.0
        p_11 = p_a - p_10
        p_00 = 1.0 - (p_11 + p_10 + p_01)
        
        probs = np.clip([p_00, p_01, p_10, p_11], 0, 1)
        probs /= np.sum(probs)
        
        sig_unadj = 0
        sig_holm1 = 0 # alpha = 0.0125 (Holm step 1 for 4 hypotheses)
        
        for _ in range(n_sims):
            counts = np.random.multinomial(n_genuine, probs)
            n_01 = counts[1]
            n_10 = counts[2]
            disc_sum = n_10 + n_01
            if disc_sum > 0:
                z = (n_10 - n_01 - 1.0) / np.sqrt(disc_sum)
                p_val = 1.0 - stats.norm.cdf(z)
                if p_val < alpha:
                    sig_unadj += 1
                if p_val < (alpha / 4.0):
                    sig_holm1 += 1
                    
        power_results.append({
            "delta_fpr": round(delta, 3),
            "power_unadjusted_0.05": sig_unadj / n_sims,
            "power_holm_0.0125": sig_holm1 / n_sims
        })
        
    return pd.DataFrame(power_results)

def run_gap_power_200(
    n_trad=93,
    n_nt=67,
    base_gap=0.12,
    deltas=np.linspace(0.04, 0.20, 9),
    n_sims=2000,
    concordance=0.93,
    alpha=0.05,
    seed=2026
):
    np.random.seed(seed)
    power_results = []
    
    for delta in deltas:
        sig_unadj = 0
        sig_holm1 = 0
        
        for _ in range(n_sims):
            # Traditional (n=93)
            p_10_tr = (1 - concordance) / 2.0
            p_01_tr = p_10_tr
            p_11_tr = 0.10 - p_10_tr
            p_00_tr = 1.0 - (p_11_tr + p_10_tr + p_01_tr)
            pr_tr = np.clip([p_00_tr, p_01_tr, p_10_tr, p_11_tr], 0, 1)
            pr_tr /= pr_tr.sum()
            c_tr = np.random.multinomial(n_trad, pr_tr)
            
            # Non-trad (n=67)
            p_10_nt = ((1 - concordance) + delta) / 2.0
            p_01_nt = ((1 - concordance) - delta) / 2.0
            p_11_nt = 0.22 - p_10_nt
            p_00_nt = 1.0 - (p_11_nt + p_10_nt + p_01_nt)
            pr_nt = np.clip([p_00_nt, p_01_nt, p_10_nt, p_11_nt], 0, 1)
            pr_nt /= pr_nt.sum()
            c_nt = np.random.multinomial(n_nt, pr_nt)
            
            gap_A = (c_nt[2] + c_nt[3]) / n_nt - (c_tr[2] + c_tr[3]) / n_trad
            gap_B = (c_nt[1] + c_nt[3]) / n_nt - (c_tr[1] + c_tr[3]) / n_trad
            diff_gap = gap_A - gap_B
            
            var_diff_nt = (c_nt[1] + c_nt[2] - (c_nt[2] - c_nt[1])**2 / n_nt) / (n_nt**2)
            var_diff_tr = (c_tr[1] + c_tr[2] - (c_tr[2] - c_tr[1])**2 / n_trad) / (n_trad**2)
            se_diff = np.sqrt(var_diff_nt + var_diff_tr)
            
            if se_diff > 0:
                z = diff_gap / se_diff
                p_val = 1.0 - stats.norm.cdf(z)
                if p_val < alpha:
                    sig_unadj += 1
                if p_val < (alpha / 4.0):
                    sig_holm1 += 1
                    
        power_results.append({
            "delta_gap": round(delta, 3),
            "power_unadjusted_0.05": sig_unadj / n_sims,
            "power_holm_0.0125": sig_holm1 / n_sims
        })
        
    return pd.DataFrame(power_results)

def calculate_noise_floors_200(n_genuine=160, n_trad=93, n_nt=67, n_permutations=10000, seed=2026):
    np.random.seed(seed)
    p_disc = 0.07 # empirical discordant rate (1 - 0.93 concordance)
    diff_fprs = []
    diff_gaps = []
    
    for _ in range(n_permutations):
        k_disc = np.random.binomial(n_genuine, p_disc)
        k_pos = np.random.binomial(k_disc, 0.5)
        diff_fpr = (2 * k_pos - k_disc) / n_genuine
        diff_fprs.append(abs(diff_fpr))
        
        k_disc_nt = np.random.binomial(n_nt, p_disc)
        k_pos_nt = np.random.binomial(k_disc_nt, 0.5)
        diff_nt = (2 * k_pos_nt - k_disc_nt) / n_nt
        
        k_disc_tr = np.random.binomial(n_trad, p_disc)
        k_pos_tr = np.random.binomial(k_disc_tr, 0.5)
        diff_tr = (2 * k_pos_tr - k_disc_tr) / n_trad
        
        diff_gaps.append(abs(diff_nt - diff_tr))
        
    diff_fprs = np.array(diff_fprs)
    diff_gaps = np.array(diff_gaps)
    
    return {
        "FPR_Diff_200": {
            "SD": float(np.std(diff_fprs)),
            "Mean_Abs": float(np.mean(diff_fprs)),
            "P90": float(np.percentile(diff_fprs, 90)),
            "P95": float(np.percentile(diff_fprs, 95)),
            "P99": float(np.percentile(diff_fprs, 99))
        },
        "Gap_NT_Diff_200": {
            "SD": float(np.std(diff_gaps)),
            "Mean_Abs": float(np.mean(diff_gaps)),
            "P90": float(np.percentile(diff_gaps, 90)),
            "P95": float(np.percentile(diff_gaps, 95)),
            "P99": float(np.percentile(diff_gaps, 99))
        }
    }

if __name__ == "__main__":
    print("=== FPR POWER SIMULATION (N = 200, Genuine = 160, Concordance = 0.93) ===")
    df_fpr = run_fpr_power_200()
    print(df_fpr.to_string(index=False))
    
    print("\n=== NON-TRADITIONAL GAP POWER SIMULATION (Trad = 93, NT = 67) ===")
    df_gap = run_gap_power_200()
    print(df_gap.to_string(index=False))
    
    print("\n=== PERMUTATION NOISE FLOORS FOR N = 200 (10,000 resamples under H0) ===")
    floors = calculate_noise_floors_200()
    print(json.dumps(floors, indent=2))

