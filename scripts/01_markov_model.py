"""
01_markov_model.py — Markov cohort simulation
Decision-analytic model comparing uniform vs risk-stratified
echo surveillance in HER2+ breast cancer patients.

Strategies compared:
  1. Current guidelines — uniform surveillance for all
  2. Risk-stratified — Cox model tertiles drive surveillance intensity

Outputs:
  - Total echos per strategy
  - CTRCDs detected
  - Cost per CTRCD detected
  - QALY analysis (dominance / ICER)
  - Cardioprotection HR sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results')
OUT.mkdir(exist_ok=True)

# ── Model parameters ─────────────────────────────────────────────────────

N_STEPS    = 20       # 6-month steps = 10 years
STEP_YEARS = 0.5

N_TOTAL = 531
N_LOW   = 177
N_MED   = 177
N_HIGH  = 177

# Background mortality (Well → Dead) per 6-month step
# Age-matched women ~55y with breast cancer
P_BACKGROUND_MORT = 0.006

# Elevated mortality in CTRCD state (HR ~2.0 vs background)
P_CTRCD_MORT = P_BACKGROUND_MORT * 2.0

# Cardioprotection effect — reduces CTRCD mortality in high-risk arm
# Base case: HR 0.50 (Guglin et al., lisinopril/carvedilol in AC+trastuzumab
#   high-risk subanalysis, HR 0.49–0.53)
# Conservative bound: HR 0.77 (SEER-Medicare adjusted HR, ACE inhibitor,
#   all-cause mortality)
CARDIOPROTECTION_HR       = 0.50
CARDIOPROTECTION_HR_RANGE = [0.50, 0.65, 0.77]
P_CTRCD_MORT_PROTECTED    = P_CTRCD_MORT * CARDIOPROTECTION_HR

# Step threshold for "early detection" enabling cardioprotection
EARLY_DETECTION_STEP = 4

# ── Utility weights ───────────────────────────────────────────────────────

# Well: HER2+ breast cancer, on treatment, no cardiac dysfunction
# Source: Kregting et al. 2024 (IJC); Portuguese cardio-oncology CEA model
U_WELL  = 0.77

# CTRCD: apply HF disutility decrement ~0.12
# Source: published EQ-5D HF utility decrements
U_CTRCD = 0.65

U_DEAD  = 0.0

# ── Surveillance schedules ────────────────────────────────────────────────

UNIFORM_SCHEDULE = [0, 1, 2, 3, 4, 6, 8]   # current guidelines (7 echos)

SCHEDULES = {
    'low':  [0, 2, 4],                       # reduced (3 echos)
    'med':  [0, 1, 2, 3, 4, 6, 8],          # standard (7 echos)
    'high': [0, 1, 2, 3, 4, 5, 6, 7, 8],   # intensified (9 echos)
}

# ── Costs ─────────────────────────────────────────────────────────────────

ECHO_COST = 800    # USD, Medicare fee schedule

# ── Echo sensitivity ──────────────────────────────────────────────────────

ECHO_SENSITIVITY = 0.75

# ── Base transition probabilities ─────────────────────────────────────────

P_CTRCD_BASE = {
    'low':  0.0058,
    'med':  0.0135,
    'high': 0.0538,
}

DECAY_RATE = 0.7

# ── Helper functions ──────────────────────────────────────────────────────

def get_p_ctrcd(grp, step):
    """Time-varying CTRCD transition probability with exponential decay
    after active treatment period (steps 0-4)."""
    base = P_CTRCD_BASE[grp]
    if step <= 4:
        return base
    return base * (DECAY_RATE ** (step - 4))


def run_markov(n_patients, grp, p_bg_mort, p_ctrcd_mort, n_steps,
               cardioprotection=False, schedule=None,
               p_ctrcd_mort_protected=None):
    """
    Run a Markov cohort simulation for one risk group.

    Returns: n_well, n_ctrcd, n_dead, new_ctrcd, qalys  (each length n_steps+1)

    p_ctrcd_mort_protected: explicit override for sensitivity analysis;
        defaults to global P_CTRCD_MORT_PROTECTED.
    """
    if p_ctrcd_mort_protected is None:
        p_ctrcd_mort_protected = P_CTRCD_MORT_PROTECTED

    n_well    = np.zeros(n_steps + 1)
    n_ctrcd   = np.zeros(n_steps + 1)
    n_dead    = np.zeros(n_steps + 1)
    new_ctrcd = np.zeros(n_steps + 1)

    n_well[0] = n_patients

    if cardioprotection and schedule is not None:
        early_echos = [s for s in schedule if s <= EARLY_DETECTION_STEP]
        frac_early  = len(early_echos) / max(len(schedule), 1)
    else:
        frac_early = 0.0

    p_ctrcd_mort_effective = (
        frac_early       * p_ctrcd_mort_protected +
        (1 - frac_early) * p_ctrcd_mort
    )

    for t in range(n_steps):
        p_ctrcd = get_p_ctrcd(grp, t)

        well_to_ctrcd = n_well[t] * p_ctrcd
        well_to_dead  = n_well[t] * p_bg_mort
        well_stay     = n_well[t] - well_to_ctrcd - well_to_dead

        ctrcd_to_dead = n_ctrcd[t] * p_ctrcd_mort_effective
        ctrcd_stay    = n_ctrcd[t] - ctrcd_to_dead

        n_well[t+1]    = well_stay
        n_ctrcd[t+1]   = ctrcd_stay + well_to_ctrcd
        n_dead[t+1]    = n_dead[t] + well_to_dead + ctrcd_to_dead
        new_ctrcd[t+1] = well_to_ctrcd

    qalys = np.zeros(n_steps + 1)
    for t in range(n_steps):
        qalys[t] = (n_well[t] * U_WELL + n_ctrcd[t] * U_CTRCD) * STEP_YEARS

    return n_well, n_ctrcd, n_dead, new_ctrcd, qalys


def count_echos(results, schedules, sizes, echo_cost):
    """Count total echos performed; echo only if patient alive (Well or CTRCD)."""
    total_echos = 0
    for grp in groups:
        n_well, n_ctrcd, n_dead, _new_ctrcd, _qalys = results[grp]
        for step in schedules[grp]:
            total_echos += n_well[step] + n_ctrcd[step]
    return total_echos, total_echos * echo_cost


def count_ctrcd_detected(results, schedules, sensitivity=ECHO_SENSITIVITY):
    """Count CTRCDs detected, accounting for imperfect echo sensitivity."""
    detected = 0
    for grp in groups:
        _, _, _, new_ctrcd, _ = results[grp]
        schedule = sorted(schedules[grp])
        for t, n in enumerate(new_ctrcd):
            if n > 0:
                future_echos = [s for s in schedule if s >= t]
                if not future_echos:
                    continue
                p_detected = 1 - (1 - sensitivity) ** len(future_echos)
                detected  += n * p_detected
    return detected


# ── Run simulations ───────────────────────────────────────────────────────

print("=== Running Markov simulation ===")

groups = ['low', 'med', 'high']
sizes  = {'low': N_LOW, 'med': N_MED, 'high': N_HIGH}
uniform_schedules = {grp: UNIFORM_SCHEDULE for grp in groups}

results_uniform = {}
for grp in groups:
    results_uniform[grp] = run_markov(
        n_patients   = sizes[grp],
        grp          = grp,
        p_bg_mort    = P_BACKGROUND_MORT,
        p_ctrcd_mort = P_CTRCD_MORT,
        n_steps      = N_STEPS,
    )

results_stratified = {}
for grp in groups:
    results_stratified[grp] = run_markov(
        n_patients             = sizes[grp],
        grp                    = grp,
        p_bg_mort              = P_BACKGROUND_MORT,
        p_ctrcd_mort           = P_CTRCD_MORT,
        n_steps                = N_STEPS,
        cardioprotection       = (grp == 'high'),
        schedule               = SCHEDULES[grp],
    )

# ── Compute primary outcomes ──────────────────────────────────────────────

echos_uniform,    cost_uniform    = count_echos(results_uniform,    uniform_schedules, sizes, ECHO_COST)
echos_stratified, cost_stratified = count_echos(results_stratified, SCHEDULES,         sizes, ECHO_COST)

detected_uniform    = count_ctrcd_detected(results_uniform,    uniform_schedules)
detected_stratified = count_ctrcd_detected(results_stratified, SCHEDULES)

dead_uniform    = sum(results_uniform[grp][2][-1]    for grp in groups)
dead_stratified = sum(results_stratified[grp][2][-1] for grp in groups)

qalys_uniform    = sum(results_uniform[grp][4].sum()    for grp in groups)
qalys_stratified = sum(results_stratified[grp][4].sum() for grp in groups)

delta_qalys = qalys_stratified - qalys_uniform
delta_cost  = cost_stratified  - cost_uniform

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\n{'':=<55}")
print(f"{'STRATEGY':<25} {'ECHOS':>8} {'COST':>12} {'CTRCD DET':>10} {'DEATHS':>8}")
print(f"{'':=<55}")
print(f"{'Uniform (guidelines)':<25} {echos_uniform:>8.0f} "
      f"${cost_uniform:>10,.0f} {detected_uniform:>10.1f} {dead_uniform:>8.1f}")
print(f"{'Risk-stratified':<25} {echos_stratified:>8.0f} "
      f"${cost_stratified:>10,.0f} {detected_stratified:>10.1f} {dead_stratified:>8.1f}")
print(f"{'':=<55}")
print(f"\nEchos saved:      {echos_uniform - echos_stratified:.0f}")
print(f"Cost saved:       ${cost_uniform - cost_stratified:,.0f}")
print(f"CTRCD missed:     {detected_uniform - detected_stratified:.1f}")
print(f"Deaths prevented: {dead_uniform - dead_stratified:.1f}")

# ── QALY analysis ─────────────────────────────────────────────────────────

print(f"\n{'':=<55}")
print(f"QALY ANALYSIS")
print(f"{'':=<55}")
print(f"QALYs (uniform):      {qalys_uniform:.2f}")
print(f"QALYs (stratified):   {qalys_stratified:.2f}")
print(f"Incremental QALYs:    {delta_qalys:.2f}")
print(f"Incremental cost:     ${delta_cost:,.0f}")

if delta_cost < 0 and delta_qalys > 0:
    print("Result: Risk-stratified DOMINATES (cheaper and more effective)")
elif delta_qalys > 0:
    print(f"ICER: ${delta_cost / delta_qalys:,.0f} per QALY gained")
else:
    print("Result: Stratified not dominant on QALYs — check parameters")

# ── Cardioprotection HR sensitivity ──────────────────────────────────────

print(f"\n{'':=<55}")
print(f"CARDIOPROTECTION HR SENSITIVITY")
print(f"{'':=<55}")
print(f"{'HR':<10} {'Deaths prevented':>18} {'Delta QALYs':>14}")
print(f"{'':=<45}")

for hr in CARDIOPROTECTION_HR_RANGE:
    p_protected_hr = P_CTRCD_MORT * hr
    results_cp = {}
    for grp in groups:
        results_cp[grp] = run_markov(
            n_patients             = sizes[grp],
            grp                    = grp,
            p_bg_mort              = P_BACKGROUND_MORT,
            p_ctrcd_mort           = P_CTRCD_MORT,
            n_steps                = N_STEPS,
            cardioprotection       = (grp == 'high'),
            schedule               = SCHEDULES[grp],
            p_ctrcd_mort_protected = p_protected_hr,
        )
    dead_cp  = sum(results_cp[grp][2][-1]   for grp in groups)
    qalys_cp = sum(results_cp[grp][4].sum() for grp in groups)
    print(f"{hr:<10.2f} {dead_uniform - dead_cp:>18.1f} {qalys_cp - qalys_uniform:>14.2f}")

# ── Breakdown by group ────────────────────────────────────────────────────

print("\n=== Breakdown by group ===")
for grp in groups:
    n_well, n_ctrcd, n_dead, new_ctrcd, _ = results_stratified[grp]
    print(f"\n{grp.upper()} risk (n={sizes[grp]}):")
    print(f"  Total CTRCD developed:     {new_ctrcd.sum():.1f}")
    print(f"  Alive at year 2 (step 4):  {n_well[4]+n_ctrcd[4]:.1f}")
    print(f"  Alive at year 5 (step 10): {n_well[10]+n_ctrcd[10]:.1f}")
    print(f"  Uniform schedule echos:    "
          f"{sum(results_uniform[grp][0][s]+results_uniform[grp][1][s] for s in UNIFORM_SCHEDULE):.1f}")
    print(f"  Stratified schedule echos: "
          f"{sum(n_well[s]+n_ctrcd[s] for s in SCHEDULES[grp]):.1f}")