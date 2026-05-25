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
  - Tornado diagram (sensitivity analysis)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results')
OUT.mkdir(exist_ok=True)

# ── Model parameters ────────────────────────────────────────────────────

# Time horizon
N_STEPS = 20          # 6-month steps = 10 years
STEP_YEARS = 0.5

# Cohort size (your dataset)
N_TOTAL = 531

# Tertile sizes (equal thirds)
N_LOW  = 177
N_MED  = 177
N_HIGH = 177

# Background mortality (Well → Dead) per 6-month step
# Age-matched women ~55y with breast cancer
P_BACKGROUND_MORT = 0.006

# Elevated mortality in CTRCD state (hazard ratio ~2.0 vs background)
P_CTRCD_MORT = P_BACKGROUND_MORT * 2.0

# Cardioprotection effect — reduces CTRCD mortality in high-risk arm
# when detected early (within first 4 steps)
# Based on published HR ~0.5 for cardioprotection vs no treatment
CARDIOPROTECTION_HR = 0.5
P_CTRCD_MORT_PROTECTED = P_CTRCD_MORT * CARDIOPROTECTION_HR

# Step threshold for "early detection" enabling cardioprotection
EARLY_DETECTION_STEP = 4

# ── Surveillance schedules ───────────────────────────────────────────────

UNIFORM_SCHEDULE = [0, 1, 2, 3, 4, 6, 8]   # current guidelines

SCHEDULES = {
    'low':  [0, 2, 4],                       # reduced
    'med':  [0, 1, 2, 3, 4, 6, 8],          # standard
    'high': [0, 1, 2, 3, 4, 5, 6, 7, 8],   # intensified
}

# ── Echo costs (USD, Medicare fee schedule) ──────────────────────────────

ECHO_COST = 800    # cost per echocardiogram

# ── Echo sensitivity ─────────────────────────────────────────────────────

ECHO_SENSITIVITY = 0.75

# ── Base transition probabilities ────────────────────────────────────────

P_CTRCD_BASE = {
    'low':  0.0058,
    'med':  0.0135,
    'high': 0.0538,
}

DECAY_RATE = 0.7

# ── Helper functions ─────────────────────────────────────────────────────

def get_p_ctrcd(grp, step):
    """
    Time-varying CTRCD transition probability.
    Full rate during active treatment (steps 0-4),
    exponential decay afterward.
    """
    base = P_CTRCD_BASE[grp]
    if step <= 4:
        return base
    else:
        return base * (DECAY_RATE ** (step - 4))


def run_markov(n_patients, grp, p_bg_mort, p_ctrcd_mort, n_steps,
               cardioprotection=False, schedule=None):
    """
    Run a Markov cohort simulation for one risk group.
    If cardioprotection=True and schedule provided, patients detected
    early (within EARLY_DETECTION_STEP) get reduced CTRCD mortality.
    """
    n_well    = np.zeros(n_steps + 1)
    n_ctrcd   = np.zeros(n_steps + 1)
    n_dead    = np.zeros(n_steps + 1)
    new_ctrcd = np.zeros(n_steps + 1)

    n_well[0] = n_patients

    # Fraction of CTRCD cases detected early under this schedule
    if cardioprotection and schedule is not None:
        early_echos = [s for s in schedule if s <= EARLY_DETECTION_STEP]
        frac_early  = len(early_echos) / max(len(schedule), 1)
    else:
        frac_early = 0.0

    # Blended CTRCD mortality: fraction early gets protected rate
    p_ctrcd_mort_effective = (
        frac_early       * P_CTRCD_MORT_PROTECTED +
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

    return n_well, n_ctrcd, n_dead, new_ctrcd


def count_echos(results, schedules, sizes, echo_cost):
    """
    Count total echos performed under a given surveillance schedule.
    Echo is only performed if patient is still alive (Well or CTRCD).
    """
    total_echos = 0
    for grp in groups:
        n_well, n_ctrcd, n_dead, _ = results[grp]
        schedule = schedules[grp]
        for step in schedule:
            alive = n_well[step] + n_ctrcd[step]
            total_echos += alive
    return total_echos, total_echos * echo_cost


def count_ctrcd_detected(results, schedules, sensitivity=ECHO_SENSITIVITY):
    """
    Count CTRCDs detected under a surveillance schedule.
    Accounts for imperfect echo sensitivity.
    """
    detected = 0
    for grp in groups:
        _, _, _, new_ctrcd = results[grp]
        schedule = sorted(schedules[grp])
        for t, n in enumerate(new_ctrcd):
            if n > 0:
                future_echos = [s for s in schedule if s >= t]
                if not future_echos:
                    continue
                p_missed_all = (1 - sensitivity) ** len(future_echos)
                p_detected   = 1 - p_missed_all
                detected    += n * p_detected
    return detected


# ── Run simulations ──────────────────────────────────────────────────────

print("=== Running Markov simulation ===")

groups = ['low', 'med', 'high']
sizes  = {'low': N_LOW, 'med': N_MED, 'high': N_HIGH}
uniform_schedules = {grp: UNIFORM_SCHEDULE for grp in groups}

# Uniform strategy — no cardioprotection differential
results_uniform = {}
for grp in groups:
    results_uniform[grp] = run_markov(
        n_patients   = sizes[grp],
        grp          = grp,
        p_bg_mort    = P_BACKGROUND_MORT,
        p_ctrcd_mort = P_CTRCD_MORT,
        n_steps      = N_STEPS,
    )

# Risk-stratified strategy — high-risk arm gets cardioprotection benefit
results_stratified = {}
for grp in groups:
    cp = (grp == 'high')
    results_stratified[grp] = run_markov(
        n_patients       = sizes[grp],
        grp              = grp,
        p_bg_mort        = P_BACKGROUND_MORT,
        p_ctrcd_mort     = P_CTRCD_MORT,
        n_steps          = N_STEPS,
        cardioprotection = cp,
        schedule         = SCHEDULES[grp],
    )

# ── Compute outcomes ─────────────────────────────────────────────────────

echos_uniform,     cost_uniform     = count_echos(
    results_uniform, uniform_schedules, sizes, ECHO_COST)
echos_stratified,  cost_stratified  = count_echos(
    results_stratified, SCHEDULES, sizes, ECHO_COST)

detected_uniform    = count_ctrcd_detected(results_uniform,    uniform_schedules)
detected_stratified = count_ctrcd_detected(results_stratified, SCHEDULES)

dead_uniform    = sum(results_uniform[grp][2][-1]    for grp in groups)
dead_stratified = sum(results_stratified[grp][2][-1] for grp in groups)

# ── Summary ──────────────────────────────────────────────────────────────

print(f"\n{'':=<55}")
print(f"{'STRATEGY':<25} {'ECHOS':>8} {'COST':>12} {'CTRCD DET':>10} {'DEATHS':>8}")
print(f"{'':=<55}")
print(f"{'Uniform (guidelines)':<25} {echos_uniform:>8.0f} "
      f"${cost_uniform:>10,.0f} {detected_uniform:>10.1f} {dead_uniform:>8.1f}")
print(f"{'Risk-stratified':<25} {echos_stratified:>8.0f} "
      f"${cost_stratified:>10,.0f} {detected_stratified:>10.1f} {dead_stratified:>8.1f}")
print(f"{'':=<55}")
print(f"\nEchos saved:      {echos_uniform - echos_stratified:.0f}")
print(f"Cost saved:       ${(cost_uniform - cost_stratified):,.0f}")
print(f"CTRCD missed:     {detected_uniform - detected_stratified:.1f}")
print(f"Deaths prevented: {dead_uniform - dead_stratified:.1f}")

# ── Sanity check: breakdown by group ─────────────────────────────────────

print("\n=== Breakdown by group ===")
for grp in groups:
    n_well, n_ctrcd, n_dead, new_ctrcd = results_stratified[grp]
    total_new = new_ctrcd.sum()
    print(f"\n{grp.upper()} risk (n={sizes[grp]}):")
    print(f"  Total CTRCD developed: {total_new:.1f}")
    print(f"  Alive at year 2 (step 4): {n_well[4]+n_ctrcd[4]:.1f}")
    print(f"  Alive at year 5 (step 10): {n_well[10]+n_ctrcd[10]:.1f}")
    print(f"  Uniform schedule echos: "
          f"{sum(results_uniform[grp][0][s]+results_uniform[grp][1][s] for s in UNIFORM_SCHEDULE):.1f}")
    print(f"  Stratified schedule echos: "
          f"{sum(n_well[s]+n_ctrcd[s] for s in SCHEDULES[grp]):.1f}")