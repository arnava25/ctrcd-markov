"""
02_sensitivity_analysis.py — One-way sensitivity analysis + tornado diagram
Tests robustness of Markov model conclusions across plausible parameter ranges.

For each parameter, we vary it across a low/high range while holding
all others at base case, and record the impact on key outcomes:
  - Echos saved
  - Cost saved
  - CTRCDs missed
  - Deaths prevented

Output: tornado diagram saved to results/tornado.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results')
OUT.mkdir(exist_ok=True)

# ── Copy core model parameters and functions ─────────────────────────────

N_STEPS = 20
N_LOW   = 177
N_MED   = 177
N_HIGH  = 177
groups  = ['low', 'med', 'high']
sizes   = {'low': N_LOW, 'med': N_MED, 'high': N_HIGH}

UNIFORM_SCHEDULE = [0, 1, 2, 3, 4, 6, 8]
SCHEDULES = {
    'low':  [0, 2, 4],
    'med':  [0, 1, 2, 3, 4, 6, 8],
    'high': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}
uniform_schedules = {grp: UNIFORM_SCHEDULE for grp in groups}

P_CTRCD_BASE = {
    'low':  0.0058,
    'med':  0.0135,
    'high': 0.0538,
}

# ── Base case parameters ─────────────────────────────────────────────────

BASE = {
    'echo_cost':            800,
    'echo_sensitivity':     0.75,
    'cardioprotection_hr':  0.50,
    'p_bg_mort':            0.006,
    'decay_rate':           0.70,
}

# ── Parameter ranges for sensitivity analysis ────────────────────────────

RANGES = {
    'echo_cost':           (600,  1200),
    'echo_sensitivity':    (0.65, 0.85),
    'cardioprotection_hr': (0.40, 0.70),
    'p_bg_mort':           (0.004, 0.010),
    'decay_rate':          (0.50, 0.90),
}

LABELS = {
    'echo_cost':           'Echo cost ($)',
    'echo_sensitivity':    'Echo sensitivity',
    'cardioprotection_hr': 'Cardioprotection HR',
    'p_bg_mort':           'Background mortality',
    'decay_rate':          'Risk decay rate',
}

# ── Model functions ──────────────────────────────────────────────────────

def get_p_ctrcd(grp, step, decay_rate):
    base = P_CTRCD_BASE[grp]
    if step <= 4:
        return base
    else:
        return base * (decay_rate ** (step - 4))


def run_markov(n_patients, grp, p_bg_mort, p_ctrcd_mort, n_steps,
               decay_rate, cardioprotection=False, schedule=None,
               cardioprotection_hr=0.5):
    n_well    = np.zeros(n_steps + 1)
    n_ctrcd   = np.zeros(n_steps + 1)
    n_dead    = np.zeros(n_steps + 1)
    new_ctrcd = np.zeros(n_steps + 1)
    n_well[0] = n_patients

    p_ctrcd_mort_protected = p_ctrcd_mort * cardioprotection_hr

    if cardioprotection and schedule is not None:
        early_echos = [s for s in schedule if s <= 4]
        frac_early  = len(early_echos) / max(len(schedule), 1)
    else:
        frac_early = 0.0

    p_ctrcd_mort_eff = (
        frac_early       * p_ctrcd_mort_protected +
        (1 - frac_early) * p_ctrcd_mort
    )

    for t in range(n_steps):
        p_ctrcd = get_p_ctrcd(grp, t, decay_rate)

        well_to_ctrcd = n_well[t] * p_ctrcd
        well_to_dead  = n_well[t] * p_bg_mort
        well_stay     = n_well[t] - well_to_ctrcd - well_to_dead

        ctrcd_to_dead = n_ctrcd[t] * p_ctrcd_mort_eff
        ctrcd_stay    = n_ctrcd[t] - ctrcd_to_dead

        n_well[t+1]    = well_stay
        n_ctrcd[t+1]   = ctrcd_stay + well_to_ctrcd
        n_dead[t+1]    = n_dead[t] + well_to_dead + ctrcd_to_dead
        new_ctrcd[t+1] = well_to_ctrcd

    return n_well, n_ctrcd, n_dead, new_ctrcd


def count_echos(results, schedules, echo_cost):
    total_echos = 0
    for grp in groups:
        n_well, n_ctrcd, _, _ = results[grp]
        for step in schedules[grp]:
            total_echos += n_well[step] + n_ctrcd[step]
    return total_echos, total_echos * echo_cost


def count_ctrcd_detected(results, schedules, sensitivity):
    detected = 0
    for grp in groups:
        _, _, _, new_ctrcd = results[grp]
        schedule = sorted(schedules[grp])
        for t, n in enumerate(new_ctrcd):
            if n > 0:
                future_echos = [s for s in schedule if s >= t]
                if not future_echos:
                    continue
                p_detected = 1 - (1 - sensitivity) ** len(future_echos)
                detected  += n * p_detected
    return detected


def run_both_strategies(params):
    """
    Run uniform and risk-stratified strategies with given parameters.
    Returns dict of key outcome differences (stratified - uniform).
    """
    p_bg_mort        = params['p_bg_mort']
    p_ctrcd_mort     = p_bg_mort * 2.0
    echo_cost        = params['echo_cost']
    sensitivity      = params['echo_sensitivity']
    cp_hr            = params['cardioprotection_hr']
    decay_rate       = params['decay_rate']

    # Uniform
    res_u = {}
    for grp in groups:
        res_u[grp] = run_markov(
            n_patients   = sizes[grp],
            grp          = grp,
            p_bg_mort    = p_bg_mort,
            p_ctrcd_mort = p_ctrcd_mort,
            n_steps      = N_STEPS,
            decay_rate   = decay_rate,
        )

    # Stratified
    res_s = {}
    for grp in groups:
        res_s[grp] = run_markov(
            n_patients       = sizes[grp],
            grp              = grp,
            p_bg_mort        = p_bg_mort,
            p_ctrcd_mort     = p_ctrcd_mort,
            n_steps          = N_STEPS,
            decay_rate       = decay_rate,
            cardioprotection = (grp == 'high'),
            schedule         = SCHEDULES[grp],
            cardioprotection_hr = cp_hr,
        )

    echos_u, cost_u = count_echos(res_u, uniform_schedules, echo_cost)
    echos_s, cost_s = count_echos(res_s, SCHEDULES,         echo_cost)
    det_u = count_ctrcd_detected(res_u, uniform_schedules, sensitivity)
    det_s = count_ctrcd_detected(res_s, SCHEDULES,         sensitivity)
    dead_u = sum(res_u[g][2][-1] for g in groups)
    dead_s = sum(res_s[g][2][-1] for g in groups)

    return {
        'echos_saved':       echos_u - echos_s,
        'cost_saved':        cost_u  - cost_s,
        'ctrcd_missed':      det_u   - det_s,
        'deaths_prevented':  dead_u  - dead_s,
    }


# ── Run sensitivity analysis ─────────────────────────────────────────────

print("=== Running sensitivity analysis ===")

# Base case
base_outcomes = run_both_strategies(BASE)
print(f"\nBase case:")
for k, v in base_outcomes.items():
    print(f"  {k}: {v:.1f}")

# One-way sensitivity
sensitivity_results = {}
for param, (low_val, high_val) in RANGES.items():
    results_low  = run_both_strategies({**BASE, param: low_val})
    results_high = run_both_strategies({**BASE, param: high_val})
    sensitivity_results[param] = {
        'low':  results_low,
        'high': results_high,
    }
    print(f"\n{LABELS[param]}:")
    print(f"  Low  ({low_val}): echos saved={results_low['echos_saved']:.0f}, "
          f"cost saved=${results_low['cost_saved']:,.0f}, "
          f"missed={results_low['ctrcd_missed']:.1f}, "
          f"deaths prevented={results_low['deaths_prevented']:.1f}")
    print(f"  High ({high_val}): echos saved={results_high['echos_saved']:.0f}, "
          f"cost saved=${results_high['cost_saved']:,.0f}, "
          f"missed={results_high['ctrcd_missed']:.1f}, "
          f"deaths prevented={results_high['deaths_prevented']:.1f}")


# ── Tornado diagram ──────────────────────────────────────────────────────

def make_tornado(sensitivity_results, base_outcomes, outcome_key,
                 xlabel, title, filename):
    """
    Draw a tornado diagram for one outcome variable.
    Bars show range of outcome as each parameter varies low → high.
    Sorted by impact magnitude (widest bar at top).
    """
    base_val = base_outcomes[outcome_key]

    bars = []
    for param, res in sensitivity_results.items():
        low_val  = res['low'][outcome_key]
        high_val = res['high'][outcome_key]
        bars.append({
            'label': LABELS[param],
            'low':   low_val,
            'high':  high_val,
            'range': abs(high_val - low_val),
        })

    # Sort by range descending
    bars = sorted(bars, key=lambda x: x['range'])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#fafaf8')
    ax.set_facecolor('#fafaf8')

    y_positions = range(len(bars))
    colors = ('#2E86AB', '#E84855')

    for i, bar in enumerate(bars):
        left  = min(bar['low'], bar['high'])
        right = max(bar['low'], bar['high'])
        # Low value bar
        ax.barh(i, bar['low']  - base_val, left=base_val,
                color=colors[0], alpha=0.85, height=0.5)
        # High value bar
        ax.barh(i, bar['high'] - base_val, left=base_val,
                color=colors[1], alpha=0.85, height=0.5)
        # Value labels
        ax.text(bar['low']  - (base_val - left) * 0.05, i,
                f'{bar["low"]:.1f}', va='center', ha='right', fontsize=8)
        ax.text(bar['high'] + (right - base_val) * 0.05, i,
                f'{bar["high"]:.1f}', va='center', ha='left', fontsize=8)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([b['label'] for b in bars])
    ax.axvline(base_val, color='#333333', linewidth=1.2, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='500', pad=12)
    ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.85, label='Low parameter value'),
        Patch(facecolor=colors[1], alpha=0.85, label='High parameter value'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150, bbox_inches='tight',
                facecolor='#fafaf8')
    print(f"Saved → results/{filename}")


print("\n=== Generating tornado diagrams ===")

make_tornado(sensitivity_results, base_outcomes,
             outcome_key='cost_saved',
             xlabel='Cost saved vs uniform surveillance (USD)',
             title='Tornado diagram: cost savings\n(risk-stratified vs uniform surveillance)',
             filename='tornado_cost.png')

make_tornado(sensitivity_results, base_outcomes,
             outcome_key='deaths_prevented',
             xlabel='Deaths prevented vs uniform surveillance',
             title='Tornado diagram: deaths prevented\n(risk-stratified vs uniform surveillance)',
             filename='tornado_deaths.png')

make_tornado(sensitivity_results, base_outcomes,
             outcome_key='ctrcd_missed',
             xlabel='CTRCDs missed vs uniform surveillance',
             title='Tornado diagram: CTRCDs missed\n(risk-stratified vs uniform surveillance)',
             filename='tornado_missed.png')

print("\n=== Sensitivity analysis complete ===")
print(f"Tornado diagrams saved to results/")