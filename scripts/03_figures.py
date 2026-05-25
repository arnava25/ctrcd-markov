"""
03_figures.py — Publication figures for Markov surveillance paper
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path('results/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────
C = {
    'low':       '#2196A6',
    'med':       '#F5A623',
    'high':      '#D94F3D',
    'uniform':   '#6B7280',
    'stratified':'#2D6A4F',
    'bg':        '#FAFAF8',
}

# ── Paste core model (same as 01 and 02) ────────────────────────────────

N_STEPS = 20
N_LOW, N_MED, N_HIGH = 177, 177, 177
groups = ['low', 'med', 'high']
sizes  = {'low': N_LOW, 'med': N_MED, 'high': N_HIGH}

UNIFORM_SCHEDULE = [0, 1, 2, 3, 4, 6, 8]
SCHEDULES = {
    'low':  [0, 2, 4],
    'med':  [0, 1, 2, 3, 4, 6, 8],
    'high': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}
uniform_schedules = {grp: UNIFORM_SCHEDULE for grp in groups}

P_CTRCD_BASE   = {'low': 0.0058, 'med': 0.0135, 'high': 0.0538}
DECAY_RATE     = 0.70
P_BG_MORT      = 0.006
P_CTRCD_MORT   = P_BG_MORT * 2.0
CP_HR          = 0.50
P_CTRCD_MORT_P = P_CTRCD_MORT * CP_HR
ECHO_COST      = 800
ECHO_SENS      = 0.75

def get_p_ctrcd(grp, step):
    base = P_CTRCD_BASE[grp]
    return base if step <= 4 else base * (DECAY_RATE ** (step - 4))

def run_markov(n_patients, grp, cardioprotection=False, schedule=None):
    n_well    = np.zeros(N_STEPS + 1)
    n_ctrcd   = np.zeros(N_STEPS + 1)
    n_dead    = np.zeros(N_STEPS + 1)
    new_ctrcd = np.zeros(N_STEPS + 1)
    n_well[0] = n_patients

    if cardioprotection and schedule is not None:
        early = [s for s in schedule if s <= 4]
        frac  = len(early) / max(len(schedule), 1)
    else:
        frac = 0.0

    p_mort_eff = frac * P_CTRCD_MORT_P + (1 - frac) * P_CTRCD_MORT

    for t in range(N_STEPS):
        p_c = get_p_ctrcd(grp, t)
        w2c = n_well[t] * p_c
        w2d = n_well[t] * P_BG_MORT
        c2d = n_ctrcd[t] * p_mort_eff
        n_well[t+1]    = n_well[t]  - w2c - w2d
        n_ctrcd[t+1]   = n_ctrcd[t] - c2d + w2c
        n_dead[t+1]    = n_dead[t]  + w2d + c2d
        new_ctrcd[t+1] = w2c
    return n_well, n_ctrcd, n_dead, new_ctrcd

def count_echos(results, schedules):
    total = 0
    for grp in groups:
        w, c, _, _ = results[grp]
        for s in schedules[grp]:
            total += w[s] + c[s]
    return total

def count_detected(results, schedules):
    det = 0
    for grp in groups:
        _, _, _, nc = results[grp]
        sched = sorted(schedules[grp])
        for t, n in enumerate(nc):
            if n > 0:
                fe = [s for s in sched if s >= t]
                if fe:
                    det += n * (1 - (1 - ECHO_SENS) ** len(fe))
    return det

# Run both strategies
res_u, res_s = {}, {}
for grp in groups:
    res_u[grp] = run_markov(sizes[grp], grp)
    res_s[grp] = run_markov(sizes[grp], grp,
                            cardioprotection=(grp == 'high'),
                            schedule=SCHEDULES[grp])

echos_u = count_echos(res_u, uniform_schedules)
echos_s = count_echos(res_s, SCHEDULES)
det_u   = count_detected(res_u, uniform_schedules)
det_s   = count_detected(res_s, SCHEDULES)
dead_u  = sum(res_u[g][2][-1] for g in groups)
dead_s  = sum(res_s[g][2][-1] for g in groups)
cost_u  = echos_u * ECHO_COST
cost_s  = echos_s * ECHO_COST

# ════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Model schematic
# ════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(C['bg'])
ax.set_facecolor(C['bg'])
ax.axis('off')

# Draw states as circles
state_pos = {'Well': (0.2, 0.5), 'CTRCD': (0.55, 0.5), 'Dead': (0.85, 0.5)}
state_colors = {'Well': '#2D6A4F', 'CTRCD': '#D94F3D', 'Dead': '#6B7280'}

for state, (x, y) in state_pos.items():
    circle = mpatches.Circle((x, y), 0.10, color=state_colors[state],
                              alpha=0.85, transform=ax.transAxes, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, state, transform=ax.transAxes,
            ha='center', va='center', fontsize=13,
            fontweight='bold', color='white', zorder=4)

# Arrows
arrow_kwargs = dict(transform=ax.transAxes, arrowstyle='->', color='#333333',
                    lw=1.8, zorder=2,
                    connectionstyle='arc3,rad=0.0')

def draw_arrow(ax, x1, y1, x2, y2, label, dy=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8))
    mx, my = (x1+x2)/2, (y1+y2)/2 + dy
    ax.text(mx, my, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', fc=C['bg'], ec='none'))

draw_arrow(ax, 0.30, 0.52, 0.45, 0.52, 'p_ctrcd(t)', dy=0.10)
draw_arrow(ax, 0.30, 0.48, 0.82, 0.38, 'p_bg_mort',  dy=-0.10)
draw_arrow(ax, 0.65, 0.48, 0.82, 0.42, 'p_ctrcd_mort', dy=-0.08)

# Self-loops (stay)
for state, (x, y) in state_pos.items():
    if state != 'Dead':
        ax.annotate('stay', xy=(x, y+0.10), xytext=(x, y+0.22),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2,
                                   connectionstyle='arc3,rad=0.5'),
                    fontsize=8, color='#888888', ha='center')

# Surveillance schedule boxes
box_y = 0.12
schedules_display = {
    'Low risk':  '↓ Reduced: baseline, 1yr, 2yr',
    'Med risk':  '→ Standard: baseline + q6mo × 2yr + q1yr',
    'High risk': '↑ Intensified: baseline + q6mo × 4yr',
}
colors_box = [C['low'], C['med'], C['high']]
for i, (label, sched) in enumerate(schedules_display.items()):
    bx = 0.05 + i * 0.32
    rect = mpatches.FancyBboxPatch((bx, 0.02), 0.28, 0.18,
                                    boxstyle='round,pad=0.01',
                                    facecolor=colors_box[i], alpha=0.15,
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(bx + 0.14, 0.14, label, transform=ax.transAxes,
            ha='center', fontsize=10, fontweight='bold',
            color=colors_box[i])
    ax.text(bx + 0.14, 0.07, sched, transform=ax.transAxes,
            ha='center', fontsize=7.5, color='#444444')

ax.set_title('Markov model structure and surveillance strategies',
             fontsize=14, fontweight='500', pad=16)

plt.tight_layout()
plt.savefig(OUT / 'fig1_schematic.png', dpi=150,
            bbox_inches='tight', facecolor=C['bg'])
print("Saved → fig1_schematic.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Base case outcomes comparison
# ════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.patch.set_facecolor(C['bg'])

metrics = [
    ('Echocardiograms\nperformed',  [echos_u,  echos_s],  '', False),
    ('Total cost\n(USD)',           [cost_u,   cost_s],   '$', True),
    ('CTRCDs\ndetected',            [det_u,    det_s],    '', False),
    ('Deaths\n(10-year)',           [dead_u,   dead_s],   '', False),
]

bar_colors = [C['uniform'], C['stratified']]
labels     = ['Uniform\n(guidelines)', 'Risk-\nstratified']

for ax, (title, vals, prefix, is_cost) in zip(axes, metrics):
    ax.set_facecolor(C['bg'])
    bars = ax.bar(labels, vals, color=bar_colors, width=0.5,
                  edgecolor='white', linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        label = f'${val:,.0f}' if is_cost else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                label, ha='center', va='bottom', fontsize=9)

    # Delta annotation
    delta = vals[0] - vals[1]
    sign  = '↓' if delta > 0 else '↑'
    dlabel = f'{sign} {abs(delta):.0f}' if not is_cost else f'{sign} ${abs(delta):,.0f}'
    ax.text(0.5, 0.92, dlabel, transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold',
            color=C['stratified'] if delta > 0 else C['high'])

    ax.set_title(title, fontsize=11, fontweight='500')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle('Base case outcomes: uniform vs risk-stratified surveillance\n(531 patients, 10-year horizon)',
             fontsize=13, fontweight='500', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig2_base_case.png', dpi=150,
            bbox_inches='tight', facecolor=C['bg'])
print("Saved → fig2_base_case.png")


# ════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Tornado panels (load existing + arrange)
# ════════════════════════════════════════════════════════════════════════

# Rerun sensitivity inline for clean panel figure
RANGES = {
    'echo_cost':           (600,   1200),
    'echo_sensitivity':    (0.65,  0.85),
    'cardioprotection_hr': (0.40,  0.70),
    'p_bg_mort':           (0.004, 0.010),
    'decay_rate':          (0.50,  0.90),
}
LABELS_SA = {
    'echo_cost':           'Echo cost ($)',
    'echo_sensitivity':    'Echo sensitivity',
    'cardioprotection_hr': 'Cardioprotection HR',
    'p_bg_mort':           'Background mortality',
    'decay_rate':          'Risk decay rate',
}
BASE_PARAMS = dict(echo_cost=800, echo_sensitivity=0.75,
                   cardioprotection_hr=0.50, p_bg_mort=0.006, decay_rate=0.70)

def run_scenario(params):
    pd = params['p_bg_mort']
    pm = pd * 2.0
    dr = params['decay_rate']
    cp = params['cardioprotection_hr']
    ec = params['echo_cost']
    es = params['echo_sensitivity']

    def _get_p(grp, step):
        b = P_CTRCD_BASE[grp]
        return b if step <= 4 else b * (dr ** (step - 4))

    ru, rs = {}, {}
    for grp in groups:
        nw = np.zeros(N_STEPS+1); nc_ = np.zeros(N_STEPS+1)
        nd = np.zeros(N_STEPS+1); nw2 = np.zeros(N_STEPS+1)
        nw[0] = sizes[grp]
        pm_eff = pm
        for t in range(N_STEPS):
            pc = _get_p(grp, t)
            w2c = nw[t]*pc; w2d = nw[t]*pd; c2d = nc_[t]*pm_eff
            nw[t+1]  = nw[t]-w2c-w2d
            nc_[t+1] = nc_[t]-c2d+w2c
            nd[t+1]  = nd[t]+w2d+c2d
            nw2[t+1] = w2c
        ru[grp] = (nw, nc_, nd, nw2)

        nw = np.zeros(N_STEPS+1); nc_ = np.zeros(N_STEPS+1)
        nd = np.zeros(N_STEPS+1); nw2 = np.zeros(N_STEPS+1)
        nw[0] = sizes[grp]
        if grp == 'high':
            sch = SCHEDULES['high']
            fe  = [s for s in sch if s <= 4]
            fr  = len(fe)/max(len(sch), 1)
            pm_eff = fr*(pm*cp) + (1-fr)*pm
        else:
            pm_eff = pm
        for t in range(N_STEPS):
            pc = _get_p(grp, t)
            w2c = nw[t]*pc; w2d = nw[t]*pd; c2d = nc_[t]*pm_eff
            nw[t+1]  = nw[t]-w2c-w2d
            nc_[t+1] = nc_[t]-c2d+w2c
            nd[t+1]  = nd[t]+w2d+c2d
            nw2[t+1] = w2c
        rs[grp] = (nw, nc_, nd, nw2)

    eu = sum(ru[g][0][s]+ru[g][1][s] for g in groups for s in UNIFORM_SCHEDULE)
    es_ = sum(rs[g][0][s]+rs[g][1][s] for g in groups for s in SCHEDULES[g])

    def _det(res, scheds):
        d = 0
        for g in groups:
            sc = sorted(scheds[g])
            for t, n in enumerate(res[g][3]):
                if n > 0:
                    fe = [s for s in sc if s >= t]
                    if fe: d += n*(1-(1-es)**len(fe))
        return d

    du = _det(ru, uniform_schedules)
    ds = _det(rs, SCHEDULES)
    ddu = sum(ru[g][2][-1] for g in groups)
    dds = sum(rs[g][2][-1] for g in groups)

    return {
        'cost_saved':       (eu-es_)*ec,
        'deaths_prevented': ddu-dds,
        'ctrcd_missed':     du-ds,
    }

base_out = run_scenario(BASE_PARAMS)
sa_res = {}
for param, (lo, hi) in RANGES.items():
    sa_res[param] = {
        'low':  run_scenario({**BASE_PARAMS, param: lo}),
        'high': run_scenario({**BASE_PARAMS, param: hi}),
    }

def tornado_panel(ax, outcome_key, xlabel, title):
    base_val = base_out[outcome_key]
    bars = sorted([{
        'label': LABELS_SA[p],
        'low':   sa_res[p]['low'][outcome_key],
        'high':  sa_res[p]['high'][outcome_key],
        'range': abs(sa_res[p]['high'][outcome_key] - sa_res[p]['low'][outcome_key]),
    } for p in RANGES], key=lambda x: x['range'])

    for i, bar in enumerate(bars):
        ax.barh(i, bar['low']  - base_val, left=base_val,
                color='#2E86AB', alpha=0.85, height=0.5)
        ax.barh(i, bar['high'] - base_val, left=base_val,
                color='#E84855', alpha=0.85, height=0.5)
        ax.text(min(bar['low'], bar['high']) - abs(base_val)*0.01,
                i, f"{bar['low']:.1f}", va='center', ha='right', fontsize=7)
        ax.text(max(bar['low'], bar['high']) + abs(base_val)*0.01,
                i, f"{bar['high']:.1f}", va='center', ha='left', fontsize=7)

    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b['label'] for b in bars], fontsize=8)
    ax.axvline(base_val, color='#333', lw=1.2, ls='--')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='500')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor(C['bg'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(C['bg'])
tornado_panel(axes[0], 'cost_saved',
              'Cost saved (USD)', 'A. Cost savings')
tornado_panel(axes[1], 'deaths_prevented',
              'Deaths prevented', 'B. Deaths prevented')
tornado_panel(axes[2], 'ctrcd_missed',
              'CTRCDs missed',    'C. CTRCDs missed')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', alpha=0.85, label='Low parameter value'),
    Patch(facecolor='#E84855', alpha=0.85, label='High parameter value'),
]
axes[2].legend(handles=legend_elements, fontsize=8, loc='lower right')
fig.suptitle('One-way sensitivity analysis: tornado diagrams',
             fontsize=13, fontweight='500', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig3_tornado.png', dpi=150,
            bbox_inches='tight', facecolor=C['bg'])
print("Saved → fig3_tornado.png")

# ════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Cumulative CTRCDs DETECTED over time by strategy
# ════════════════════════════════════════════════════════════════════════

time_years = np.arange(N_STEPS + 1) * 0.5
grp_labels = {'low': 'Low risk', 'med': 'Intermediate risk', 'high': 'High risk'}


def cumulative_detected_over_time(results, schedules, sensitivity, grp):
    _, _, _, new_ctrcd = results[grp]
    sched = sorted(schedules[grp])
    detected_at = np.zeros(N_STEPS + 1)

    for t, n in enumerate(new_ctrcd):
        if n > 0:
            future_echos = [s for s in sched if s >= t]
            for i, echo_step in enumerate(future_echos):
                # Probability detected at THIS echo (not earlier ones)
                p_detected_here = (
                    (1 - sensitivity) ** i * sensitivity
                )
                if echo_step <= N_STEPS:
                    detected_at[echo_step] += n * p_detected_here

    return np.cumsum(detected_at) / sizes[grp] * 100


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
fig.patch.set_facecolor(C['bg'])

for ax, grp in zip(axes, groups):
    ax.set_facecolor(C['bg'])

    cum_u = cumulative_detected_over_time(
        res_u, uniform_schedules, ECHO_SENS, grp)
    cum_s = cumulative_detected_over_time(
        res_s, SCHEDULES, ECHO_SENS, grp)

    ax.plot(time_years, cum_u, color=C['uniform'],
            linewidth=2.5, label='Uniform', linestyle='--')
    ax.plot(time_years, cum_s, color=C[grp],
            linewidth=2.5, label='Risk-stratified')

    # Mark echo timepoints as tick marks on x axis
    for step in UNIFORM_SCHEDULE:
        ax.axvline(step * 0.5, color=C['uniform'],
                   alpha=0.15, linewidth=1, linestyle=':')
    for step in SCHEDULES[grp]:
        ax.axvline(step * 0.5, color=C[grp],
                   alpha=0.20, linewidth=1, linestyle=':')

    # Shade the gap between strategies
    ax.fill_between(time_years, cum_u, cum_s,
                    alpha=0.12, color=C[grp])

    ax.set_xlabel('Years', fontsize=10)
    ax.set_ylabel('Cumulative CTRCDs detected (%)', fontsize=10)
    ax.set_title(grp_labels[grp], fontsize=11,
                 fontweight='bold', color=C[grp])
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, 10)

fig.suptitle('Cumulative CTRCDs detected over 10 years by risk group\n'
             'Shaded area = detection difference between strategies',
             fontsize=13, fontweight='500', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig4_timecourse.png', dpi=150,
            bbox_inches='tight', facecolor=C['bg'])
print("Saved → fig4_timecourse.png")

print("\n=== All figures saved to results/figures/ ===")