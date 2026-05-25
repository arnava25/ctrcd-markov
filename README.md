# CTRCD Markov Surveillance Model

Decision-analytic Markov cohort model comparing uniform vs risk-stratified 
echocardiographic surveillance for cancer therapy-related cardiac dysfunction 
(CTRCD) in HER2-positive breast cancer.

## Overview

This model evaluates the clinical and economic impact of surveillance strategies 
guided by a validated Cox prediction model, across a 531-patient cohort over a 
10-year horizon.

## Structure

- `scripts/01_markov_model.py` — Core Markov simulation
- `scripts/02_sensitivity_analysis.py` — One-way sensitivity analysis and tornado diagrams
- `scripts/03_figures.py` — Publication figures

## Key findings

- 347 fewer echocardiograms under risk-stratified surveillance
- $277,357 cost savings over 10 years
- 2.6 deaths prevented
- Fewer than 2 CTRCDs missed
- Findings robust across all sensitivity analyses

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/01_markov_model.py
python scripts/02_sensitivity_analysis.py
python scripts/03_figures.py
```

## Citation

Amit A. Clinical Impact of Risk-Stratified Echocardiographic Surveillance 
for Cardiotoxicity in HER2-Positive Breast Cancer: A Decision-Analytic 
Markov Model. [Under preparation]

## Data

The BC_cardiotox dataset used to derive transition probabilities is publicly 
available at Figshare:
Minchole A, et al. BC_cardiotox: a cardiotoxicity dataset for breast cancer 
patients. Sci Data 2023;10:542.
DOI: 10.6084/m9.figshare.22650748

Download the dataset and place the CSV files in the `data/` directory before 
running the scripts.
