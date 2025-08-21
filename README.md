https://experimentation-guide.streamlit.app

# A/B Testing Power Tool

A beginner-friendly, fully interactive **Streamlit** app for planning, running, and interpreting A/B tests. It covers **sample size & power**, **SRM checks**, **frequentist tests**, **segmented analysis**, **uplift modeling**, **trend checks**, **multiple testing corrections**, and a concise **education hub** that explains every concept used in the app.

---

## Features

- **Upload & Preview** — Upload CSV or use a sample dataset; quick preview table.
- **Sample Size & Power** — Compute users per group from baseline rate, MDE, α, and power.
- **A/B Testing** — Auto-selects the right test (Welch t-test vs. Mann–Whitney U) based on normality; visualizes distributions; reports p-values.
- **Segmented Testing** — Per-segment lifts (e.g., device, region, cohort).
- **Uplift Modeling (T-Learner)** — Two Random Forests (treatment vs. control) → individual uplift scores.
- **Pre/Post Trends** — Daily metric lines to check parallel trends visually.
- **Multiple Testing** — Bonferroni & Benjamini–Hochberg (FDR).
- **Education Hub** — Clear explanations of control/treatment, randomization, MDE, power, CIs, effect sizes, SRM, and more.

---

## Quickstart

```bash
# optional: create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install
pip install -r requirements.txt

# run
streamlit run app.py
```

Python 3.9+ recommended.

---

## Project Structure

```bash
.
├── app.py                      # Main Streamlit app
├── sample_ab_test_dataset.csv  # Optional sample data
├── requirements.txt            # Dependencies
└── README.md
```

Example `requirements.txt`:

```txt
streamlit>=1.32
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
```

---

## Data Format (CSV)

Minimum required columns:

- `variant` (string): e.g., A or B  
- `metric` (number): outcome variable (binary 0/1 or continuous)

Optional but useful:

- `date` (YYYY-MM-DD): for trend charts  
- other features (e.g., `device`, `region`) for segmentation/uplift

**Example**

```csv
variant,metric,date,device,region
A,1,2025-07-01,mobile,US
B,0,2025-07-01,desktop,US
A,1,2025-07-02,desktop,EU
B,1,2025-07-02,mobile,EU
```

---

## Experimentation Crash Course (Beginner-Friendly)

### Core Roles & Randomization
- **Control group (A)**: current version (baseline).  
- **Treatment group (B)**: new version (change under test).  
- **Randomization** ensures groups are comparable so differences can be attributed to the change (not confounders).

### Key Metric
- **Conversion Rate** = `number_of_conversions / total_users`

### Hypothesis Setup
- **H₀ (Null)**: no difference between A and B.  
- **H₁ (Alt.)**: a difference exists (two-sided) or B > A (one-sided).  
- Choose **α** (e.g., 0.05) and **Power** = 1 − β (e.g., 0.80).

### Minimum Detectable Effect (MDE)
The smallest **absolute** lift worth detecting. Example: baseline p = 0.10, MDE = 0.02 means you care to detect an increase to 0.12.

### Sample Size (per group, two-sided, proportions)
Plain-text formula (no LaTeX):

```
n = ((Z_{α/2} + Z_{β})^2 * 2 * p * (1 - p)) / MDE^2
```

Where:
- `p` = baseline conversion rate (e.g., 0.10)  
- `MDE` = absolute lift to detect (e.g., 0.02)  
- `Z_{α/2}` ≈ 1.96 for α = 0.05  
- `Z_{β}` ≈ 0.84 for 80% power  

Use the app’s **Sample Size & Power** tab to compute this interactively.

### SRM (Sample Ratio Mismatch)
If your allocation (e.g., 50/50) is violated in observed counts, run a **chi-square** goodness-of-fit test. SRM indicates assignment or tracking issues → **invalidate** the test until fixed.

### Normality Check → Pick the Test
- **Shapiro–Wilk** checks normality of your metric distribution.  
- If (approx.) normal & continuous → **Welch’s t-test**.  
- If not normal / ordinal → **Mann–Whitney U**.  
- (For large-n binary conversions, the **two-proportion z-test** is common.)

**Two-proportion Z (reference)**

```
Z = (p_B − p_A) / sqrt( p * (1 − p) * (1/n_A + 1/n_B) )   where p is pooled rate
```

### Effect Size & Confidence Interval
- **Effect size**: magnitude (e.g., Cohen’s d for means, Cliff’s delta non-parametric).  
- **95% CI** around the difference tells you the plausible range of the true effect.  
- Statistical significance ≠ practical significance → always check effect size & CI.

### Multiple Testing
Testing many metrics/segments inflates false positives.  
- **Bonferroni** (strict) and **Benjamini–Hochberg (FDR)** (balanced) supported in the app.

### Segmented Testing
View results by device/region/cohort to uncover **heterogeneous effects**. Use multiple-testing corrections.

### Uplift Modeling (T-Learner)
- Train a model on **treatment** and one on **control**.  
- `uplift(x) = P(y=1 | x, treatment) − P(y=1 | x, control)` → prioritize users most likely to benefit.

### Trend Checks
Plot pre/post daily metrics for A and B. Parallel pre-trends support cleaner causal comparisons.

---

## Product Case Study (Worked Example)

**Scenario**: You manage a product page and want to increase **Add-to-Cart** conversions by changing the CTA copy.

- Control (A): “Add to Cart”  
- Treatment (B): “Get 20% Off — Add to Cart”  
- Metric: Conversion Rate (Add-to-Cart / Users)  
- Baseline (p): 10%  
- MDE: 2% (detect 0.12 vs 0.10)  
- α = 0.05 → Z_{α/2} ≈ 1.96  
- Power = 0.80 → Z_{β} ≈ 0.84  

**Sample Size per Group**

```
n = ((1.96 + 0.84)^2 * 2 * 0.10 * 0.90) / 0.02^2
  ≈ (7.84 * 0.18) / 0.0004
  ≈ 1.4112 / 0.0004
  ≈ 3,528 users per group (approx.)
```

**Observed Results (after running the test)**

```
A: 5,000 users, 450 conversions → 9.0%
B: 5,000 users, 550 conversions → 11.0%
Absolute lift = 2.0%
Relative lift = (11 − 9) / 9 = 22.2%
```

**Interpretation Checklist**

1. **SRM**: A and B both 5,000 → OK.  
2. **Test**: With large binary samples, a **two-proportion z-test** is appropriate.  
3. **p-value**: Suppose p = 0.012 → significant at 0.05.  
4. **Effect size & CI**: Report the 95% CI for the difference; ensure it excludes 0 and is practically meaningful.  
5. **Decision**: The treatment increases conversions with meaningful magnitude → **roll out** or **ramp**.

---

## When Things Go Wrong (Fast FAQ)

- **SRM detected** → pause and audit assignment/logging.  
- **Underpowered test** → extend duration or accept larger MDE (or higher α).  
- **Significant but tiny lift** → check confidence intervals and business impact.  
- **Segment contradictions** → adjust for multiple testing; ensure per-segment sample size.

---


## Contributing

Issues and PRs welcome. Ideas: CUPED variance reduction, Bayesian modules, bandits, richer report exports.
