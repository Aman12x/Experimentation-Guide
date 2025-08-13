# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, chi2_contingency, shapiro, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.set_page_config(layout="wide", page_title="A/B Testing Power Tool")

st.markdown("""
# 🧪 An Exploration Tool for A/B Testing 
Welcome to your one-stop dashboard for experiment design, validation, and inference.
Use the sidebar to navigate between modules. Upload your own data or use a sample.
""")

st.sidebar.markdown("## 🧭 Navigation")
tab = st.sidebar.selectbox(
    "Go to section",
    [
        "📤 Upload & Preview",
        "📊 Sample Size & Power",
        "🧪 A/B Testing",
        "🔍 Segmented Testing",
        "📈 Uplift Modeling",
        "🕰️ Trend Check",
        "⚖️ Multiple Corrections",
        "📚 Things You Should Know",
    ],
)

# DataFrame placeholder
df = st.session_state.get("df", None)

# Upload Section
if tab == "📤 Upload & Preview":
    st.header("📤 Upload & Preview")
    uploaded_file = st.file_uploader("Upload a CSV", type="csv")
    use_sample = st.checkbox("Use built-in sample dataset")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")
    elif use_sample:
        df = pd.read_csv("sample_ab_test_dataset.csv")
        st.session_state.df = df
        st.info("Using sample dataset.")

    if df is not None:
        st.markdown("### 👁️ Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

# Power Analysis
if tab == "📊 Sample Size & Power":
    st.header("📊 Sample Size Calculator")
    p1 = st.slider("Baseline Conversion Rate", 0.01, 0.5, 0.1)
    mde = st.slider("Minimum Detectable Effect", 0.01, 0.3, 0.05)
    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05)
    power = st.slider("Statistical Power (1 - β)", 0.7, 0.99, 0.8)

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = np.sqrt(2 * p1 * (1 - p1))
    sample = ((z_alpha + z_beta) * pooled / mde) ** 2

    col1, col2 = st.columns(2)
    col1.metric("📊 Sample Size per Group", f"{int(np.ceil(sample))}")
    col2.metric("🧠 Power", f"{int(power * 100)}%")

# SRM
if tab == "🧪 A/B Testing" and df is not None:
    st.header("🧪 A/B Test")
    from scipy.stats import ttest_ind, mannwhitneyu

    variant_col = "variant"
    metric_col = "metric"
    var = df[variant_col].unique()

    if len(var) == 2:
        data1 = df[df[variant_col] == var[0]][metric_col]
        data2 = df[df[variant_col] == var[1]][metric_col]

        # Distribution
        st.write("### Distribution of Metrics")
        fig, ax = plt.subplots()
        sns.histplot(data1, color="blue", kde=True, label=f"Variant {var[0]}", ax=ax)
        sns.histplot(data2, color="green", kde=True, label=f"Variant {var[1]}", ax=ax)
        ax.legend()
        st.pyplot(fig)

        # Test selection
        p1, p2 = shapiro(data1)[1], shapiro(data2)[1]
        if p1 > 0.05 and p2 > 0.05:
            stat, p = ttest_ind(data1, data2, equal_var=False)
            st.write(f"T-test p-value: {p:.4f}")
        else:
            stat, p = mannwhitneyu(data1, data2)
            st.write(f"Mann-Whitney U p-value: {p:.4f}")

# Segmented A/B Test
if tab == "🔍 Segmented Testing" and df is not None:
    st.header("🔍 Segmented A/B Testing")
    segment_col = st.selectbox(
        "Segment by", [col for col in df.columns if col not in ["variant", "metric"]]
    )
    segments = df[segment_col].unique()
    for s in segments:
        st.markdown(f"#### Segment: {s}")
        sub = df[df[segment_col] == s]
        v = sub["variant"].unique()
        if len(v) != 2:
            st.warning("Need 2 variants.")
            continue
        g1, g2 = (
            sub[sub["variant"] == v[0]]["metric"],
            sub[sub["variant"] == v[1]]["metric"],
        )
        stat, p = ttest_ind(g1, g2)
        st.write(f"p-value: {p:.4f}")

# Uplift Modeling
if tab == "📈 Uplift Modeling" and df is not None:
    st.header("📈 Uplift Modeling")
    features = st.multiselect(
        "Choose features",
        [col for col in df.columns if col not in ["variant", "metric"]],
    )
    if features:
        df["treatment"] = (df["variant"] == df["variant"].unique()[1]).astype(int)
        X = pd.get_dummies(df[features], drop_first=True)
        y = df["metric"]
        model_t = RandomForestClassifier().fit(
            X[df["treatment"] == 1], y[df["treatment"] == 1]
        )
        model_c = clone(model_t).fit(X[df["treatment"] == 0], y[df["treatment"] == 0])
        uplift = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
        st.line_chart(uplift)

# Pre/Post Trend
if tab == "🕰️ Trend Check" and df is not None:
    st.header("🕰️ Pre/Post Trend Analysis")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby(["date", "variant"])["metric"].mean().unstack()
        st.line_chart(daily)
    else:
        st.warning("Missing 'date' column")

# Multiple Testing Correction
if tab == "⚖️ Multiple Corrections":
    st.header("⚖️ Multiple Testing Corrections")
    pvals = {"Metric A": 0.03, "Metric B": 0.04, "Metric C": 0.06}
    df_p = pd.DataFrame(pvals.items(), columns=["metric", "p_value"])
    method = st.selectbox("Correction Method", ["Bonferroni", "Benjamini-Hochberg"])
    if method == "Bonferroni":
        df_p["adj_p"] = df_p["p_value"] * len(df_p)
    else:
        df_p = df_p.sort_values("p_value").reset_index(drop=True)
        df_p["rank"] = df_p.index + 1
        df_p["adj_p"] = df_p["p_value"] * len(df_p) / df_p["rank"]
    df_p["significant"] = df_p["adj_p"] < 0.05
    st.write(df_p)

# Intro to A/B Testing
if tab == "📚 Things You Should Know":
    st.header("📚 A/B Testing Beginners Guide")
    st.markdown(r"""
### 
---

#### 🧪 What is A/B Testing?

A/B testing is a **controlled experiment** where users are randomly split into two groups:

- **Control Group (A)**: Sees the current version.
- **Treatment Group (B)**: Sees a new version with a change (e.g., new button, layout, algorithm).

The goal is to **measure the impact of the change** on a specific metric (e.g., click-through rate, purchase rate).

---

#### 🔄 Randomization

Randomly assigning users ensures that both groups are statistically equivalent. This avoids biases from external factors (like device, time of day, user type).

✅ **Why it matters**: Any difference in outcome can be attributed to the change you're testing — not something else.

---

#### 🎯 Hypothesis Testing: Step-by-Step

1. **Define Hypotheses**
   - **Null Hypothesis (H₀)**: There is no difference between control and treatment.
   - **Alternative Hypothesis (H₁)**: There is a statistically significant difference.

2. **Choose a Metric**  
   Example: Conversion Rate  
   Conversion Rate = Number of Conversions / Total Users

3. **Set Parameters**
   - Significance Level (α): typically 0.05
   - Statistical Power (1 - β): usually 0.8
   - Minimum Detectable Effect (MDE): smallest effect worth detecting (e.g., 2%)

4. **Calculate Sample Size**
    Use the formula:
    n = (Zα/2 + Zβ)^2 * (2 * p * (1-p)) / MDE^2,

   
    Where:
   - \( p \): Baseline conversion rate
   - \( MDE \): Minimum lift we want to detect
   - \( Z \): Z-scores from normal distribution
   - \(Zα/2\): is the Z-score corresponding to the desired significance level (α), typically 0.05 for 95% confidence.
   - \(Zβ\): is the Z-score corresponding to the desired power (1 - β), typically 0.84 for 80% power. 

You can use the sample size calculator in this app to automate this step.

---

#### 🧮 Example: Testing Add-to-Cart Button on Amazon

- **Metric**: Add-to-Cart Conversion Rate  
- **Baseline Rate**: 10%  
- **MDE**: 2% (want to detect increase to 12%)  
- **Significance Level**: 0.05  
- **Power**: 0.8

💡 Using the formula, you’d need ~3,900 users per group.

---

#### 📐 Confidence Interval (CI)

CI gives a range of likely values for the difference in performance.

Example:
- If 95% CI = [0.01, 0.05], then we are 95% confident that the treatment group is **1% to 5% better**.

---

#### 📊 What Is a p-value?

- **p < 0.05** → Statistically significant → Reject H₀
- **p ≥ 0.05** → Not enough evidence → Do not reject H₀

But don’t rely only on p-values — always report **effect size** and **CI**.

---
                
#### 📊 Power Analysis Notes
- **Statistical Power** is the probability of detecting a true effect (1 - β). Commonly set to 80% or 90%.
- **Low power** increases the risk of Type II errors (false negatives).
- **Factors affecting power**: sample size, MDE, baseline rate, and significance level (α).
- **Use power analysis** before running experiments to avoid underpowered tests that yield inconclusive results.
- **Tools**: Use libraries like `statsmodels` in Python or online calculators.

---
#### 🧪 Real Case Study: Google Link Color Test

Google tested **shades of blue** for search result links to increase clicks.

- **Control**: Old blue
- **Treatment**: Slightly brighter blue
- **Result**: A 0.3% increase in click-through rate  
- **Impact**: ~$200M/year in additional revenue

✅ Lesson: Small changes can have massive business impact at scale.

---

#### 📊 Summary Table

| Concept         | Explanation |
|----------------|-------------|
| Control Group   | No changes applied |
| Treatment Group | New change being tested |
| Randomization   | Ensures fairness |
| Hypotheses      | H₀: No effect; H₁: Some effect |
| Sample Size     | Number of users per group |
| MDE             | Minimum change you want to detect |
| p-value         | Probability the result is due to chance |
| CI              | Range where true difference likely falls |
| Effect Size     | How big the change is |
| SRM             | Are group sizes balanced? |
| Uplift Modeling | Which users benefit most |
| Segmentation    | Run tests by user type |
| Trend Checking  | Were groups stable pre-test? |

---

### 🏁 Final Thoughts

A/B testing allows companies like **Amazon**, **Meta**, and **Google** to make **data-driven product decisions** every day.

> If you understand how to formulate a hypothesis, compute sample size, assign users randomly, and interpret the results — you already know more than most!

""")
