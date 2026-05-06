import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Page setup
# ==========================================================
st.set_page_config(
    page_title="Uncertainty Calculator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CSS (MATCHED TO YOUR ESTIMATOR)
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --font:   "Avenir Next","Avenir","Nunito","Century Gothic",sans-serif;
  --mono:   "IBM Plex Mono",monospace;
  --navy:   #012169;
  --basf:   #0050a0;
  --basflt: #0071c5;
  --gold:   #b5893a;
  --body:   #2c3e55;
  --label:  #4a5e78;
  --muted:  #6b7c94;
  --border: #dce3ec;
}

html, body, [class*="css"]  {
    font-family: var(--font);
    color: var(--body);
}

h1, h2, h3 {
    color: var(--navy);
    font-weight: 600;
}

.section {
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.label {
    color: var(--label);
    font-size: 0.9rem;
    font-weight: 500;
}

.muted {
    color: var(--muted);
    font-size: 0.85rem;
}

.metric {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--basf);
}

.box {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Title
# ==========================================================
st.markdown("<h1>Uncertainty Calculator</h1>", unsafe_allow_html=True)
st.markdown('<div class="muted">Pedigree-based uncertainty with Monte Carlo simulation</div>', unsafe_allow_html=True)

# ==========================================================
# Assumptions (Styled, readable)
# ==========================================================
with st.expander("Model assumptions & equations", expanded=False):
    st.markdown("""
**Distribution**  
Lognormal: ln(X) ~ Normal(μ, σ²)

**Variance from pedigree**  
σ² = Σ (ln fᵢ)²

**Mean transformation**  
μ = ln(m) − ½σ²

**Monte Carlo**  
X = exp(μ + σZ),  Z ~ N(0,1)

**Settings**  
- 10,000 simulations  
- 95% CI (2.5–97.5 percentile)
""")

# ==========================================================
# Input Section
# ==========================================================
st.markdown('<div class="section"><h3>Input</h3></div>', unsafe_allow_html=True)

input_value = st.number_input("Value", value=100.0)

# ==========================================================
# Pedigree
# ==========================================================
st.markdown('<div class="section"><h3>Data Quality (Pedigree)</h3></div>', unsafe_allow_html=True)

PEDIGREE_OPTIONS = {
    "Reliability": {
        "Verified data based on measurements": 1,
        "Verified data partly based on assumptions": 2,
        "Non-verified data partly based on measurements": 3,
        "Qualified estimate": 4,
        "Non-qualified estimate": 5,
    },
    "Completeness": {
        "Representative data from all sites": 1,
        "Representative data from most sites": 2,
        "Representative data from some sites": 3,
        "Limited data": 4,
        "Very limited data": 5,
    },
    "Temporal": {
        "Less than 3 years difference": 1,
        "3–6 years": 2,
        "6–10 years": 3,
        "10–15 years": 4,
        "More than 15 years": 5,
    },
    "Geographical": {
        "Same region": 1,
        "Similar region": 2,
        "Different region with adjustments": 3,
        "Different region": 4,
        "Unknown region": 5,
    },
    "Technological": {
        "Same technology": 1,
        "Similar technology": 2,
        "Different technology with adjustments": 3,
        "Different technology": 4,
        "Unknown technology": 5,
    }
}

PEDIGREE_FACTORS = {
    "Reliability": [1.0, 1.05, 1.1, 1.2, 1.5],
    "Completeness": [1.0, 1.02, 1.05, 1.1, 1.2],
    "Temporal": [1.0, 1.03, 1.08, 1.15, 1.25],
    "Geographical": [1.0, 1.02, 1.06, 1.1, 1.2],
    "Technological": [1.0, 1.02, 1.05, 1.1, 1.2],
}

scores = {}

cols = st.columns(2)
i = 0

for category, options in PEDIGREE_OPTIONS.items():
    with cols[i % 2]:
        choice = st.selectbox(category, list(options.keys()))
        scores[category] = options[choice]
    i += 1

# ==========================================================
# Sigma Calculation
# ==========================================================
def calculate_sigma(scores):
    variance = 0
    for key, score in scores.items():
        factor = PEDIGREE_FACTORS[key][score - 1]
        variance += (np.log(factor))**2
    return np.sqrt(variance)

sigma = calculate_sigma(scores)
gsd = np.exp(sigma)

# Display nicely
st.markdown('<div class="section"><h3>Derived Uncertainty</h3></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="label">σ (log-space)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">{sigma:.4f}</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="label">Geometric SD (GSD)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">{gsd:.4f}</div>', unsafe_allow_html=True)

# ==========================================================
# Monte Carlo
# ==========================================================
N_RUNS = 10000

def monte_carlo(mean, sigma, n):
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma, n)

# ==========================================================
# Run Simulation
# ==========================================================
st.markdown('<div class="section"></div>', unsafe_allow_html=True)

if st.button("Run Monte Carlo"):
    samples = monte_carlo(input_value, sigma, N_RUNS)

    mean_val = np.mean(samples)
    std_dev = np.std(samples)
    lower = np.percentile(samples, 2.5)
    upper = np.percentile(samples, 97.5)

    st.markdown('<div class="section"><h3>Results</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="label">Mean</div><div class="metric">{mean_val:.2f}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="label">Std Dev</div><div class="metric">{std_dev:.2f}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="label">95% CI</div><div class="metric">[{lower:.1f}, {upper:.1f}]</div>', unsafe_allow_html=True)

    # Plot
    fig, ax = plt.subplots()
    ax.hist(samples, bins=50)
    ax.set_title("Distribution")
    st.pyplot(fig)

    st.markdown('<div class="muted">10,000 Monte Carlo simulations (percentile-based CI)</div>', unsafe_allow_html=True)