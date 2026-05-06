import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO
import base64, os

# ==========================================================
# Page setup
# ==========================================================
st.set_page_config(
    page_title="Microplastic Accumulation & Impact Estimator (Beta Version)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CSS
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --font:   "Avenir Next","Avenir","Nunito","Century Gothic",sans-serif;
  --mono:   "Avenir Next","Avenir","IBM Plex Mono",monospace;
  --navy:   #012169;
  --basf:   #0050a0;
  --basflt: #0071c5;
  --gold:   #b5893a;
  --body:   #2c3e55;
  --label:  #4a5e78;
  --muted:  #6b7c94;
  --border: #dce3ec;
  --brd2:   #b8c4d4;
  --bg2:    #f7f8fa;
  --hi:     #e8f0fb;
  --green:  #1a8c5c;
}

/* Force light always */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
  color-scheme: light only !important;
  background: #ffffff !important;
  color: #2c3e55 !important;
}
@media (prefers-color-scheme: dark) {
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="block-container"] {
    background: #ffffff !important;
    color: #2c3e55 !important;
  }
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
[data-testid="block-container"] {
  padding: 0 2.5rem 4rem !important;
  max-width: 1300px !important;
}

/* Base type */
* { font-family: var(--font) !important; }
h1,h2,h3,h4 { color: var(--navy) !important; font-weight: 700 !important; }
p, li, label, div, span { color: var(--body) !important; }

/* ── Header ───────────────────────────────────────────────── */
.hdr {
  background: linear-gradient(135deg, #012169 0%, #1a3a6e 55%, #0050a0 100%);
  padding: 1.5rem 2.8rem;
  margin: 0 -2.5rem 2.8rem;
  display: flex; align-items: center; justify-content: space-between;
  position: relative; overflow: hidden;
  box-shadow: 0 4px 24px rgba(1,33,105,0.18);
}
.hdr::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg,transparent,#b5893a 30%,#0071c5 70%,transparent);
}
.hdr-logos { display:flex; align-items:center; gap:1.4rem; z-index:1; }
.hdr-div   { width:1px; height:34px; background:rgba(255,255,255,0.35); }
.hdr-title { text-align:center; flex:1; z-index:1; }
.hdr-t1 {
  font-size:1.55rem !important; font-weight:400 !important;
  font-family:"Avenir Next","Avenir","Nunito","Century Gothic",sans-serif !important;
  color:#ffffff !important; letter-spacing:0.01em;
  margin:0; line-height:1.25; display:block;
}
.hdr-t2 {
  font-size:0.68rem !important; font-weight:500 !important;
  color:rgba(255,255,255,0.7) !important;
  letter-spacing:0.2em; text-transform:uppercase;
  margin-top:0.35rem; display:block;
}

/* ── Info strip ───────────────────────────────────────────── */
.info-strip {
  background: var(--hi);
  border-left: 3px solid var(--basflt);
  border-radius: 0 6px 6px 0;
  padding: 0.9rem 1.3rem; margin-bottom: 1.8rem;
}
.info-strip p { font-size:0.85rem !important; color:var(--body) !important; line-height:1.65 !important; margin:0 !important; }

/* ── Section headings ─────────────────────────────────────── */
.sec { display:flex; align-items:center; gap:0.9rem; margin:2.4rem 0 1.2rem; }
.sec .num { font-size:0.65rem; color:var(--gold); letter-spacing:0.14em; font-weight:700; min-width:2rem; }
.sec h2   { font-size:1.05rem !important; color:var(--navy) !important; font-weight:700 !important; margin:0 !important; }
.sec .rule { flex:1; height:1px; background:linear-gradient(90deg,var(--brd2),transparent); }

/* ── Metric cards ─────────────────────────────────────────── */
.mc {
  background:#fff; border:1px solid var(--border);
  border-top:3px solid var(--navy); border-radius:8px;
  padding:1.4rem 1.6rem 1.3rem;
  box-shadow:0 2px 12px rgba(1,33,105,0.07);
}
.mc .lbl { font-size:0.62rem; color:var(--gold) !important; letter-spacing:0.18em; text-transform:uppercase; font-weight:700; margin-bottom:0.6rem; display:block; }
.mc .ttl { font-size:0.92rem; color:var(--navy) !important; font-weight:700; margin-bottom:1rem; padding-bottom:0.7rem; border-bottom:1px solid var(--border); display:block; }
.mi      { display:flex; justify-content:space-between; align-items:baseline; padding:0.32rem 0; }
.mi .k   { font-size:0.78rem; color:var(--label) !important; }
.mi .v   { font-family:var(--mono) !important; font-size:0.84rem; color:#0d1b2e !important; font-weight:600; }
.mi .vg  { font-family:var(--mono) !important; font-size:0.88rem; color:var(--green) !important; font-weight:700; }
.mi .vb  { font-family:var(--mono) !important; font-size:0.84rem; color:var(--basflt) !important; font-weight:600; }

/* ── Inputs ───────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
  background:#fff !important; color:#0d1b2e !important;
  border:1px solid var(--brd2) !important; border-radius:5px !important;
  font-family:var(--mono) !important; font-size:0.88rem !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color:var(--basflt) !important;
  box-shadow:0 0 0 3px rgba(0,113,197,0.12) !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
  color:var(--label) !important; font-size:0.75rem !important;
  letter-spacing:0.06em !important; text-transform:uppercase !important;
  font-weight:600 !important;
}

/* ── RADIO BUTTONS ────────────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label * {
  letter-spacing: normal !important;
  text-transform: none !important;
  color: #1a1a1a !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  background: transparent !important;
}

/* ── Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
  background: var(--basflt) !important; border-color: var(--basflt) !important;
}

/* ── Run button — compact ─────────────────────────────────── */
[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #012169, #0050a0) !important;
  color: #ffffff !important;
  border: none !important; border-radius: 5px !important;
  font-size: 0.82rem !important; font-weight: 600 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important;
  padding: 0.55rem 2rem !important;
  width: auto !important;
  box-shadow: 0 4px 18px rgba(1,33,105,0.25) !important;
  transition: all 0.22s ease !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(1,33,105,0.38) !important;
}
[data-testid="stButton"] > button *,
[data-testid="stButton"] > button p,
[data-testid="stButton"] > button span { color: #ffffff !important; }

/* ── Advanced params box (no expander = no arrow bug) ────── */
.adv-box {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.7rem 1.2rem 0.5rem; margin-bottom: 0.8rem;
}
.adv-label {
  font-size: 0.72rem !important; font-weight: 700 !important;
  color: var(--label) !important; letter-spacing: 0.1em !important;
  text-transform: uppercase !important; display: block; margin-bottom: 0 !important;
}

/* ── Download buttons ─────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
  background:#fff !important; border:1px solid var(--brd2) !important;
  color:var(--navy) !important; font-size:0.78rem !important;
  letter-spacing:0.1em !important; text-transform:uppercase !important;
  font-weight:600 !important; padding:0.5rem 1.4rem !important;
  border-radius:4px !important; box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
  transition:all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background:var(--hi) !important; border-color:var(--basflt) !important; color:var(--basflt) !important;
}
[data-testid="stDownloadButton"] > button * { color:inherit !important; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:8px !important; overflow:hidden !important; }
[data-testid="stDataFrame"] th { background:var(--navy) !important; color:#fff !important; font-size:0.72rem !important; text-transform:uppercase !important; letter-spacing:0.08em !important; }
[data-testid="stDataFrame"] td { color:#0d1b2e !important; background:#fff !important; font-size:0.82rem !important; }
[data-testid="stDataFrame"] tr:nth-child(even) td { background:var(--bg2) !important; }

/* ── Footer ───────────────────────────────────────────────── */
.ftr { margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--border); display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.5rem; }
.ftr p { font-size:0.7rem !important; color:var(--muted) !important; font-weight:500 !important; margin:0 !important; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#f0f3f7; }
::-webkit-scrollbar-thumb { background:var(--brd2); border-radius:3px; }

/* Dropdown portal — rendered outside main DOM, needs global override */
body [data-baseweb="popover"],
body [data-baseweb="popover"] *,
body ul[role="listbox"],
body li[role="option"] {
  background: #ffffff !important;
  color: #012169 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
}
body li[role="option"]:hover,
body li[role="option"][aria-selected="true"] {
  background: #dbeafe !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# Header
# ==========================================================
st.markdown("""
<div class="hdr">
  <div class="hdr-title" style="text-align:center; width:100%;">
    <span class="hdr-t1">Microplastic Accumulation &amp; Impact Estimator (Beta Version)</span>
    <span class="hdr-t2">Proprietary Seed Coatings' Environmental Impact Prediction</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Intro strip
# ==========================================================
st.markdown("""
<div class="info-strip">
  <p>
    This tool estimates long-term accumulation and ecotoxicity of polymer seed coatings in agricultural soils,
    comparing <strong style="color:#012169;">Styrene Butadiene Rubber (SBR)</strong>,
    <strong style="color:#012169;">Polyurethane (PU)</strong>, and
    <strong style="color:#012169;">Polyethylene Oxide (PEO)</strong> across multi-year simulation horizons.
    Configure your parameters below and run the simulation to generate results and export data.
  </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Polymer constants (defaults)
# ==========================================================
sbr_cf = 1.46E-01
pu_cf  = 4.82E-06
peo_cf = 8.03E-05
k_sbr  = 0.0026
k_pu   = 25.9
k_peo  = 9.42E-04

# ==========================================================
# SECTION 1 — Simulation Settings
# ==========================================================
st.markdown('<div class="sec"><span class="num">01</span><h2>Simulation Settings</h2><div class="rule"></div></div>', unsafe_allow_html=True)
years = st.slider("Years to simulate", min_value=1, max_value=100, value=30)

# ==========================================================
# SECTION 2 — Input Methods
# ==========================================================
st.markdown('<div class="sec"><span class="num">02</span><h2>Input Methods</h2><div class="rule"></div></div>', unsafe_allow_html=True)

st.markdown("""
<style>
/* Selectbox trigger */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
  background: #ffffff !important;
  border: 1.5px solid #b8c4d4 !important;
  border-radius: 6px !important;
  color: #012169 !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stSelectbox"] > div > div:focus,
[data-testid="stSelectbox"] > div > div:active {
  border-color: #012169 !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stSelectbox"] svg { color: #012169 !important; }

/* Open dropdown — every possible container Streamlit/BaseWeb renders */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[data-baseweb="list"],
ul[role="listbox"],
div[role="listbox"] {
  background: #ffffff !important;
  border: 1px solid #b8c4d4 !important;
  border-radius: 6px !important;
  box-shadow: 0 4px 20px rgba(1,33,105,0.12) !important;
}
/* Each option */
li[role="option"],
div[role="option"],
[data-baseweb="menu-item"] {
  background: #ffffff !important;
  color: #012169 !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
[data-baseweb="menu-item"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
  background: #e8f0fb !important;
  color: #012169 !important;
}
</style>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    seed_input_method = st.selectbox(
        "Annual seed quantity definition",
        ["Area + seeds per hectare", "Total number of seeds", "Total seed mass"]
    )
with col_b:
    coating_input_method = st.selectbox(
        "Annual polymer input definition",
        ["Polymer coating as mg per g seed", "Total polymer mass applied per year"]
    )

# ==========================================================
# SECTION 3 — Required Inputs
# ==========================================================
st.markdown('<div class="sec"><span class="num">03</span><h2>Required Inputs</h2><div class="rule"></div></div>', unsafe_allow_html=True)

land = seeds_per_hectare = total_seeds = None
total_seed_mass_kg = one_seed_mass_g = None
coating_mg_per_g = total_polymer_mass_g = None

col1, col2 = st.columns(2, gap="large")
with col1:
    if seed_input_method == "Area + seeds per hectare":
        land = st.number_input("Farm area (hectares)", min_value=0.1, value=100.0, step=0.1)
        seeds_per_hectare = st.number_input("Seeds per hectare", min_value=1, value=75000, step=1000)
    elif seed_input_method == "Total number of seeds":
        total_seeds = st.number_input("Total number of seeds per year", min_value=1, value=7500000, step=1000)
    elif seed_input_method == "Total seed mass":
        total_seed_mass_kg = st.number_input("Total seed mass per year (kg)", min_value=0.001, value=2250.0, step=1.0)

with col2:
    need_one_seed_mass = (
        coating_input_method == "Polymer coating as mg per g seed"
        and seed_input_method in ["Area + seeds per hectare", "Total number of seeds"]
    )
    if need_one_seed_mass:
        one_seed_mass_g = st.number_input("Mass of one seed (g)", min_value=0.0001, value=0.3, step=0.01)
    if coating_input_method == "Polymer coating as mg per g seed":
        coating_mg_per_g = st.number_input("Polymer coating (mg per g seed)", min_value=0.0, value=20.0, step=0.1)
    else:
        total_polymer_mass_g = st.number_input("Total polymer mass applied per year (g)", min_value=0.0, value=45000.0, step=100.0)

# ==========================================================
# SECTION 4 — Advanced Parameters
# (replaced st.expander with plain HTML label + st.columns
#  to permanently eliminate the arrow overlap bug)
# ==========================================================
st.markdown('<div class="sec"><span class="num">04</span><h2>Advanced Parameters</h2><div class="rule"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="adv-box"><span class="adv-label">Polymer Degradation Constants</span></div>', unsafe_allow_html=True)

adv1, adv2, adv3 = st.columns(3, gap="medium")
with adv1:
    k_sbr = st.number_input("k — SBR (yr⁻¹)", value=0.0026, format="%.4f")
    sbr_cf = st.number_input("CF — SBR (CTUe/g)", value=float(sbr_cf), format="%e")
with adv2:
    k_pu = st.number_input("k — PU (yr⁻¹)", value=float(k_pu), format="%.4f")
    pu_cf = st.number_input("CF — PU (CTUe/g)", value=float(pu_cf), format="%e")
with adv3:
    k_peo = st.number_input("k — PEO (yr⁻¹)", value=float(k_peo), format="%.4f")
    peo_cf = st.number_input("CF — PEO (CTUe/g)", value=float(peo_cf), format="%e")

# ==========================================================
# Run button
# ==========================================================
run = st.button("▶  Run Simulation")

# ==========================================================
# Run simulation
# ==========================================================
if run:

    # Resolve total_seeds
    if seed_input_method == "Area + seeds per hectare":
        total_seeds = land * seeds_per_hectare

    # Resolve total_seed_mass_g
    if seed_input_method == "Total seed mass":
        total_seed_mass_g = total_seed_mass_kg * 1000
    elif one_seed_mass_g is not None:
        total_seed_mass_g = total_seeds * one_seed_mass_g
    else:
        total_seed_mass_g = total_seeds * 0.3 if total_seeds else 0.0

    # Resolve annual_input_g
    if coating_input_method == "Polymer coating as mg per g seed":
        annual_input_g = total_seed_mass_g * (coating_mg_per_g / 1000)
    else:
        annual_input_g = total_polymer_mass_g

    t = np.arange(0, years + 1)

    def accumulation(k, annual_input, t):
        if np.isclose(k, 0):
            return annual_input * t
        return annual_input * (1 - np.exp(-k * t)) / (1 - np.exp(-k))

    mass_sbr   = accumulation(k_sbr, annual_input_g, t)
    mass_pu    = accumulation(k_pu,  annual_input_g, t)
    mass_peo   = accumulation(k_peo, annual_input_g, t)
    impact_sbr = mass_sbr * sbr_cf
    impact_pu  = mass_pu  * pu_cf
    impact_peo = mass_peo * peo_cf

    steady_sbr = (np.inf if np.isclose(k_sbr, 0) else annual_input_g / (1 - np.exp(-k_sbr)))
    steady_pu  = (np.inf if np.isclose(k_pu,  0) else annual_input_g / (1 - np.exp(-k_pu)))
    steady_peo = (np.inf if np.isclose(k_peo, 0) else annual_input_g / (1 - np.exp(-k_peo)))

    total_impact_sbr = np.sum(impact_sbr)
    total_impact_pu  = np.sum(impact_pu)
    total_impact_peo = np.sum(impact_peo)
    reduction_mass_kg = (steady_sbr - steady_pu) / 1000
    reduction_toxicity_percent = (
        (total_impact_sbr - total_impact_pu) / total_impact_sbr * 100
        if total_impact_sbr != 0 else 0
    )

    seeds_display = f"{int(total_seeds):,}" if total_seeds is not None else "N/A"

    # ── Results summary ──────────────────────────────────────
    st.markdown('<div class="sec"><span class="num">05</span><h2>Results Summary</h2><div class="rule"></div></div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3, gap="large")
    with rc1:
        st.markdown(f"""
        <div class="mc">
          <span class="lbl">Annual Inputs</span>
          <span class="ttl">Seed &amp; Polymer Quantities</span>
          <div class="mi"><span class="k">Total seeds / yr</span><span class="v">{seeds_display}</span></div>
          <div class="mi"><span class="k">Seed mass / yr</span><span class="v">{total_seed_mass_g/1000:,.2f} kg</span></div>
          <div class="mi"><span class="k">Polymer input / yr</span><span class="vb">{annual_input_g/1000:,.2f} kg</span></div>
        </div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="mc">
          <span class="lbl">Steady-State Mass</span>
          <span class="ttl">Plastic in Soil at Equilibrium</span>
          <div class="mi"><span class="k">SBR accumulation</span><span class="v">{steady_sbr/1000:,.2f} kg</span></div>
          <div class="mi"><span class="k">PU accumulation</span><span class="vb">{steady_pu/1000:,.2f} kg</span></div>
          <div class="mi"><span class="k">PEO accumulation</span><span class="vg">{steady_peo/1000:,.2f} kg</span></div>
        </div>""", unsafe_allow_html=True)

    with rc3:
        st.markdown(f"""
        <div class="mc">
          <span class="lbl">Comparative Benefits · PU</span>
          <span class="ttl">Environmental Improvement</span>
          <div class="mi"><span class="k">Microplastic reduction</span><span class="vg">{reduction_mass_kg:,.2f} kg</span></div>
          <div class="mi"><span class="k">Ecotoxicity reduction</span><span class="vg">{reduction_toxicity_percent:,.4f}%</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────
    st.markdown('<div class="sec" style="margin-top:2.4rem;"><span class="num">06</span><h2>Simulation Charts</h2><div class="rule"></div></div>', unsafe_allow_html=True)

    FG    = "#0d1b2e"
    GRID  = "#dce3ec"
    SBR_C = "#b5893a"
    PU_C  = "#0071c5"
    PEO_C = "#1a8c5c"

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#ffffff")
        ax.figure.patch.set_facecolor("#f7f8fa")
        for sp in ax.spines.values():
            sp.set_color(GRID); sp.set_linewidth(0.8)
        ax.tick_params(colors="#4a5e78", labelsize=8)
        ax.set_xlabel(xlabel, color="#6b7c94", fontsize=8.5, labelpad=8)
        ax.set_ylabel(ylabel, color="#6b7c94", fontsize=8.5, labelpad=8)
        ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=14)
        ax.grid(axis="both", color=GRID, linewidth=0.7, linestyle="--", alpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1000:,.1f}k" if abs(x) >= 1000 else f"{x:,.2g}"
        ))
        ax.legend(frameon=True, framealpha=0.92, facecolor="#fff",
                  edgecolor=GRID, labelcolor=FG, fontsize=8.5, loc="upper left")

    pc1, pc2 = st.columns(2, gap="large")
    with pc1:
        fig1, ax1 = plt.subplots(figsize=(6.2, 3.8))
        ax1.plot(t, mass_sbr, color=SBR_C, linewidth=2.2, label="SBR", solid_capstyle="round")
        ax1.plot(t, mass_pu,  color=PU_C,  linewidth=2.2, label="PU",  solid_capstyle="round")
        ax1.plot(t, mass_peo, color=PEO_C, linewidth=2.2, label="PEO", solid_capstyle="round")
        ax1.fill_between(t, mass_sbr, mass_pu, alpha=0.10, color=SBR_C)
        style_ax(ax1, "Plastic Accumulation Over Time", "Time (years)", "Plastic Mass in Soil (g)")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)

    with pc2:
        fig2, ax2 = plt.subplots(figsize=(6.2, 3.8))
        ax2.plot(t, impact_sbr, color=SBR_C, linewidth=2.2, label="SBR", solid_capstyle="round")
        ax2.plot(t, impact_pu,  color=PU_C,  linewidth=2.2, label="PU",  solid_capstyle="round")
        ax2.plot(t, impact_peo, color=PEO_C, linewidth=2.2, label="PEO", solid_capstyle="round")
        ax2.fill_between(t, impact_sbr, impact_pu, alpha=0.10, color=SBR_C)
        style_ax(ax2, "Soil Ecotoxicity Over Time", "Time (years)", "Ecotoxicity (CTUe)")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # ── Export ───────────────────────────────────────────────
    st.markdown('<div class="sec" style="margin-top:2.4rem;"><span class="num">07</span><h2>Export Data</h2><div class="rule"></div></div>', unsafe_allow_html=True)

    results_df = pd.DataFrame({
        "Year":            t,
        "Mass_SBR_g":      mass_sbr,
        "Mass_PU_g":       mass_pu,
        "Mass_PEO_g":      mass_peo,
        "Impact_SBR_CTUe": impact_sbr,
        "Impact_PU_CTUe":  impact_pu,
        "Impact_PEO_CTUe": impact_peo
    })
    st.dataframe(results_df, use_container_width=True)

    dl1, dl2 = st.columns(2, gap="medium")
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    with dl1:
        st.download_button("⬇  Download CSV", data=csv_data,
                           file_name="plastaccum_plot_data.csv", mime="text/csv")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        results_df.to_excel(writer, sheet_name="Plot Data", index=False)
        summary_df = pd.DataFrame({
            "Metric": [
                "Total seeds per year", "Total seed mass per year (kg)",
                "Annual polymer input (kg)", "Steady-state SBR mass (kg)",
                "Steady-state PU mass (kg)", "Steady-state PEO mass (kg)",
                "Total impact SBR", "Total impact PU", "Total impact PEO",
                "Reduction in mass (kg)", "Reduction in ecotoxicity (%)"
            ],
            "Value": [
                seeds_display, total_seed_mass_g / 1000, annual_input_g / 1000,
                steady_sbr / 1000, steady_pu / 1000, steady_peo / 1000,
                total_impact_sbr, total_impact_pu, total_impact_peo,
                reduction_mass_kg, reduction_toxicity_percent
            ]
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    with dl2:
        st.download_button("⬇  Download Excel", data=excel_buffer.getvalue(),
                           file_name="plastaccum_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==========================================================
# Footer
# ==========================================================
st.markdown("""
<div class="ftr">
  <p>Duke University · Pratt School of Engineering</p>
  <p>BASF SE · Safe and Sustainable By Design Innovation</p>
  <p>Confidential — For Research Use Only</p>
</div>
""", unsafe_allow_html=True)
