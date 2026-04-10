import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PV String Performance Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLE
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fbf7 0%, #eef6ee 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.top-card {
    background: linear-gradient(135deg, #ffffff 0%, #f7fbf4 100%);
    border: 1px solid rgba(44, 95, 45, 0.10);
    border-radius: 22px;
    padding: 22px 26px 18px 26px;
    box-shadow: 0 8px 24px rgba(20, 40, 20, 0.06);
    margin-bottom: 20px;
}
.soft-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(44, 95, 45, 0.08);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 6px 16px rgba(20, 40, 20, 0.05);
    margin-bottom: 16px;
}
.project-title {
    color: #214c2b;
    font-size: 30px;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 8px;
}
.project-subtitle {
    color: #50655a;
    font-size: 16px;
    font-weight: 500;
    margin-bottom: 4px;
}
.muted-note {
    color: #6b7c71;
    font-size: 13px;
    margin-top: 4px;
}
.section-title {
    color: #1e5b3b;
    font-size: 22px;
    font-weight: 750;
    margin-top: 8px;
    margin-bottom: 10px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #e6f6de 0%, #fff5cf 100%);
    color: #2f5b2c;
    font-size: 13px;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 8px;
    border: 1px solid rgba(64, 122, 58, 0.10);
}
.status-good {
    padding: 12px 14px;
    border-radius: 14px;
    background: #edf9ef;
    border-left: 6px solid #2e8b57;
    color: #1f4f35;
    font-weight: 700;
}
.status-mid {
    padding: 12px 14px;
    border-radius: 14px;
    background: #fff8e8;
    border-left: 6px solid #d5a021;
    color: #6e5311;
    font-weight: 700;
}
.status-bad {
    padding: 12px 14px;
    border-radius: 14px;
    background: #fff0ef;
    border-left: 6px solid #c75b52;
    color: #7a2f2a;
    font-weight: 700;
}
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4faf2 0%, #eef6ee 100%);
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(44, 95, 45, 0.08);
    padding: 10px 12px;
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(20, 40, 20, 0.04);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DEFAULT DATA FROM REPORT
# =========================================================
SITES = {
    "Al-Khafah": {
        "default_panel_name": "JKM575N-72HL4-BDV",
        "panel_options": {
            "JKM575N-72HL4-BDV": {
                "pmax_panel": 575.0,
                "vmp_panel": 42.44,
                "imp_panel": 13.55,
                "voc_panel": 51.27,
                "isc_panel": 14.31,
                "panel_length_mm": 2278.0,
                "panel_width_mm": 1134.0,
            }
        },
        "irradiance_default": 890.0,
        "module_temp_default": 67.0,
        "ambient_temp_default": 40.0,
    },
    "Ar-Rass": {
        "default_panel_name": "JKM620N-78HL4-BDV",
        "panel_options": {
            "JKM620N-78HL4-BDV": {
                "pmax_panel": 620.0,
                "vmp_panel": 45.93,
                "imp_panel": 13.50,
                "voc_panel": 55.58,
                "isc_panel": 14.19,
                "panel_length_mm": 2465.0,
                "panel_width_mm": 1134.0,
            }
        },
        "irradiance_default": 940.0,
        "module_temp_default": 65.0,
        "ambient_temp_default": 42.0,
    },
}

# =========================================================
# HELPERS
# =========================================================
def mm_to_m(mm: float) -> float:
    return mm / 1000.0

def panel_area_m2(length_mm: float, width_mm: float) -> float:
    return mm_to_m(length_mm) * mm_to_m(width_mm)

def calc_efficiency_from_pmax(pmax, n_modules, length_mm, width_mm, irradiance):
    denominator = n_modules * length_mm * width_mm * (10**-6) * irradiance
    if denominator <= 0:
        return 0.0
    return (pmax / denominator) * 100.0

def estimate_vmp_from_voc(voc: float) -> float:
    return 0.82 * voc

def estimate_imp_from_isc(isc: float) -> float:
    return 0.95 * isc

def generate_normalized_iv_curve(voc: float, isc: float, n_points: int = 300):
    v = np.linspace(0, voc, n_points)
    x = v / max(voc, 1e-9)

    a = 8.5
    b = 1.35
    i_norm = np.maximum(0, (1 - x**a)**b)

    i = isc * i_norm
    p = v * i

    pmax_idx = int(np.argmax(p))
    vmp = float(v[pmax_idx])
    imp = float(i[pmax_idx])
    pmax = float(p[pmax_idx])

    return v, i, p, vmp, imp, pmax

# =========================================================
# HEADER
# =========================================================
with st.container():
    st.markdown('<div class="top-card">', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])

    # الشعار
    with c1:
        try:
            st.image("logo.jpeg", width=280)
        except:
            st.info("ضع ملف الشعار باسم logo.jpeg")

    # النص
    with c2:
        st.markdown(
            """
            <div class="project-title">
            Dust Accumulation Impacts on Photovoltaic System Performance in the Saudi Desert Environment
            </div>
            <div class="project-subtitle">
            Qassim University – College of Engineering – Electrical Engineering Department
            </div>
            <div class="muted-note">
            Senior Design Project | PV String Performance Analyzer
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <span class="badge">Renewable Energy</span>
            <span class="badge">Solar PV</span>
            <span class="badge">I-V Curve</span>
            <span class="badge">P-V Curve</span>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("System Configuration")

site_name = st.sidebar.selectbox("Select Site / المحطة", list(SITES.keys()))
site = SITES[site_name]
module_temp = site["module_temp_default"]
ambient_temp = site["ambient_temp_default"]

panel_name = st.sidebar.selectbox(
    "Select Panel Type / نوع اللوح",
    list(site["panel_options"].keys()),
    index=list(site["panel_options"].keys()).index(site["default_panel_name"])
)
panel = site["panel_options"][panel_name]

st.sidebar.markdown("---")
st.sidebar.subheader("Measured Inputs / القيم المدخلة")

series_modules = st.sidebar.number_input(
    "Number of Modules in Series",
    min_value=1,
    value=28,
    step=1
)

if site_name == "Al-Khafah":
    default_measured_isc = 13.40
    default_measured_voc = 1473.50
elif site_name == "Ar-Rass":
    default_measured_isc = 13.78
    default_measured_voc = 1473.62
else:
    default_measured_isc = panel["isc_panel"]
    default_measured_voc = panel["voc_panel"] * series_modules

measured_isc = st.sidebar.number_input(
    "Measured Isc (A)",
    min_value=0.0,
    value=default_measured_isc,
    step=0.01
)

measured_voc = st.sidebar.number_input(
    "Measured Voc (V)",
    min_value=0.0,
    value=default_measured_voc,
    step=0.01
)

irradiance = st.sidebar.number_input(
    "Irradiance (W/m²)",
    min_value=0.0,
    value=1000.0,
    step=10.0
)



st.sidebar.markdown("---")

# =========================================================
# DERIVED VALUES
# =========================================================
single_panel_area = panel_area_m2(panel["panel_length_mm"], panel["panel_width_mm"])
total_area = single_panel_area * series_modules

v_curve, i_curve, p_curve, curve_vmp, curve_imp, curve_pmax = generate_normalized_iv_curve(
    measured_voc, measured_isc
)

vmp_per_panel_est = curve_vmp / series_modules
voc_per_panel_est = measured_voc / series_modules

efficiency = calc_efficiency_from_pmax(
    curve_pmax,
    series_modules,
    2465,
    1134,
    irradiance
)

fill_factor = 0.0
if measured_voc > 0 and measured_isc > 0:
    fill_factor = curve_pmax / (measured_voc * measured_isc)

# =========================================================
# OVERVIEW METRICS
# =========================================================
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Site", site_name)
m2.metric("Panel Type", panel_name)
m3.metric("Modules in Series", f"{series_modules}")
m4.metric("Total Area (m²)", f"{total_area:.3f}")

st.markdown("")

r1, r2, r3 = st.columns(3)
r4, r5, r6 = st.columns(3)

r1.metric("Entered Isc (A)", f"{measured_isc:.2f}")
r2.metric("Entered Voc (V)", f"{measured_voc:.2f}")
r3.metric("Calculated Pmax (W)", f"{curve_pmax:.2f}")

r4.metric("Calculated Vmp (V)", f"{curve_vmp:.2f}")
r5.metric("Calculated Imp (A)", f"{curve_imp:.2f}")
r6.metric("Efficiency (%)", f"{efficiency:.2f}")

st.markdown("")

# =========================================================
# INFO TABLES
# =========================================================
left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Panel & String Information</div>', unsafe_allow_html=True)

    info_df = pd.DataFrame({
        "Item": [
            "Site",
            "Panel Type",
            "Modules in Series",
            "Single Panel Datasheet Voc (V)",
            "Single Panel Datasheet Isc (A)",
            "Single Panel Area (m²)",
            "Estimated String Vmp (V)",
            "Estimated String Imp (A)",
            "Estimated String Pmax (W)",
            "Estimated String Fill Factor",
            "Estimated Voc per Panel (V)",
            "Estimated Vmp per Panel (V)",
        ],
        "Value": [
            site_name,
            panel_name,
            int(series_modules),
            f"{panel['voc_panel']:.2f}",
            f"{panel['isc_panel']:.2f}",
            f"{single_panel_area:.4f}",
            f"{curve_vmp:.2f}",
            f"{curve_imp:.2f}",
            f"{curve_pmax:.2f}",
            f"{fill_factor:.3f}",
            f"{voc_per_panel_est:.2f}",
            f"{vmp_per_panel_est:.2f}",
        ]
    })
    st.dataframe(info_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Environmental Inputs</div>', unsafe_allow_html=True)

    env_df = pd.DataFrame({
        "Parameter": [
            "Irradiance (W/m²)",
            "Module Temperature (°C)",
            "Ambient Temperature (°C)",
        ],
        "Value": [
            f"{irradiance:.2f}",
            f"{module_temp:.2f}",
            f"{ambient_temp:.2f}",
        ]
    })

    st.dataframe(env_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CURVES
# =========================================================
st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Generated I-V and P-V Curves</div>', unsafe_allow_html=True)

c_iv, c_pv = st.columns(2)

with c_iv:
    fig_iv, ax_iv = plt.subplots(figsize=(6.5, 4.2))
    ax_iv.plot(v_curve, i_curve, linewidth=2.5)
    ax_iv.scatter([0, measured_voc, curve_vmp], [measured_isc, 0, curve_imp], s=50)
    ax_iv.set_xlabel("Voltage (V)")
    ax_iv.set_ylabel("Current (A)")
    ax_iv.set_title("I-V Curve")
    ax_iv.grid(True, alpha=0.35)
    st.pyplot(fig_iv)

with c_pv:
    fig_pv, ax_pv = plt.subplots(figsize=(6.5, 4.2))
    ax_pv.plot(v_curve, p_curve, linewidth=2.5)
    ax_pv.scatter([curve_vmp], [curve_pmax], s=55)
    ax_pv.set_xlabel("Voltage (V)")
    ax_pv.set_ylabel("Power (W)")
    ax_pv.set_title("P-V Curve")
    ax_pv.grid(True, alpha=0.35)
    st.pyplot(fig_pv)

st.markdown(
    """
    <div class="muted-note">
    The curves are generated from the entered <b>Isc</b> and <b>Voc</b> using a normalized PV-shaped model,
    so the graph keeps the well-known smooth photovoltaic form even when current and voltage values change.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CURVE POINTS TABLE
# =========================================================
st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sample Curve Points</div>', unsafe_allow_html=True)

sample_idx = np.linspace(0, len(v_curve) - 1, 14, dtype=int)
curve_points_df = pd.DataFrame({
    "Voltage (V)": np.round(v_curve[sample_idx], 2),
    "Current (A)": np.round(i_curve[sample_idx], 2),
    "Power (W)": np.round(p_curve[sample_idx], 2),
})

st.dataframe(curve_points_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HOW IT WORKS
# =========================================================
st.markdown('<div class="soft-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">How the App Works</div>', unsafe_allow_html=True)

st.write("""
- The user selects the **site** and **panel type**.
- Then the user enters only:
  - **Measured Isc**
  - **Measured Voc**
  - **Number of modules in series**
- The environmental values such as:
  - **Irradiance**
  - **Module temperature**
  - **Ambient temperature**
  remain pre-filled from the project defaults and can still be edited.
- The app builds a **normalized photovoltaic-shaped I-V curve** from the entered Isc and Voc.
- From that curve, it automatically calculates:
  - **Vmp**
  - **Imp**
  - **Pmax**
  - **Efficiency**
- The generated I-V and P-V curves keep the familiar PV curve shape even when the user changes current and voltage values.
""")

st.markdown("</div>", unsafe_allow_html=True)