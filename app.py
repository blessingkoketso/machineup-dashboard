"""
SAPCS Dashboard — Main Entry Point
===================================
This file is the assembly shell. It imports one get_tabs() function
from each layer module and renders them as Streamlit tabs.
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

st.set_page_config(
    page_title="SAPCS Molecular Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
   
)

# ── Import layer modules ──────────────────────────────────────────────────────
from src.layers.layer1.tabs import get_tabs as l1_tabs 
#from src.layers.layer2.tabs import get_tabs as l2_tabs 
#from src.layers.layer3.tabs import get_tabs as l3_tabs
from src.shared.styles import inject_dashboard_styles  

# ── Global header ─────────────────────────────────────────────────────────────
inject_dashboard_styles()

# ── Sidebar Filters & Navigation ─────────────────────────────────────────────
st.sidebar.title("⚙️ Global Filters")

from src.shared.data_utils import load_cohort
df_global = load_cohort()
if df_global.empty:
    st.error(
        "No cohort data found. Add 'merged_feature_matrix_SA.csv' (or 'clinical.csv') "
        "to dashboard/data/processed before launching the app." \
        "To get this file 'merged_feature_matrix_SA.csv', you have to get 41586_2022_5154_MOESM3_ESM.xlsx (raw excel) in the mit808-2026-project-machineup/data/raw" \
        "and run 12_Summary_and_Feature_Matrix.ipynb notebook to generate the merged feature matrix csv file that will be mit808-2026-project-machineup/data/processed "
    )
    st.stop()

st.sidebar.markdown("### Patient Filters")
min_age = int(df_global["Age_numeric"].min())
max_age = int(df_global["Age_numeric"].max())
selected_age = st.sidebar.slider("Age Range (years)", min_age, max_age, (min_age, max_age), help="Filter the patient cohort by their age at diagnosis.")

min_psa = float(df_global["PSA_log"].min())
max_psa = float(df_global["PSA_log"].max())
selected_psa = st.sidebar.slider("Log(PSA) Range", min_psa, max_psa, (min_psa, max_psa), help="Filter by the natural logarithm of Prostate-Specific Antigen (PSA) levels, a key indicator for prostate cancer.")

# Filtering logic
filtered_df = df_global[
    df_global["Age_numeric"].between(*selected_age) &
    df_global["PSA_log"].between(*selected_psa)
]

# We define a global state so tabs can access it
st.session_state.filtered_df = filtered_df

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Use these filters to update data across the dashboard. Note: Model training is not affected by these filters.")

st.title("🧬 SAPCS Molecular Dashboard")
st.caption(
    "Southern African Prostate Cancer Study · "
    "African-ancestry cohort · "
)

# ── Assemble all tabs ──────────────────────────────────────────────────────────
all_tab_defs = l1_tabs() #+ l2_tabs() + l3_tabs()

if not all_tab_defs:
    st.info("No tabs registered yet. Implement get_tabs() in a layer module.")
else:
    tab_objects = st.tabs([t["name"] for t in all_tab_defs])
    for tab_obj, tab_def in zip(tab_objects, all_tab_defs):
        with tab_obj:
            tab_def["render"]()
