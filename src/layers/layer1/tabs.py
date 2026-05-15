"""
Layer 1 — Aim 1: Screening Layer
Entry point for the SAPCS multi-layer dashboard.

Implements the get_tabs() contract:
    get_tabs() returns a list of tab descriptors with a name and a render callback.
"""

from __future__ import annotations

import html

import streamlit as st

from src.features.prepare_model_data import get_model_report
from src.shared.styles import inject_dashboard_styles

from .data_loader import cohort_kpis, load_cohort, load_patient
from .visualize import (
    fig_actual_vs_scenario,
    fig_isup_bar,
    fig_model_confusion_matrix,
    fig_model_feature_importance,
    fig_model_pr_curve,
    fig_model_roc_curve,
    fig_onco_subtypes,
    fig_pga_driver_scatter,
    fig_patient_vs_median,
    fig_psa_age_scatter,
    fig_tmb_by_isup,
)


def _inject_css():
    """Apply the shared dashboard CSS once per render."""
    inject_dashboard_styles()


# ── helpers ────────────────────────────────────────────────────────────────────
def _kpi_html(label, value, sub, colour="blue", sm=False, help_text=""):
    """Render a KPI card as HTML for the overview metrics grid."""
    val_cls = "kpi-value sm" if sm else "kpi-value"
    help_icon = f"""
    <div class="help-tooltip" aria-hidden="true" style="margin-left: 4px;">
        <span style="font-size:12px;">❔</span>
        <span class="tooltip-text">{html.escape(help_text)}</span>
    </div>
    <span class="sr-only" style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0;">{html.escape(help_text)}</span>
    """ if help_text else ""
    
    return f"""
    <div class="kpi-card {colour}" role="status" aria-label="{label}: {value}. {sub} - {help_text}">
        <div class="kpi-label">{label}{help_icon}</div>
        <div class="{val_cls}">{value}</div>
        <div class="kpi-sub" aria-hidden="true">{sub}</div>
    </div>"""


def _badge(value, tone="blue"):
    """Return a styled badge HTML snippet for categorical values."""
    tone = tone if tone in {"high", "low", "blue", "amber", "proto"} else "blue"
    return f'<span class="badge badge-{tone}">{html.escape(str(value))}</span>'


def _stat_row(label, value=None, badge_html=""):
    """Return one row of the patient stats table as HTML."""
    value_html = badge_html if badge_html else f'<span class="stat-val">{html.escape(str(value))}</span>'
    return f"""<div class="stat-row">
        <span class="stat-lbl">{html.escape(str(label))}</span>
        {value_html}
    </div>"""


# ── raw model helper ───────────────────────────────────────────────────────────
def _live_ml_predict(patient, scenario_features):
    """Run the notebook-derived model on raw feature values."""

    from src.features.prepare_model_data import get_model_report, load_model_source_data
    import pandas as pd

    report = get_model_report()
    source_df, _ = load_model_source_data()
    row_df = source_df[source_df["Sample_id"] == patient["Sample_id"]].copy()

    if row_df.empty:
        row_df = pd.DataFrame(columns=source_df.columns)
        row_df.loc[0] = 0

    for column in [
        "Age_numeric",
        "PSA_log",
        "TMB",
        "Cellularity",
        "Ploidy",
        "PGA_200",
        "Driver counts",
        "Chromothripsis",
        "GMS_subtype",
    ]:
        if column in row_df.columns and column in scenario_features:
            row_df[column] = scenario_features[column]

    proba_array = report.pipeline.predict_proba(row_df)[0]
    p_high = float(proba_array[1])
    is_high_risk = p_high >= report.optimal_threshold

    proba = {
        "Low-Risk": round(1 - p_high, 3),
        "High-Risk": round(p_high, 3),
    }
    risk = "High-Risk" if is_high_risk else "Low-Risk"
    return risk, p_high, proba


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def _render_overview():
    """Render the cohort overview tab."""
    _inject_css()
    df = st.session_state.filtered_df if "filtered_df" in st.session_state else load_cohort()
    if df.empty:
        st.warning("No cohort data available. Add merged_feature_matrix_SA.csv to data/processed and reload.")
        return
    kpis = cohort_kpis(df)

    st.markdown("## Aim 1 — Screening Layer: Cohort Overview")

    st.download_button(
        label="📥 Download filtered cohort (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sapcs_filtered_cohort.csv",
        mime="text/csv",
    )

    kpi_html = "".join([
        _kpi_html("Total Samples", str(kpis["total"]), "raw cohort", "blue", help_text="Total number of patients in the filtered cohort."),
        _kpi_html("Median Age", str(kpis["median_age"]), "years", "amber", help_text="Median age at diagnosis for the selected cohort."),
        _kpi_html("Median Log(PSA)", str(kpis["median_psa_log"]), "raw log value", "purple", help_text="Median natural logarithm of Prostate-Specific Antigen levels. Higher values typically increase cancer risk probability."),
        _kpi_html("Median ISUP GG", str(kpis["median_isup"]), "raw grade group", "green", help_text="ISUP Grade Group measures prostate cancer aggressiveness on a scale of 1 to 5 based on cellular patterns."),
        _kpi_html("Median TMB", str(kpis["median_tmb"]), "mut/Mb", "red", help_text="Tumor Mutational Burden (mutations per megabase). Measures the number of mutations inside the tumor cells."),
    ])
    st.markdown(f'<div class="kpi-wrap">{kpi_html}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        with st.container(border=True):
            st.markdown('**ISUP_GG Distribution**', help="Shows how patients are distributed across the 5 ISUP Grade Groups. Grade 1 is the least aggressive, while Grade 5 is the most aggressive form of prostate cancer.")
            st.plotly_chart(fig_isup_bar(df), config={"displayModeBar": False}, key="fig_isup1", use_container_width=True)

    with c2:
        with st.container(border=True):
            st.markdown('**GMS_subtype by High-Risk status**', help="Displays the proportion of Genomic Molecular Subtypes (GMS) segmented by their High-Risk clinical status prediction.")
            st.plotly_chart(fig_onco_subtypes(df), config={"displayModeBar": False}, key="fig_onco1", use_container_width=True)

    with c3:
        with st.container(border=True):
            st.markdown('**TMB vs PGA_200 correlation**', help="Scatter correlation between Tumor Mutational Burden and PGA_200 (Percentage of Genome Altered). Explores the relationship between mutation frequency and large-scale chromosomal alterations.")
            st.plotly_chart(fig_tmb_by_isup(df), config={"displayModeBar": False}, key="fig_tmb_1", use_container_width=True)

    c4, c5 = st.columns([1, 1])
    with c4:
        with st.container(border=True):
            st.markdown('**Age vs Log(PSA)**', help="Scatter plot comparing patient age with their Log(PSA) levels. This correlation helps determine if older patients tend to present with higher tumor markers.")
            st.plotly_chart(fig_psa_age_scatter(df), config={"displayModeBar": False}, key="fig_scatter1")

    with c5:
        with st.container(border=True):
            st.markdown('**PGA_200 vs Driver counts**', help="Correlation between the Percentage of Genome Altered (PGA_200) and the absolute count of driver gene mutations. Demonstrates how structural cancer instability relates to driver gene counts.")
            st.plotly_chart(fig_pga_driver_scatter(df), config={"displayModeBar": False}, key="fig_pga_driver1")

    with st.container(border=True):
        st.markdown('**Raw Data Exploration (Data Table)**', help="Interactive table showing the raw clinical and genomic features for each patient in the current filtered cohort.")
        raw_columns = [
            column
            for column in [
                "Sample_id",
                "Age_numeric",
                "ISUP_GG",
                "PSA_log",
                "TMB",
                "Cellularity",
                "Ploidy",
                "PGA_200",
                "Chromothripsis",
                "Driver counts",
                "GMS_subtype",
            ]
            if column in df.columns
        ]
        st.dataframe(df[raw_columns], width="stretch", hide_index=True)

    patient_list = df["Sample_id"].tolist()
    selected_patient = st.selectbox("Patient ID", patient_list)

    if selected_patient:
        patient_data = df[df["Sample_id"] == selected_patient].iloc[0]
        st.dataframe(patient_data.to_frame().T, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDIVIDUAL PATIENT
# ══════════════════════════════════════════════════════════════════════════════
def _render_individual():
    """Render the individual patient explorer tab."""
    _inject_css()
    df = load_cohort()
    if df.empty:
        st.warning("No cohort data available. Add merged_feature_matrix_SA.csv to data/processed and reload.")
        return

    st.markdown("## Individual Patient Explorer")
    st.caption("Select a patient, inspect the raw record, then tweak raw inputs to refresh the live model output.")

    patient_ids = df["Sample_id"].tolist()
    if "layer1_patient" not in st.session_state:
        st.session_state["layer1_patient"] = patient_ids[0]

    selected_id = st.selectbox(
        "Patient",
        patient_ids,
        index=patient_ids.index(st.session_state["layer1_patient"]),
        key="layer1_patient_select",
        label_visibility="collapsed",
    )
    st.session_state["layer1_patient"] = selected_id
    patient = load_patient(selected_id)

    st.markdown(
        '<div class="alert-info">ℹ️ Select a patient from the dropdown above, inspect the raw record, then edit raw inputs below to refresh the model output.</div>',
        unsafe_allow_html=True,
    )

    kpi_html = "".join([
        _kpi_html("Sample", selected_id, str(patient.get("GMS_subtype", "")), "blue", sm=True),
        _kpi_html("Age_numeric", f"{patient['Age_numeric']:.1f}", "years", "amber"),
        _kpi_html("PSA_log", f"{patient['PSA_log']:.3f}", "raw log value", "purple"),
        _kpi_html("ISUP_GG", f"{patient['ISUP_GG']:.1f}", "raw grade group", "green"),
        _kpi_html("TMB", f"{patient['TMB']:.3f}", "mut/Mb", "red"),
        _kpi_html("Cellularity", f"{patient['Cellularity']:.3f}", "raw fraction", "blue"),
    ])
    st.markdown(f'<div class="kpi-wrap">{kpi_html}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_live, tab_record = st.tabs([" Raw Classification Scenario", " Raw Patient Record"])

    with tab_live:
        st.markdown('<div class="section-lbl">Editable Raw Features</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            scn_age = st.number_input("Age_numeric", value=float(patient["Age_numeric"]), step=0.1, format="%.1f")
            scn_psa = st.number_input("PSA_log", value=float(patient["PSA_log"]), step=0.01, format="%.3f")
            scn_tmb = st.number_input("TMB", value=float(patient["TMB"]), step=0.01, format="%.3f")
        with c2:
            scn_cell = st.number_input("Cellularity", value=float(patient["Cellularity"]), step=0.01, format="%.2f")
            scn_pga = st.number_input("PGA_200", value=float(patient["PGA_200"]), step=0.001, format="%.3f")
            scn_ploi = st.number_input("Ploidy", value=float(patient["Ploidy"]), step=0.1, format="%.1f")
        with c3:
            scn_driv = st.number_input("Driver counts", value=int(patient["Driver counts"]), step=1)
            scn_chro = st.selectbox("Chromothripsis", [0, 1], index=0 if int(patient["Chromothripsis"]) == 0 else 1)
            subtype_options = sorted(df["GMS_subtype"].dropna().astype(str).unique().tolist())
            scn_subtype = st.selectbox(
                "GMS_subtype",
                subtype_options,
                index=subtype_options.index(str(patient["GMS_subtype"])) if str(patient["GMS_subtype"]) in subtype_options else 0,
            )

        scenario_features = {
            "Age_numeric": scn_age,
            "PSA_log": scn_psa,
            "TMB": scn_tmb,
            "Cellularity": scn_cell,
            "PGA_200": scn_pga,
            "Ploidy": scn_ploi,
            "Driver counts": scn_driv,
            "Chromothripsis": scn_chro,
            "GMS_subtype": scn_subtype,
        }
        actual_features = {
            "Age_numeric": float(patient["Age_numeric"]),
            "PSA_log": float(patient["PSA_log"]),
            "TMB": float(patient["TMB"]),
            "Cellularity": float(patient["Cellularity"]),
            "PGA_200": float(patient["PGA_200"]),
            "Ploidy": float(patient["Ploidy"]),
            "Driver counts": int(patient["Driver counts"]),
            "Chromothripsis": int(patient["Chromothripsis"]),
            "GMS_subtype": str(patient["GMS_subtype"]),
        }
        changed_inputs = sum(1 for k in actual_features if scenario_features.get(k) != actual_features.get(k))

        pred_class, confidence, proba = _live_ml_predict(patient, scenario_features)

        st.markdown("---")
        st.markdown('<div class="section-lbl">Live ML Prediction Results (Real Model Pipeline)</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="mini-metric"><div class="mm-lbl">Predicted Class</div>'
                f'<div class="mm-val">{pred_class}</div>'
                f'<div class="mm-sub">Reference ISUP_GG: {patient["ISUP_GG"]}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="mini-metric"><div class="mm-lbl">Confidence</div>'
                f'<div class="mm-val">{confidence:.0%}</div>'
                f'<div class="mm-sub">Model probability (positive class)</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="mini-metric"><div class="mm-lbl">Changed Inputs</div>'
                f'<div class="mm-val">{changed_inputs}</div>'
                f'<div class="mm-sub">vs raw record</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Actual vs Scenario Inputs (Standardised)")
        st.plotly_chart(
            fig_actual_vs_scenario(actual_features, scenario_features),
            config={"displayModeBar": False},
            key="fig_scenario_individual",
            use_container_width=True,
        )

    with tab_record:
        r1, r2 = st.columns([1, 1.3])
        with r1:
            with st.container(border=True):
                st.markdown('<div class="section-lbl">Clinical Snapshot</div>', unsafe_allow_html=True)

                risk_label = "High-Risk" if float(patient["ISUP_GG"]) >= 3 else "Low-Risk"
                risk_badge = _badge(risk_label, "high" if risk_label == "High-Risk" else "low")
                isup_badge = _badge(f"ISUP {int(round(float(patient['ISUP_GG'])))}", "amber")
                subtype_badge = _badge(patient["GMS_subtype"], "blue")

                rows_html = "".join([
                    _stat_row("Sample ID", selected_id),
                    _stat_row("Age_numeric", f"{patient['Age_numeric']:.1f}"),
                    _stat_row("ISUP Grade", badge_html=isup_badge),
                    _stat_row("High-Risk", badge_html=risk_badge),
                    _stat_row("PSA_log", f"{patient['PSA_log']:.3f}"),
                    _stat_row("TMB", f"{patient['TMB']:.3f} mut/Mb"),
                    _stat_row("PGA_200", f"{patient['PGA_200']:.3f}"),
                    _stat_row("Cellularity", f"{patient['Cellularity']:.3f}"),
                    _stat_row("Ploidy", f"{patient['Ploidy']:.2f}"),
                    _stat_row("Chromothripsis", str(int(patient["Chromothripsis"]))),
                    _stat_row("Driver counts", str(int(patient["Driver counts"]))),
                    _stat_row("GMS_subtype", badge_html=subtype_badge),
                ])

                st.markdown(f'<div class="stat-table">{rows_html}</div>', unsafe_allow_html=True)

                with st.expander("Raw patient row", expanded=False):
                    st.dataframe(patient.to_frame().T, hide_index=True, use_container_width=True)

        with r2:
            with st.container(border=True):
                cohort_medians = load_cohort()[
                    ["Age_numeric", "PSA_log", "TMB", "Cellularity", "PGA_200", "Driver counts"]
                ].median()

                st.markdown("**Selected Patient vs Cohort Median**")
                st.plotly_chart(
                    fig_patient_vs_median(patient, cohort_medians),
                    config={"displayModeBar": False},
                    key="fig_patient_vs_median_individual",
                    use_container_width=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Patient in Cohort Context (Age_numeric vs PSA_log)**")
                st.plotly_chart(
                    fig_psa_age_scatter(df, highlight_id=selected_id),
                    config={"displayModeBar": False},
                    key="fig_psa_age_scatter_individual",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
def _render_model_notebook():
    """Render the notebook-derived model evaluation tab."""
    _inject_css()

    if load_cohort().empty:
        st.warning("No cohort data available. Add merged_feature_matrix_SA.csv to data/processed and reload.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("## Notebook-Derived Classifier")
    with col2:
        target_recall_pct = st.slider(
            "Target Recall (%)",
            min_value=50,
            max_value=99,
            value=85,
            step=1,
            help="Expected sensitivity/recall threshold to optimize the classification cut-off.",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Retrain Model with Latest Data", width="stretch", type="primary"):
            st.session_state["force_model_refresh"] = True
            st.rerun()

    force_refresh = st.session_state.pop("force_model_refresh", False)

    with st.spinner("Loading and evaluating the model pipeline..."):
        report = get_model_report(force_refresh=force_refresh, target_recall=target_recall_pct / 100.0)

    st.caption(
        f"Source: {report.source_label} · "
        f"{report.clean_shape[0]} cleaned rows · "
        f"{report.feature_counts['numerical']} numerical / {report.feature_counts['categorical']} categorical / {report.feature_counts['binary']} binary features"
    )

    if report.source_label.startswith("Synthetic fallback"):
        st.info("The processed matrix was not found, so the dashboard is using a deterministic fallback built from the cohort.")
    else:
        st.success("Model source matrix loaded from disk.")

    metric_html = "".join(
        [
            _kpi_html("ROC-AUC", f"{report.metrics['roc_auc']:.3f}", "OOF predictions", "blue", sm=True),
            _kpi_html("PR-AUC", f"{report.metrics['pr_auc']:.3f}", "Class imbalance aware", "amber", sm=True),
            _kpi_html("Accuracy", f"{report.metrics['accuracy']:.3f}", "Chosen threshold", "green", sm=True),
            _kpi_html("Balanced Acc", f"{report.metrics['balanced_accuracy']:.3f}", "Class-balanced view", "purple", sm=True),
            _kpi_html("Recall @ thr", f"{report.metrics['recall']:.3f}", "Clinical target point", "red", sm=True),
            _kpi_html("Threshold", f"{report.optimal_threshold:.3f}", f"Recall target {report.target_recall:.0%}", "blue", sm=True),
        ]
    )
    st.markdown(f'<div class="kpi-wrap">{metric_html}</div>', unsafe_allow_html=True)
    st.caption(
        f"Class distribution: {report.class_distribution.get(0, 0.0):.1%} non-extreme · "
        f"{report.class_distribution.get(1, 0.0):.1%} extreme · "
        f"Precision @ thr {report.metrics['precision']:.3f} · F1 {report.metrics['f1']:.3f} · MCC {report.metrics['mcc']:.3f}"
    )

    with st.container(border=True):
        st.markdown("**Classification Report**")
        st.dataframe(report.classification_report_df.round(3), width="stretch", hide_index=False)

        with st.expander("📖 How to read these metrics?"):
            st.markdown(
                """
                **Class-specific metrics (Low-Risk vs High-Risk):**
                - **Precision**: Out of all patients predicted as a certain class by the model, how many actually belong to that class?
                - **Recall (Sensitivity)**: Out of all ACTUAL "High-Risk" patients, what percentage did the model successfully detect?
                - **F1-Score**: The harmonic mean of Precision and Recall.
                - **Support**: The actual number of patients in each category during evaluation.

                **Global model evaluation scores:**
                - **Accuracy**: The overall rate of correct predictions.
                - **Macro avg**: The unweighted mean of all metrics across classes.
                - **Weighted avg**: The mean of scores weighted by the Support.
                """
            )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(
                fig_model_confusion_matrix(report.confusion_matrix, report.target_names),
                config={"displayModeBar": False},
                key="fig_model_cm",
            )
    with c2:
        with st.container(border=True):
            st.markdown("**ROC Curve**")
            st.plotly_chart(
                fig_model_roc_curve(report.fpr, report.tpr, report.metrics["roc_auc"]),
                config={"displayModeBar": False},
                key="fig_model_roc",
            )
    with c3:
        with st.container(border=True):
            st.markdown("**Precision-Recall Curve**")
            st.plotly_chart(
                fig_model_pr_curve(
                    report.precisions,
                    report.recalls,
                    report.metrics["pr_auc"],
                    report.target_recall,
                    report.optimal_threshold_index,
                ),
                config={"displayModeBar": False},
                key="fig_model_pr",
            )

    with st.container(border=True):
        st.markdown("**Discriminative Variables**")
        display_cols = ["feature", "rf_importance", "perm_importance_mean", "perm_importance_std", report.score_column]
        if "shap_mean_abs" in report.explain_df.columns:
            display_cols.insert(4, "shap_mean_abs")
        st.dataframe(report.explain_df[display_cols].head(12).round(4), width="stretch", hide_index=True)
        st.plotly_chart(
            fig_model_feature_importance(report.explain_df, report.score_column),
            config={"displayModeBar": False},
            key="fig_model_feat_imp",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def get_tabs():
    """Return the Layer 1 tab definitions."""

    return [
        {"name": " Aim 1 — Screening", "render": _render_overview},
        {"name": " Notebook Model", "render": _render_model_notebook},
        {"name": " Individual Patient", "render": _render_individual},
    ]
