"""Plotly visualizations used by the dashboard and notebook-derived model tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Shared palette (mirrors the dashboard style system)
ACCENT = "#0EA5E9"
TEAL = "#0D9488"
HIGH_CLR = "#E53E3E"
LOW_CLR = "#38A169"
AMBER = "#D97706"
PURPLE = "#7C3AED"
NAVY = "#071121"
TEXT_MID = "#475569"
TEXT_SOFT = "#94A3B8"
BORDER = "#E2E8F0"

ISUP_COLORS = {1: LOW_CLR, 2: "#86EFAC", 3: AMBER, 4: "#F97316", 5: HIGH_CLR}

_LAYOUT = dict(
	paper_bgcolor="white",
	plot_bgcolor="white",
	font=dict(family="Inter, DM Sans, Arial, sans-serif", size=12, color=TEXT_MID),
	margin=dict(l=10, r=10, t=30, b=10),
)


def fig_isup_bar(df):
	"""Horizontal bar chart: distribution of raw ISUP_GG values."""
	frame = df.copy()
	frame["_ISUP_GG"] = pd.to_numeric(frame["ISUP_GG"], errors="coerce").round()
	counts = frame["_ISUP_GG"].dropna().astype(int).value_counts().sort_index().reset_index(name="Count")
	counts.columns = ["ISUP_GG", "Count"]
	counts["ISUP_GG"] = counts["ISUP_GG"].astype(str)
	fig = px.bar(counts, x="Count", y="ISUP_GG", orientation="h", color="ISUP_GG", color_discrete_sequence=px.colors.qualitative.Set2)
	fig.update_layout(
		**_LAYOUT,
		height=300,
		legend=dict(orientation="h", x=0, y=-0.25),
		xaxis=dict(title="Count", showgrid=True, gridcolor=BORDER, zeroline=False),
		yaxis=dict(title="ISUP_GG", showgrid=False),
	)
	return fig


def fig_onco_subtypes(df):
	"""Horizontal stacked bar chart: GMS_subtype composition by High-Risk status."""
	frame = df.copy()
	frame["_subtype"] = frame["GMS_subtype"].fillna("Unknown").astype(str)

	if "ISUP_GG" in frame.columns:
		isup = pd.to_numeric(frame["ISUP_GG"], errors="coerce")
		frame["_risk_group"] = np.where(isup >= 3, "High-Risk", "Low-Risk")
	else:
		frame["_risk_group"] = "Unknown"

	if frame.empty:
		fig = go.Figure()
		fig.add_annotation(text="No GMS_subtype data available", showarrow=False, font=dict(color=TEXT_MID))
		fig.update_layout(**_LAYOUT, height=220)
		return fig

	summary = frame.groupby("_subtype").agg(
		Total=("_subtype", "size"),
		HighRisk=("_risk_group", lambda s: int((s == "High-Risk").sum())),
	).reset_index()
	summary["HighRiskRate"] = summary["HighRisk"] / summary["Total"].replace(0, np.nan)
	summary = summary.sort_values(["HighRiskRate", "Total", "_subtype"], ascending=[False, False, True])
	order = summary["_subtype"].tolist()

	counts = frame.groupby(["_subtype", "_risk_group"]).size().reset_index(name="Count")
	counts = counts.merge(summary, on="_subtype", how="left")
	counts["_subtype"] = pd.Categorical(counts["_subtype"], categories=order, ordered=True)

	fig = go.Figure()
	for risk_group in ["Low-Risk", "High-Risk", "Unknown"]:
		subset = counts[counts["_risk_group"] == risk_group].sort_values("_subtype")
		if subset.empty:
			continue
		fig.add_trace(
			go.Bar(
				y=subset["_subtype"].astype(str),
				x=subset["Count"],
				orientation="h",
				name=risk_group,
				marker=dict(color={"Low-Risk": LOW_CLR, "High-Risk": HIGH_CLR, "Unknown": TEXT_SOFT}[risk_group]),
				customdata=np.column_stack(
					[
						subset["Total"].to_numpy(),
						subset["HighRisk"].to_numpy(),
						(subset["HighRiskRate"].fillna(0).to_numpy() * 100),
						((subset["Count"].to_numpy() / subset["Total"].replace(0, np.nan).to_numpy()) * 100),
					]
				),
				hovertemplate=(
					"<b>%{y}</b><br>"
					"Segment: %{fullData.name}<br>"
					"Patients in segment: %{x}<br>"
					"Share of subtype: %{customdata[3]:.1f}%<br>"
					"Subtype total: %{customdata[0]}<br>"
					"High-Risk within subtype: %{customdata[1]} (%{customdata[2]:.1f}%)<extra></extra>"
				),
			)
		)

	fig.update_layout(
		**_LAYOUT,
		height=300,
		barmode="stack",
		xaxis=dict(title="Patients", showgrid=True, gridcolor=BORDER, zeroline=False),
		yaxis=dict(title="GMS_subtype", showgrid=False),
		legend=dict(orientation="h", x=0, y=-0.25),
	)
	return fig


def fig_psa_age_scatter(df, highlight_id=None):
	"""Scatter plot: Age_numeric vs PSA_log, optionally highlighting one patient."""

	fig = go.Figure()
	frame = df.copy()
	frame["_ISUP_GG"] = pd.to_numeric(frame["ISUP_GG"], errors="coerce").round().astype("Int64")
	for grade in sorted(frame["_ISUP_GG"].dropna().unique().tolist()):
		grp = frame[frame["_ISUP_GG"] == grade]
		ids_to_skip = {highlight_id} if highlight_id else set()
		grp_bg = grp[~grp["Sample_id"].isin(ids_to_skip)]
		fig.add_trace(
			go.Scatter(
				x=grp_bg["Age_numeric"],
				y=grp_bg["PSA_log"],
				mode="markers",
				name=f"ISUP {int(grade)}",
				marker=dict(color=ISUP_COLORS.get(int(grade), TEXT_SOFT), size=5, opacity=0.45, line=dict(width=0)),
				hovertemplate="<b>%{text}</b><br>Age_numeric: %{x:.1f}<br>PSA_log: %{y:.3f}<extra></extra>",
				text=grp_bg["Sample_id"],
			)
		)
	if highlight_id:
		row = df[df["Sample_id"] == highlight_id]
		if not row.empty:
			fig.add_trace(
				go.Scatter(
					x=row["Age_numeric"],
					y=row["PSA_log"],
					mode="markers+text",
					name=highlight_id,
					text=[highlight_id],
					textposition="top center",
					marker=dict(color=NAVY, size=12, symbol="star", line=dict(width=1.5, color=ACCENT)),
				)
			)
	fig.update_layout(
		**_LAYOUT,
		height=260,
		xaxis=dict(title="Age_numeric", showgrid=True, gridcolor=BORDER),
		yaxis=dict(title="PSA_log", showgrid=True, gridcolor=BORDER),
		legend=dict(orientation="h", x=0, y=-0.30),
	)
	return fig


def fig_tmb_by_isup(df):
	"""Scatter plot: TMB vs PGA_200, colored by ISUP_GG, with a linear trend line."""

	frame = df.copy()
	frame["_ISUP_GG"] = pd.to_numeric(frame["ISUP_GG"], errors="coerce").round().astype("Int64")
	plot_frame = frame.dropna(subset=["TMB", "PGA_200"])

	if plot_frame.empty:
		fig = go.Figure()
		fig.add_annotation(text="No TMB/PGA_200 data available", showarrow=False, font=dict(color=TEXT_MID))
		fig.update_layout(**_LAYOUT, height=240)
		return fig

	corr_frame = plot_frame[["TMB", "PGA_200"]].astype(float)
	pearson_r = corr_frame["TMB"].corr(corr_frame["PGA_200"])

	fig = go.Figure()
	for grade in sorted(plot_frame["_ISUP_GG"].dropna().unique().tolist()):
		grp = plot_frame[plot_frame["_ISUP_GG"] == grade]
		fig.add_trace(
			go.Scatter(
				x=grp["PGA_200"],
				y=grp["TMB"],
				mode="markers",
				name=f"ISUP {int(grade)}",
				marker=dict(color=ISUP_COLORS.get(int(grade), TEXT_SOFT), size=7, opacity=0.75, line=dict(width=0)),
				hovertemplate="<b>%{text}</b><br>PGA_200: %{x:.3f}<br>TMB: %{y:.3f}<extra></extra>",
				text=grp["Sample_id"],
			)
		)

	unknown_grp = plot_frame[plot_frame["_ISUP_GG"].isna()]
	if not unknown_grp.empty:
		fig.add_trace(
			go.Scatter(
				x=unknown_grp["PGA_200"],
				y=unknown_grp["TMB"],
				mode="markers",
				name="Unknown ISUP",
				marker=dict(color=TEXT_SOFT, size=7, opacity=0.55, line=dict(width=0)),
				hovertemplate="<b>%{text}</b><br>PGA_200: %{x:.3f}<br>TMB: %{y:.3f}<extra></extra>",
				text=unknown_grp["Sample_id"],
			)
		)

	if corr_frame["PGA_200"].nunique() > 1 and corr_frame["TMB"].nunique() > 1:
		x_line = np.linspace(corr_frame["PGA_200"].min(), corr_frame["PGA_200"].max(), 100)
		slope, intercept = np.polyfit(corr_frame["PGA_200"], corr_frame["TMB"], 1)
		y_line = slope * x_line + intercept
		fig.add_trace(
			go.Scatter(
				x=x_line,
				y=y_line,
				mode="lines",
				name="Linear fit",
				line=dict(color=TEXT_MID, dash="dash", width=1.5),
				hoverinfo="skip",
			)
		)

	fig.add_annotation(
		xref="paper",
		yref="paper",
		x=0.01,
		y=1.10,
		text=f"Pearson r = {pearson_r:.2f}" if np.isfinite(pearson_r) else "Pearson r = n/a",
		showarrow=False,
		font=dict(size=12, color=TEXT_MID),
	)
	fig.update_layout(
		**_LAYOUT,
		height=300,
		xaxis=dict(title="PGA_200", showgrid=True, gridcolor=BORDER, zeroline=False),
		yaxis=dict(title="TMB", showgrid=True, gridcolor=BORDER, zeroline=False),
		legend=dict(orientation="h", x=0, y=-0.35),
	)
	return fig


def fig_actual_vs_scenario(actual, scenario):
	"""Grouped bar chart: actual vs scenario standardized raw feature values."""

	features = ["Age_numeric", "PSA_log", "TMB", "Cellularity", "PGA_200", "Ploidy", "Driver counts"]
	keys = ["Age_numeric", "PSA_log", "TMB", "Cellularity", "PGA_200", "Ploidy", "Driver counts"]

	anchors = {
		"Age_numeric": (63, 9),
		"PSA_log": (1.5, 1.0),
		"TMB": (0.6, 0.6),
		"Cellularity": (0.55, 0.15),
		"PGA_200": (0.30, 0.18),
		"Ploidy": (2.4, 0.5),
		"Driver counts": (120, 60),
	}

	def std(key, val):
		mu, sigma = anchors.get(key, (0, 1))
		return (val - mu) / sigma if sigma else 0.0

	act_z = [std(key, actual.get(key, 0)) for key in keys]
	scn_z = [std(key, scenario.get(key, 0)) for key in keys]

	fig = go.Figure()
	fig.add_trace(go.Bar(name="Actual", x=features, y=act_z, marker_color=TEXT_SOFT, opacity=0.80))
	fig.add_trace(go.Bar(name="Scenario", x=features, y=scn_z, marker_color=TEAL, opacity=0.90))
	fig.update_layout(
		**_LAYOUT,
		height=220,
		barmode="group",
		yaxis=dict(title="Std. units", showgrid=True, gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER),
		xaxis=dict(showgrid=False),
		legend=dict(orientation="h", x=0, y=-0.20),
	)
	return fig


def fig_pga_driver_scatter(df):
	"""Scatter plot: PGA_200 vs driver counts, grouped by raw GMS_subtype."""

	fig = go.Figure()
	for subtype in sorted(df["GMS_subtype"].dropna().astype(str).unique()):
		grp = df[df["GMS_subtype"].astype(str) == subtype]
		fig.add_trace(
			go.Scatter(
				x=grp["PGA_200"],
				y=grp["Driver counts"],
				mode="markers",
				name=subtype,
				marker=dict(size=6, opacity=0.65, line=dict(width=0.5, color="white")),
				hovertemplate="<b>%{text}</b><br>PGA_200: %{x:.3f}<br>Driver counts: %{y}<extra></extra>",
				text=grp["Sample_id"],
			)
		)
	fig.update_layout(
		**_LAYOUT,
		height=260,
		xaxis=dict(title="PGA_200", showgrid=True, gridcolor=BORDER, zeroline=False),
		yaxis=dict(title="Driver counts", showgrid=True, gridcolor=BORDER, zeroline=False),
		legend=dict(orientation="h", x=0, y=-0.30),
	)
	return fig


def fig_patient_vs_median(patient, cohort_medians):
	"""Grouped bar chart: selected patient vs cohort median for raw metrics."""

	metrics = ["Age_numeric", "PSA_log", "TMB", "Cellularity", "PGA_200", "Driver counts"]
	labels = ["Age_numeric", "PSA_log", "TMB", "Cellularity", "PGA_200", "Driver counts"]
	p_vals = [patient.get(metric, 0) for metric in metrics]
	m_vals = [cohort_medians.get(metric, 0) for metric in metrics]

	fig = go.Figure()
	fig.add_trace(go.Bar(name="Patient", x=labels, y=p_vals, marker_color=ACCENT, opacity=0.90))
	fig.add_trace(go.Bar(name="Median", x=labels, y=m_vals, marker_color=TEXT_SOFT, opacity=0.70))
	fig.update_layout(
		**_LAYOUT,
		height=220,
		barmode="group",
		yaxis=dict(showgrid=True, gridcolor=BORDER),
		xaxis=dict(showgrid=False),
		legend=dict(orientation="h", x=0, y=-0.20),
	)
	return fig


def fig_model_confusion_matrix(confusion, class_names):
	"""Annotated confusion matrix for the notebook-derived classifier.
	
	Purpose: Breaks down the exact amounts of true positives, false positives, true negatives, 
	and false negatives to give users unambiguous insight into where the ML model struggles.
	"""

	cm = np.asarray(confusion)
	labels = list(class_names) if len(class_names) == cm.shape[0] else [f"Class {i}" for i in range(cm.shape[0])]
	x_labels = [f"Predicted {label}" for label in labels]
	y_labels = [f"True {label}" for label in labels]

	fig = go.Figure(
		data=go.Heatmap(
			z=cm,
			x=x_labels,
			y=y_labels,
			colorscale="Blues",
			showscale=False,
			zmin=0,
			zmax=max(int(cm.max()), 1),
		)
	)
	for row_idx, row_label in enumerate(y_labels):
		for col_idx, col_label in enumerate(x_labels):
			value = cm[row_idx, col_idx]
			fig.add_annotation(
				x=col_label,
				y=row_label,
				text=str(int(value)),
				showarrow=False,
				font=dict(color="white" if value > cm.max() / 2 else NAVY, size=14),
			)
	layout = dict(_LAYOUT)
	layout["margin"] = dict(l=10, r=10, t=40, b=10)
	fig.update_layout(**layout, height=280, xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
	return fig


def fig_model_roc_curve(fpr, tpr, roc_auc):
	"""ROC curve for the notebook-derived classifier.
	
	Purpose: Evaluates the diagnostic ability of a binary classifier system as its discrimination 
	threshold is varied. High AUC guarantees a good measure of separability between risk classes.
	"""

	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=fpr,
			y=tpr,
			mode="lines",
			line=dict(color="darkorange", width=3),
			name=f"ROC AUC = {roc_auc:.2f}",
		)
	)
	fig.add_trace(
		go.Scatter(
			x=[0, 1],
			y=[0, 1],
			mode="lines",
			line=dict(color=NAVY, width=2, dash="dash"),
			name="Chance",
		)
	)
	fig.update_layout(
		**_LAYOUT,
		height=280,
		xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor=BORDER),
		yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor=BORDER),
		legend=dict(orientation="h", x=0, y=1.15),
	)
	return fig


def fig_model_pr_curve(
	precisions,
	recalls,
	pr_auc,
	target_recall,
	optimal_index=None,
):
	"""Precision-recall curve with the chosen clinical operating point highlighted.
	
	Purpose: Crucial in severe class imbalance situations (e.g. rare but fatal High-Risk disease). 
	Visualizes the direct tradeoff between being overly cautious (Precision) and overly safe (Recall).
	"""

	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			x=recalls,
			y=precisions,
			mode="lines",
			line=dict(color="green", width=3),
			name=f"PR AUC = {pr_auc:.2f}",
		)
	)
	fig.add_vline(x=target_recall, line_width=1.5, line_dash="dash", line_color=HIGH_CLR)
	if optimal_index is not None and 0 <= optimal_index < len(recalls):
		fig.add_trace(
			go.Scatter(
				x=[recalls[optimal_index]],
				y=[precisions[optimal_index]],
				mode="markers",
				marker=dict(color=HIGH_CLR, size=10),
				name="Chosen threshold",
			)
		)
	fig.update_layout(
		**_LAYOUT,
		height=280,
		xaxis=dict(title="Recall (Sensitivity)", showgrid=True, gridcolor=BORDER),
		yaxis=dict(title="Precision", showgrid=True, gridcolor=BORDER),
		legend=dict(orientation="h", x=0, y=1.15),
	)
	return fig


def fig_model_feature_importance(explain_df, score_column, top_k=12):
	"""Horizontal bar chart for the top discriminative variables.
	
	Purpose: Acts as an XAI (Explainable AI) tool. Reveals exactly which patient features 
	(e.g., TMB, PSA, Age) the underlying model relies on the most to make its final risk prediction.
	"""

	if explain_df.empty:
		return go.Figure()

	if score_column not in explain_df.columns:
		score_column = "discriminative_score"

	top_df = explain_df.sort_values(score_column, ascending=False).head(top_k).sort_values(score_column, ascending=True)
	fig = go.Figure(
		go.Bar(
			x=top_df[score_column],
			y=top_df["feature"],
			orientation="h",
			marker=dict(
				color=top_df[score_column],
				colorscale="Viridis",
				colorbar=dict(title="Score"),
			),
			hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
		)
	)
	fig.update_layout(
		**_LAYOUT,
		height=340,
		xaxis=dict(title="Combined discriminative score", showgrid=True, gridcolor=BORDER),
		yaxis=dict(title="Feature", showgrid=False),
	)
	return fig


__all__ = [
	"ACCENT",
	"AMBER",
	"BORDER",
	"HIGH_CLR",
	"ISUP_COLORS",
	"LOW_CLR",
	"NAVY",
	"PURPLE",
	"TEAL",
	"fig_actual_vs_scenario",
	"fig_isup_bar",
	"fig_model_confusion_matrix",
	"fig_model_feature_importance",
	"fig_model_pr_curve",
	"fig_model_roc_curve",
	"fig_onco_subtypes",
	"fig_pga_driver_scatter",
	"fig_patient_vs_median",
	"fig_psa_age_scatter",
	"fig_tmb_by_isup",
]
