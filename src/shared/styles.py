"""Shared dashboard styles used by the Streamlit app and layer tabs."""

from __future__ import annotations

import streamlit as st

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
	font-family: 'DM Sans', 'Inter', sans-serif !important;
}

.block-container {
	padding-top: 1.5rem;
}

/* KPI cards */
.kpi-wrap {
	display: flex;
	flex-wrap: wrap;
	gap: 14px;
	margin-bottom: 4px;
}

.kpi-card {
	background: #fff;
	border: 1px solid #E2E8F0;
	border-radius: 10px;
	padding: 14px 18px;
	min-width: 140px;
	flex: 1;
	position: relative;
	box-shadow: 0 1px 3px rgba(0,0,0,.06);
}

.kpi-card::before {
	content: '';
	position: absolute;
	top: 0;
	left: 0;
	right: 0;
	height: 3px;
	border-radius: 10px 10px 0 0;
}

.kpi-card.blue::before   { background: #0284C7; } /* Assombri pour l'accessibilité */
.kpi-card.red::before    { background: #DC2626; }
.kpi-card.green::before  { background: #16A34A; }
.kpi-card.amber::before  { background: #D97706; }
.kpi-card.purple::before { background: #7C3AED; }

.kpi-label {
	font-size: 10px;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: .07em;
	color: #475569; /* Contraste WCAG AAA (anciennement #94A3B8) */
	margin-bottom: 6px;
}

.kpi-value {
	font-size: 26px;
	font-weight: 700;
	color: #0F172A;
	font-family: 'DM Mono', monospace;
	line-height: 1;
}

.kpi-value.sm {
	font-size: 18px;
	padding-top: 4px;
}

.kpi-sub {
	font-size: 11px;
	color: #475569; /* Contraste WCAG AAA */
	margin-top: 4px;
}

/* Tooltip custom pour les KPI */
.help-tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.help-tooltip .tooltip-text {
    visibility: hidden;
    width: 220px;
    background-color: #1E293B;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 9999;
    bottom: 150%; 
    left: 50%;
    margin-left: -110px;
    font-size: 12px;
    font-weight: 500;
    text-transform: none;
    letter-spacing: normal;
    opacity: 0;
    transition: opacity 0.2s, visibility 0.2s;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.help-tooltip .tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #1E293B transparent transparent transparent;
}

.help-tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* Panels */
.panel {
	background: #fff;
	border: 1px solid #E2E8F0;
	border-radius: 14px;
	box-shadow: 0 1px 3px rgba(0,0,0,.06);
	margin-bottom: 18px;
	overflow: hidden;
}

.panel-header {
	padding: 14px 20px;
	border-bottom: 1px solid #E2E8F0;
	font-size: 13px;
	font-weight: 600;
	color: #0F172A;
	display: flex;
	align-items: center;
	gap: 8px;
}

.panel-header.accent {
	border-top: 3px solid #0EA5E9;
}

.panel-body {
	padding: 20px;
}

/* Section label */
.section-lbl {
	font-size: 11px;
	font-weight: 600;
	text-transform: uppercase;
	letter-spacing: .06em;
	color: #0F172A;
	border-left: 3px solid #0EA5E9;
	padding-left: 8px;
	margin: 0 0 14px;
}

/* Stat rows */
.stat-table { width: 100%; }

.stat-row {
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 9px 0;
	border-bottom: 1px solid #F1F5F9;
	font-size: 12.5px;
}

.stat-row:last-child {
	border-bottom: none;
}

.stat-lbl { color: #475569; }
.stat-val { font-weight: 600; color: #0F172A; font-family: 'DM Mono', monospace; }

/* Risk badge */
.badge {
	display: inline-flex;
	padding: 2px 9px;
	border-radius: 20px;
	font-size: 10.5px;
	font-weight: 600;
}

.badge-high  { background: #FEF2F2; color: #991B1B; }
.badge-low   { background: #F0FDF4; color: #166534; }
.badge-blue  { background: #EFF6FF; color: #1D4ED8; }
.badge-amber { background: #FFFBEB; color: #92400E; }
.badge-proto { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }

/* Alert info */
.alert-info {
	background: #EFF6FF;
	border: 1px solid #BFDBFE;
	color: #1E40AF;
	border-radius: 8px;
	padding: 10px 14px;
	font-size: 12px;
	display: flex;
	align-items: flex-start;
	gap: 8px;
	margin-bottom: 12px;
}

/* Mini metric */
.mini-metric {
	background: #F8FAFC;
	border: 1px solid #E2E8F0;
	border-radius: 8px;
	padding: 12px 16px;
}

.mm-lbl {
	font-size: 10px;
	text-transform: uppercase;
	letter-spacing: .07em;
	color: #94A3B8;
	font-weight: 600;
}

.mm-val {
	font-size: 22px;
	font-weight: 700;
	font-family: 'DM Mono', monospace;
	color: #0F172A;
	margin: 3px 0 2px;
}

.mm-sub {
	font-size: 11px;
	color: #475569;
}

/* Cohort table overrides */
.stDataFrame thead th {
	font-size: 11px !important;
	font-weight: 600 !important;
	text-transform: uppercase !important;
	letter-spacing: .05em !important;
}
</style>
"""


def inject_dashboard_styles():
	"""Inject the shared dashboard CSS into the current Streamlit page."""

	st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


__all__ = ["DASHBOARD_CSS", "inject_dashboard_styles"]
