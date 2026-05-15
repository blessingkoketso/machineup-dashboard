"""Shared data loading and cohort utilities for the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parents[2]
CLINICAL_CSV = DASHBOARD_ROOT / "data" / "processed" / "clinical.csv"
MODEL_MATRIX_CSV = DASHBOARD_ROOT / "data" / "processed" / "merged_feature_matrix_SA.csv"
RANDOM_SEED = 42

RAW_COHORT_COLUMNS = [
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


def resolve_existing_path(*candidates):
	"""Return the first path that exists, or ``None`` if no candidate matches."""

	for candidate in candidates:
		path = Path(candidate).expanduser()
		if path.exists():
			return path
	return None


def unique_keep_order(items):
	"""Return a de-duplicated list while preserving the original order."""
	return list(dict.fromkeys(items))


def _coerce_raw_cohort(df):
	"""Normalize raw cohort columns to numeric types where possible."""
	frame = df.copy()
	if "Sample_id" in frame.columns:
		frame["Sample_id"] = frame["Sample_id"].astype(str)
	for column in [
		"Age_numeric",
		"ISUP_GG",
		"PSA_log",
		"TMB",
		"Cellularity",
		"Ploidy",
		"PGA_200",
		"Chromothripsis",
		"Driver counts",
	]:
		if column in frame.columns:
			frame[column] = pd.to_numeric(frame[column], errors="coerce")
	if "Chromothripsis" in frame.columns:
		frame["Chromothripsis"] = frame["Chromothripsis"].fillna(0).astype(int)
	return frame

@st.cache_data(show_spinner=False)
def load_cohort():
	"""Return the clinical cohort DataFrame, or a deterministic synthetic fallback."""

	candidate_paths = [CLINICAL_CSV, MODEL_MATRIX_CSV]
	for path in candidate_paths:
		try:
			if not path.exists():
				continue

			df = pd.read_csv(path)
			if {"Sample_id", "ISUP_GG"}.issubset(df.columns):
				return _coerce_raw_cohort(df)
		except FileNotFoundError:
			continue

	return pd.DataFrame(columns=RAW_COHORT_COLUMNS)

@st.cache_data(show_spinner=False)
def load_patient(sample_id):
	"""Return a single patient record by ``Sample_id``."""

	df = load_cohort()
	if df.empty:
		return pd.Series({column: pd.NA for column in RAW_COHORT_COLUMNS})
	row = df[df["Sample_id"] == sample_id]
	if row.empty:
		return df.iloc[0].copy()
	return row.iloc[0].copy()


@st.cache_data(show_spinner=False)
def cohort_kpis(df):
	"""Pre-compute top-line KPIs for the overview panel."""

	return {
		"total": len(df),
		"median_age": round(df["Age_numeric"].median(), 1) if "Age_numeric" in df else 0.0,
		"median_psa_log": round(df["PSA_log"].median(), 3) if "PSA_log" in df else 0.0,
		"median_isup": round(df["ISUP_GG"].median(), 1) if "ISUP_GG" in df else 0.0,
		"median_tmb": round(df["TMB"].median(), 3) if "TMB" in df else 0.0,
	}


__all__ = [
	"CLINICAL_CSV",
	"MODEL_MATRIX_CSV",
	"RANDOM_SEED",
	"cohort_kpis",
	"load_cohort",
	"load_patient",
	"resolve_existing_path",
	"unique_keep_order",
]
