"""Notebook-derived model preparation and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	matthews_corrcoef,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder

from src.shared.data_utils import load_cohort, resolve_existing_path, unique_keep_order

MODEL_TARGET_MIN_ISUP = 3
TARGET_RECALL = 0.85
RANDOM_SEED = 42


@dataclass
class ModelReport:
	"""Container for the translated classifier notebook outputs."""

	source_label: str
	raw_shape: tuple[int, int]
	clean_shape: tuple[int, int]
	target_min_isup: int
	target_names: list[str]
	feature_counts: dict[str, int]
	class_distribution: dict[int, float]
	metrics: dict[str, float]
	target_recall: float
	optimal_threshold: float
	optimal_threshold_index: int
	optimal_precision: float
	optimal_recall: float
	feature_names: list[str]
	score_column: str
	classification_report_df: pd.DataFrame
	explain_df: pd.DataFrame
	pipeline: ImbPipeline = field(repr=False)
	y_true: np.ndarray = field(repr=False)
	y_probs: np.ndarray = field(repr=False)
	y_pred: np.ndarray = field(repr=False)
	precisions: np.ndarray = field(repr=False)
	recalls: np.ndarray = field(repr=False)
	thresholds: np.ndarray = field(repr=False)
	fpr: np.ndarray = field(repr=False)
	tpr: np.ndarray = field(repr=False)
	confusion_matrix: np.ndarray = field(repr=False)
	has_shap: bool = False


def _candidate_model_paths():
	"""Return candidate paths for the processed feature matrix CSV."""
	dashboard_root = Path(__file__).resolve().parents[2]
	sibling_repo = dashboard_root.parent / "mit808-2026-project-machineup"
	return [
		dashboard_root / "data" / "processed" / "merged_feature_matrix_SA.csv",
		sibling_repo / "data" / "processed" / "merged_feature_matrix_SA.csv",
	]


def _build_demo_model_matrix(cohort):
	"""Build a deterministic fallback matrix from the cohort input."""
	return cohort.copy()


def load_model_source_data(data_path=None):
	"""Load the notebook source matrix or build a synthetic fallback from the cohort."""

	candidates: list[Path] = []
	if data_path is not None:
		candidates.append(Path(data_path))
	candidates.extend(_candidate_model_paths())

	path = resolve_existing_path(*candidates)
	if path is not None:
		try:
			df = pd.read_csv(path)
			if {"Sample_id", "ISUP_GG"}.issubset(df.columns):
				return df, str(path)
		except Exception:
			pass

	cohort = load_cohort()
	return _build_demo_model_matrix(cohort), "Synthetic fallback derived from the dashboard cohort"


def _infer_feature_columns(X):
	"""Split columns into numeric, categorical, and binary feature groups."""
	candidate_categorical = ["GMS_subtype"]
	candidate_numerical = ["Age_numeric", "PSA_log", "TMB", "Cellularity", "Ploidy", "PGA_200", "Driver counts"]

	categorical_cols = [column for column in candidate_categorical if column in X.columns]
	numerical_cols = [column for column in candidate_numerical if column in X.columns]

	remaining_cols = [column for column in X.columns if column not in categorical_cols + numerical_cols]
	binary_cols: list[str] = []
	for column in remaining_cols:
		numeric = pd.to_numeric(X[column], errors="coerce")
		values = set(numeric.dropna().unique().tolist())
		if values and values.issubset({0, 1}):
			binary_cols.append(column)
		elif X[column].dtype == "object" and X[column].dropna().nunique() <= 10:
			categorical_cols.append(column)
		else:
			numerical_cols.append(column)

	categorical_cols = unique_keep_order([column for column in categorical_cols if column in X.columns])
	numerical_cols = unique_keep_order([column for column in numerical_cols if column in X.columns and column not in categorical_cols])
	binary_cols = unique_keep_order([column for column in binary_cols if column in X.columns and column not in categorical_cols and column not in numerical_cols])
	return numerical_cols, categorical_cols, binary_cols


def _build_preprocessor(numerical_cols, categorical_cols, binary_cols):
	"""Create the preprocessing pipeline for mixed feature types."""
	transformers = [
		("num", KNNImputer(n_neighbors=5), numerical_cols),
		(
			"cat",
			ImbPipeline([
				("imputer", SimpleImputer(strategy="most_frequent")),
				("encoder", OrdinalEncoder()),
			]),
			categorical_cols,
		),
		("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
	]
	return ColumnTransformer(transformers=transformers, remainder="drop")


def _safe_k_neighbors(minority_count, max_k=3):
	"""Pick a valid k-neighbors value for SMOTE with small minority classes."""
	if minority_count <= 1:
		raise ValueError("The target must contain at least two minority-class samples to build the notebook model.")
	return max(1, min(max_k, minority_count - 1))


def _build_pipeline(preprocessor, categorical_count, binary_count, minority_count):
	"""Build the imbalanced-learn pipeline with SMOTE and a random forest."""
	num_count = len(preprocessor.transformers[0][2]) if preprocessor.transformers else 0
	cat_indices = list(range(num_count, num_count + categorical_count + binary_count))

	if cat_indices:
		sampler = SMOTENC(
			categorical_features=cat_indices,
			k_neighbors=_safe_k_neighbors(minority_count),
			random_state=RANDOM_SEED,
		)
	else:
		sampler = SMOTE(k_neighbors=_safe_k_neighbors(minority_count), random_state=RANDOM_SEED)

	pipeline = ImbPipeline(
		[
			("preprocessor", preprocessor),
			("smote", sampler),
			(
				"classifier",
				RandomForestClassifier(
					n_estimators=100,
					class_weight="balanced",
					random_state=RANDOM_SEED,
				),
			),
		]
	)
	return pipeline, cat_indices


def _choose_threshold_index(recalls, thresholds, target_recall):
	"""Select the last threshold meeting the target recall, or the best recall."""
	if thresholds.size == 0:
		return 0, 0.5

	valid_indices = np.where(recalls[:-1] >= target_recall)[0]
	if valid_indices.size:
		index = int(valid_indices[-1])
	else:
		index = int(np.argmax(recalls[:-1]))
	return index, float(thresholds[index])


def _classification_report_frame(y_true, y_pred, target_names):
	"""Return a cleaned classification report DataFrame."""
	report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
	df = pd.DataFrame(report).T
	# Clean up the 'accuracy' row which pandas messes up by duplicating the float
	if 'accuracy' in df.index:
		acc_val = df.loc['accuracy', 'f1-score'] # it duplicates everywhere
		df.loc['accuracy', :] = None
		df.loc['accuracy', 'f1-score'] = acc_val
		df.loc['accuracy', 'support'] = df.loc['macro avg', 'support'] # put the total support
	
	# Ensure support is integer (where possible)
	df['support'] = df['support'].fillna(0).astype(int)
	df.index.name = 'Metric'
	return df


def _augment_with_shap(
	pipeline,
	X,
	explain_df,
):
	"""Optionally compute SHAP importances when the dependency is available."""
	try:
		import shap
	except Exception as exc:  # pragma: no cover - optional dependency
		return explain_df, False, f"SHAP not available ({exc})"

	try:
		X_model = pipeline.named_steps["preprocessor"].transform(X)
		if hasattr(X_model, "toarray"):
			X_model = X_model.toarray()

		model_fitted = pipeline.named_steps["classifier"]
		explainer = shap.TreeExplainer(model_fitted)
		shap_values = explainer.shap_values(X_model)

		if isinstance(shap_values, list):
			shap_values_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
		elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
			shap_values_pos = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
		else:
			shap_values_pos = shap_values

		explain_df = explain_df.copy()
		explain_df["shap_mean_abs"] = np.mean(np.abs(shap_values_pos), axis=0)
		shap_scaled = explain_df["shap_mean_abs"] / (explain_df["shap_mean_abs"].max() + 1e-12)
		explain_df["discriminative_score_shap"] = (
			explain_df["rf_importance"] / (explain_df["rf_importance"].max() + 1e-12)
			+ explain_df["perm_importance_mean"].clip(lower=0) / (explain_df["perm_importance_mean"].clip(lower=0).max() + 1e-12)
			+ shap_scaled
		) / 3.0
		return explain_df, True, "SHAP available"
	except Exception as exc:  # pragma: no cover - optional dependency
		return explain_df, False, f"SHAP computation failed ({exc})"


def build_model_report(data_path=None, target_recall=TARGET_RECALL):
	"""Build the notebook-derived model report from the source matrix."""

	source_df, source_label = load_model_source_data(data_path)
	raw_shape = source_df.shape

	df_clean = source_df.copy()
	if "ISUP_GG" not in df_clean.columns:
		raise ValueError("The model source data must contain an 'ISUP_GG' target column.")

	df_clean["ISUP_GG"] = pd.to_numeric(df_clean["ISUP_GG"], errors="coerce")
	df_clean = df_clean.dropna(subset=["ISUP_GG"]).copy()

	if df_clean.empty:
		raise ValueError("The model source data does not contain any valid target values after cleaning.")

	df_clean["Target_High_Risk"] = (df_clean["ISUP_GG"] >= MODEL_TARGET_MIN_ISUP).astype(int)
	target_names = ["Non-Extreme (ISUP 0-3)", "Extreme High (ISUP 4-5)"]

	X = df_clean.drop(columns=["Sample_id", "ISUP_GG", "Target_High_Risk"], errors="ignore")
	y = df_clean["Target_High_Risk"]

	if y.nunique() < 2:
		raise ValueError("The target has a single class after cleaning. Supervised training is not possible.")

	numerical_cols, categorical_cols, binary_cols = _infer_feature_columns(X)
	feature_counts = {"numerical": len(numerical_cols), "categorical": len(categorical_cols), "binary": len(binary_cols)}

	preprocessor = _build_preprocessor(numerical_cols, categorical_cols, binary_cols)
	minority_count = int(y.value_counts().min())
	pipeline, cat_indices = _build_pipeline(preprocessor, len(categorical_cols), len(binary_cols), minority_count)

	min_class_count = int(y.value_counts().min())
	n_splits = max(2, min(5, min_class_count))
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

	y_probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
	precisions, recalls, thresholds = precision_recall_curve(y, y_probs)
	optimal_index, optimal_threshold = _choose_threshold_index(recalls, thresholds, target_recall)
	y_pred = (y_probs >= optimal_threshold).astype(int)

	roc_auc = roc_auc_score(y, y_probs)
	pr_auc = average_precision_score(y, y_probs)
	accuracy = accuracy_score(y, y_pred)
	balanced_accuracy = balanced_accuracy_score(y, y_pred)
	precision_hr = precision_score(y, y_pred, zero_division=0)
	recall_hr = recall_score(y, y_pred, zero_division=0)
	f1_hr = f1_score(y, y_pred, zero_division=0)
	weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
	mcc = matthews_corrcoef(y, y_pred)

	metrics = {
		"roc_auc": float(roc_auc),
		"pr_auc": float(pr_auc),
		"accuracy": float(accuracy),
		"balanced_accuracy": float(balanced_accuracy),
		"precision": float(precision_hr),
		"recall": float(recall_hr),
		"f1": float(f1_hr),
		"weighted_f1": float(weighted_f1),
		"mcc": float(mcc),
	}

	pipeline.fit(X, y)

	feature_names = numerical_cols + categorical_cols + binary_cols
	model_fitted = pipeline.named_steps["classifier"]
	rf_importances = model_fitted.feature_importances_
	if len(rf_importances) != len(feature_names):
		feature_names = [f"feature_{i}" for i in range(len(rf_importances))]

	perm = permutation_importance(
		pipeline,
		X,
		y,
		n_repeats=30,
		random_state=RANDOM_SEED,
		scoring="average_precision",
		n_jobs=-1,
	)

	explain_df = pd.DataFrame(
		{
			"feature": feature_names,
			"rf_importance": rf_importances,
			"perm_importance_mean": perm.importances_mean,
			"perm_importance_std": perm.importances_std,
		}
	)

	rf_scaled = explain_df["rf_importance"] / (explain_df["rf_importance"].max() + 1e-12)
	perm_positive = explain_df["perm_importance_mean"].clip(lower=0)
	perm_scaled = perm_positive / (perm_positive.max() + 1e-12)
	explain_df["discriminative_score"] = 0.5 * rf_scaled + 0.5 * perm_scaled
	score_column = "discriminative_score"
	has_shap = False
	shap_status = "SHAP unavailable"

	explain_df, has_shap, shap_status = _augment_with_shap(pipeline, X, explain_df)
	if has_shap and "discriminative_score_shap" in explain_df.columns:
		score_column = "discriminative_score_shap"

	explain_df = explain_df.sort_values(score_column, ascending=False).reset_index(drop=True)

	confusion = confusion_matrix(y, y_pred)
	report_df = _classification_report_frame(y, y_pred, target_names)

	fpr, tpr, _ = roc_curve(y, y_probs)
	class_distribution = {int(index): float(value) for index, value in y.value_counts(normalize=True).sort_index().items()}

	return ModelReport(
		source_label=source_label,
		raw_shape=raw_shape,
		clean_shape=df_clean.shape,
		target_min_isup=MODEL_TARGET_MIN_ISUP,
		target_names=target_names,
		feature_counts=feature_counts,
		class_distribution=class_distribution,
		metrics=metrics,
		target_recall=target_recall,
		optimal_threshold=optimal_threshold,
		optimal_threshold_index=optimal_index,
		optimal_precision=float(precisions[optimal_index]),
		optimal_recall=float(recalls[optimal_index]),
		feature_names=feature_names,
		score_column=score_column,
		classification_report_df=report_df,
		explain_df=explain_df,
		pipeline=pipeline,
		y_true=y.to_numpy(),
		y_probs=y_probs,
		y_pred=y_pred,
		precisions=precisions,
		recalls=recalls,
		thresholds=thresholds,
		fpr=fpr,
		tpr=tpr,
		confusion_matrix=confusion,
		has_shap=has_shap,
	)


@lru_cache(maxsize=4)
def _cached_model_report(target_recall=TARGET_RECALL):
	"""Cache the model report to avoid recomputing during a session."""
	return build_model_report(target_recall=target_recall)


def get_model_report(force_refresh=False, target_recall=TARGET_RECALL):
	"""Return the cached model report built from the translated notebook pipeline."""

	if force_refresh:
		_cached_model_report.cache_clear()
	return _cached_model_report(target_recall)


__all__ = [
	"MODEL_TARGET_MIN_ISUP",
	"ModelReport",
	"TARGET_RECALL",
	"build_model_report",
	"get_model_report",
	"load_model_source_data",
]
