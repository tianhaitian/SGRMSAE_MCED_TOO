
"""
Classic early fusion + LR/RF/SVM/XGB for MCED and TOO.

Outputs
-------
The script creates:
    classic_early_fusion_MCED_TOO_windows_results/
    classic_early_fusion_MCED_TOO_windows_results.zip

The zip file contains fold-level results, seed summaries, method summaries,
predictions, TOO probabilities, and plot-friendly CSV files.
"""

import glob
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    print("WARNING: XGBoost is not available. EF_XGB will be skipped.")
    print("XGBoost import error:", e)
    raise ImportError("Install xgboost before the final 16-method benchmark.") from e


# ============================================================
# 1. User settings
# ============================================================

# If None, the script automatically uses the first CSV file in the same folder.
# Alternatively, set this directly, e.g. DATA_FILE = "elife_data.csv"
DATA_FILE = "elife_ready_to_use_dataset.csv"  # Edit to your exact input CSV filename

MCED_LABEL_COL = "Label"
TOO_LABEL_COL = "Diagnosis"

SEEDS = list(range(2021, 2031))
N_SPLITS = 5

HEALTHY_NAMES = {
    "control", "healthy", "normal", "non-cancer", "non_cancer", "non cancer",
    "no cancer", "nocancer", "negative", "benign", "0"
}

EXCLUDE_COLS_EXTRA = {
    "Sample", "sample", "SampleID", "sampleid", "sample_id", "ID", "id",
    "Patient", "patient", "PatientID", "patientid",
    "Cohort", "cohort", "Split", "split", "Set", "set",
    "Stage", "stage", "Age", "age", "Gender", "gender", "Sex", "sex", "Batch", "batch"
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "classic_early_fusion_MCED_TOO_windows_results"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data from the same Windows folder
# ============================================================

if DATA_FILE is None:
    csv_files = sorted(SCRIPT_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV file was found in the same folder as this script. "
            "Please put the eLife CSV file and this .py file in the same folder."
        )
    if len(csv_files) > 1:
        print("Multiple CSV files found:")
        for i, p in enumerate(csv_files):
            print(f"{i}: {p.name}")
        print("\nUsing the first CSV file by default. Edit DATA_FILE if this is not correct.")
    DATA_PATH = csv_files[0]
else:
    DATA_PATH = SCRIPT_DIR / DATA_FILE

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot find DATA_PATH: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
if df.columns.duplicated().any():
    raise ValueError("Duplicate column names in the input CSV.")
if df[[MCED_LABEL_COL, TOO_LABEL_COL]].isna().any().any():
    raise ValueError("Missing MCED/TOO labels; inspect and resolve these before the benchmark.")
_ID_CANDIDATES = ["SampleID", "sample_id", "Sample", "PatientID", "patient_id", "Patient", "ID", "id"]
ID_COLUMN = next((c for c in _ID_CANDIDATES if c in df.columns), None)
if ID_COLUMN and df[ID_COLUMN].duplicated().any():
    raise ValueError(f"Duplicate sample IDs in {ID_COLUMN}; resolve patient/sample identity first.")
df["_original_row_index"] = np.arange(len(df))

print("Loaded file:", DATA_PATH)
print("Data shape:", df.shape)
print("XGB_OK =", XGB_OK)
print("First columns:", list(df.columns[:25]), "..." if len(df.columns) > 25 else "")


# ============================================================
# 3. Label and feature utilities
# ============================================================

def is_healthy_value(x):
    return str(x).strip().lower() in HEALTHY_NAMES


def make_mced_label(df_input, mced_col=MCED_LABEL_COL, too_col=TOO_LABEL_COL):
    # Return binary MCED label: 0 = healthy/control, 1 = cancer.
    if mced_col is not None and mced_col in df_input.columns:
        raw = df_input[mced_col]
        if pd.api.types.is_numeric_dtype(raw):
            y = raw.fillna(-999).astype(int).to_numpy()
            uniq = sorted(np.unique(y).tolist())
            if len(uniq) == 2:
                if uniq != [0, 1]:
                    raise ValueError("MCED Label must use 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {uniq}. Will try Diagnosis-based MCED.")
        else:
            vals = raw.astype(str).str.strip()
            uniq = sorted(vals.unique().tolist())
            if len(uniq) == 2:
                raise ValueError("MCED Label must be numeric 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {uniq[:20]}. Will try Diagnosis-based MCED.")

    if too_col not in df_input.columns:
        raise ValueError(f"Cannot create MCED label: {mced_col} is invalid and {too_col} is missing.")

    y = np.array([0 if is_healthy_value(v) else 1 for v in df_input[too_col]], dtype=int)
    if len(np.unique(y)) != 2:
        print("Diagnosis values:")
        print(df_input[too_col].value_counts(dropna=False).head(40))
        raise ValueError("Created MCED label still has only one class. Please update HEALTHY_NAMES or label columns.")
    return y, f"created_from_{too_col}"


def make_too_labels(df_input, y_mced, too_col=TOO_LABEL_COL):
    # Return cancer-only dataframe and encoded TOO labels.
    if too_col not in df_input.columns:
        raise ValueError(f"TOO_LABEL_COL = {too_col} not found in dataframe.")
    cancer_mask = np.asarray(y_mced) == 1
    df_cancer = df_input.loc[cancer_mask].copy()
    nonhealthy_mask = ~df_cancer[too_col].map(is_healthy_value)
    df_cancer = df_cancer.loc[nonhealthy_mask].copy()
    le = LabelEncoder()
    y_too = le.fit_transform(df_cancer[too_col].astype(str).str.strip())
    if len(np.unique(y_too)) < 2:
        raise ValueError("TOO requires at least two cancer classes. Please check Diagnosis values.")
    return df_cancer.reset_index(drop=True), y_too.astype(int), le


def get_feature_matrix(df_input, exclude_cols):
    # Use all numeric columns except labels, IDs, cohort/split columns, and extra excluded columns.
    exclude = set(exclude_cols) | set(EXCLUDE_COLS_EXTRA)
    feature_cols, skipped_cols = [], []
    for col in df_input.columns:
        if col in exclude or col.startswith("_original_"):
            skipped_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df_input[col]):
            feature_cols.append(col)
        else:
            skipped_cols.append(col)
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns detected. Please check your data file.")
    X = df_input[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return X, feature_cols, skipped_cols


def safe_stratified_kfold(y, n_splits=N_SPLITS, seed=1):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    k = n_splits
    if min_count < k:
        raise ValueError(f"A class has only {min_count} samples; cannot run {k} folds.")
    if k < 2:
        raise ValueError(f"Not enough samples in at least one class for stratified CV. Counts:\n{counts}")
    if k < n_splits:
        print(f"WARNING: using {k}-fold CV because the smallest class has only {min_count} samples.")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), k


# ============================================================
# 4. Build MCED and TOO datasets
# ============================================================

y_mced, mced_source = make_mced_label(df, MCED_LABEL_COL, TOO_LABEL_COL)
print("\nMCED label source:", mced_source)
print("MCED distribution: 0=healthy/control, 1=cancer")
print(pd.Series(y_mced).value_counts().sort_index())

label_cols_to_exclude = {
    MCED_LABEL_COL, TOO_LABEL_COL,
    "Label", "label", "Diagnosis", "diagnosis", "CancerType", "cancertype",
    "Cancer_type", "cancer_type", "Class", "class", "Group", "group"
}
label_cols_to_exclude = {c for c in label_cols_to_exclude if c is not None and c in df.columns}

X_mced, feature_cols_mced, skipped_cols_mced = get_feature_matrix(df, label_cols_to_exclude)
print("\nMCED feature matrix:", X_mced.shape)
print("Number of MCED features:", len(feature_cols_mced))

df_too, y_too, too_encoder = make_too_labels(df, y_mced, TOO_LABEL_COL)
X_too, feature_cols_too, skipped_cols_too = get_feature_matrix(df_too, label_cols_to_exclude)

print("\nTOO distribution:")
print(pd.Series(df_too[TOO_LABEL_COL].astype(str).values).value_counts())
print("\nTOO encoded classes:")
for i, c in enumerate(too_encoder.classes_):
    print(i, "=", c)
print("\nTOO feature matrix:", X_too.shape)
print("Number of TOO features:", len(feature_cols_too))


# ============================================================
# 5. Metric utilities
# ============================================================

def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return {
        "AUC": auc, "AP": ap, "Accuracy": acc, "F1": f1,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def multiclass_specificity(y_true, y_pred, labels):
    specs = []
    supports = []
    for lab in labels:
        y_bin_true = (y_true == lab).astype(int)
        y_bin_pred = (y_pred == lab).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp + 1e-12))
        supports.append(np.sum(y_true == lab))
    specs = np.asarray(specs, dtype=float)
    supports = np.asarray(supports, dtype=float)
    return float(np.mean(specs)), float(np.average(specs, weights=supports))


def multiclass_metrics(y_true, y_prob, y_pred, labels):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    labels = np.asarray(labels).astype(int)
    y_bin = label_binarize(y_true, classes=labels)

    out = {
        "AUC_macro": np.nan, "AUC_weighted": np.nan,
        "AP_macro": np.nan, "AP_weighted": np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Sensitivity_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Sensitivity_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Specificity_macro": np.nan,
        "Specificity_weighted": np.nan,
    }

    try:
        out["AUC_macro"] = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr", labels=labels)
        out["AUC_weighted"] = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovr", labels=labels)
    except Exception:
        pass
    try:
        out["AP_macro"] = average_precision_score(y_bin, y_prob, average="macro")
        out["AP_weighted"] = average_precision_score(y_bin, y_prob, average="weighted")
    except Exception:
        pass
    try:
        spec_macro, spec_weighted = multiclass_specificity(y_true, y_pred, labels)
        out["Specificity_macro"] = spec_macro
        out["Specificity_weighted"] = spec_weighted
    except Exception:
        pass
    return out


# ============================================================
# 6. Classifier definitions
# ============================================================

def make_model(method, seed, task, n_classes=None):
    if method == "EF_LR":
        clf = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if method == "EF_RF":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    if method == "EF_SVM":
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if method == "EF_XGB":
        if not XGB_OK:
            raise ImportError("XGBoost is not available.")
        if task == "binary":
            clf = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
        else:
            clf = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    raise ValueError(f"Unknown method: {method}")


METHODS = ["EF_LR", "EF_RF", "EF_SVM"] + (["EF_XGB"] if XGB_OK else [])
print("\nMethods:", METHODS)


# ============================================================
# 7. Run MCED
# ============================================================

mced_rows = []
mced_pred_rows = []

for seed in SEEDS:
    print(f"\n================ MCED seed {seed} ================")
    skf, actual_splits = safe_stratified_kfold(y_mced, N_SPLITS, seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_mced, y_mced), start=1):
        Xtr, Xte = X_mced[tr_idx], X_mced[te_idx]
        ytr, yte = y_mced[tr_idx], y_mced[te_idx]
        for method in METHODS:
            print(f"MCED | seed={seed} fold={fold}/{actual_splits} method={method}")
            model = make_model(method, seed, task="binary")
            model.fit(Xtr, ytr)
            y_score = model.predict_proba(Xte)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            met = binary_metrics(yte, y_score, threshold=0.5)
            met.update({"Task": "MCED", "Method": method, "Seed": seed, "Fold": fold, "N_splits": actual_splits})
            mced_rows.append(met)
            for local_i, global_i in enumerate(te_idx):
                mced_pred_rows.append({
                    "Task": "MCED", "Method": method, "Seed": seed, "Fold": fold,
                    "Sample_index": int(global_i), "SampleID": str(df.iloc[global_i][ID_COLUMN]) if ID_COLUMN else str(global_i), "y_true": int(yte[local_i]),
                    "y_pred": int(y_pred[local_i]), "score_cancer": float(y_score[local_i]),
                })

mced_results = pd.DataFrame(mced_rows)
mced_predictions = pd.DataFrame(mced_pred_rows)


# ============================================================
# 8. Run TOO
# ============================================================

too_rows = []
too_pred_rows = []
too_prob_rows = []
labels_too = np.arange(len(too_encoder.classes_))

for seed in SEEDS:
    print(f"\n================ TOO seed {seed} ================")
    skf, actual_splits = safe_stratified_kfold(y_too, N_SPLITS, seed)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_too, y_too), start=1):
        Xtr, Xte = X_too[tr_idx], X_too[te_idx]
        ytr, yte = y_too[tr_idx], y_too[te_idx]
        for method in METHODS:
            print(f"TOO | seed={seed} fold={fold}/{actual_splits} method={method}")
            model = make_model(method, seed, task="multiclass", n_classes=len(labels_too))
            model.fit(Xtr, ytr)
            y_prob = model.predict_proba(Xte)
            y_pred = np.argmax(y_prob, axis=1)
            met = multiclass_metrics(yte, y_prob, y_pred, labels_too)
            met.update({"Task": "TOO", "Method": method, "Seed": seed, "Fold": fold, "N_splits": actual_splits})
            too_rows.append(met)
            for local_i, global_i in enumerate(te_idx):
                base_row = {
                    "Task": "TOO", "Method": method, "Seed": seed, "Fold": fold,
                    "Cancer_sample_index": int(global_i),
                    "SampleID": str(df_too.iloc[global_i][ID_COLUMN]) if ID_COLUMN else str(df_too.iloc[global_i]["_original_row_index"]),
                    "y_true_encoded": int(yte[local_i]),
                    "y_true_label": str(too_encoder.classes_[yte[local_i]]),
                    "y_pred_encoded": int(y_pred[local_i]),
                    "y_pred_label": str(too_encoder.classes_[y_pred[local_i]]),
                }
                too_pred_rows.append(base_row)
                for k, cls_name in enumerate(too_encoder.classes_):
                    too_prob_rows.append({
                        **base_row,
                        "Class_id": int(k),
                        "Class_name": str(cls_name),
                        "Probability": float(y_prob[local_i, k]),
                    })

too_results = pd.DataFrame(too_rows)
too_predictions = pd.DataFrame(too_pred_rows)
too_probabilities = pd.DataFrame(too_prob_rows)


# ============================================================
# 9. Summaries for paper tables and plotting
# ============================================================

def summarize_by_seed_and_method(df_results, metric_cols):
    return df_results.groupby(["Task", "Method", "Seed"], as_index=False)[metric_cols].mean()


def summarize_by_method(seed_summary, metric_cols):
    mean_df = seed_summary.groupby(["Task", "Method"])[metric_cols].mean().add_suffix("_mean")
    std_df = seed_summary.groupby(["Task", "Method"])[metric_cols].std().add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()


mced_metric_cols = ["AUC", "AP", "Accuracy", "F1", "Sensitivity", "Specificity"]
too_metric_cols = [
    "AUC_macro", "AUC_weighted", "AP_macro", "AP_weighted",
    "Accuracy", "F1_macro", "F1_weighted",
    "Sensitivity_macro", "Sensitivity_weighted",
    "Specificity_macro", "Specificity_weighted",
]

mced_seed_summary = summarize_by_seed_and_method(mced_results, mced_metric_cols)
too_seed_summary = summarize_by_seed_and_method(too_results, too_metric_cols)
mced_method_summary = summarize_by_method(mced_seed_summary, mced_metric_cols)
too_method_summary = summarize_by_method(too_seed_summary, too_metric_cols)

all_fold_results = pd.concat([mced_results, too_results], ignore_index=True, sort=False)
all_seed_summary = pd.concat([mced_seed_summary, too_seed_summary], ignore_index=True, sort=False)
all_method_summary = pd.concat([mced_method_summary, too_method_summary], ignore_index=True, sort=False)
all_predictions = pd.concat([mced_predictions, too_predictions], ignore_index=True, sort=False)

mced_plot_long = mced_seed_summary.melt(
    id_vars=["Task", "Method", "Seed"],
    value_vars=mced_metric_cols,
    var_name="Metric",
    value_name="Value",
)
too_plot_long = too_seed_summary.melt(
    id_vars=["Task", "Method", "Seed"],
    value_vars=too_metric_cols,
    var_name="Metric",
    value_name="Value",
)
plot_long = pd.concat([mced_plot_long, too_plot_long], ignore_index=True, sort=False)


# ============================================================
# 10. Save all result files
# ============================================================

mced_results.to_csv(OUTDIR / "EF_MCED_fold_results.csv", index=False)
too_results.to_csv(OUTDIR / "EF_TOO_fold_results.csv", index=False)
all_fold_results.to_csv(OUTDIR / "EF_all_fold_results.csv", index=False)

mced_seed_summary.to_csv(OUTDIR / "EF_MCED_seed_summary.csv", index=False)
too_seed_summary.to_csv(OUTDIR / "EF_TOO_seed_summary.csv", index=False)
all_seed_summary.to_csv(OUTDIR / "EF_all_seed_summary.csv", index=False)

mced_method_summary.to_csv(OUTDIR / "EF_MCED_method_summary.csv", index=False)
too_method_summary.to_csv(OUTDIR / "EF_TOO_method_summary.csv", index=False)
all_method_summary.to_csv(OUTDIR / "EF_all_method_summary.csv", index=False)

mced_predictions.to_csv(OUTDIR / "EF_MCED_predictions.csv", index=False)
too_predictions.to_csv(OUTDIR / "EF_TOO_predictions.csv", index=False)
all_predictions.to_csv(OUTDIR / "EF_all_predictions.csv", index=False)
too_probabilities.to_csv(OUTDIR / "EF_TOO_probabilities_long.csv", index=False)
plot_long.to_csv(OUTDIR / "EF_plot_long_format.csv", index=False)

config = pd.DataFrame([{
    "DATA_PATH": str(DATA_PATH),
    "SEEDS": str(SEEDS),
    "N_SPLITS": N_SPLITS,
    "METHODS": str(METHODS),
    "MCED_LABEL_SOURCE": mced_source,
    "TOO_LABEL_COL": TOO_LABEL_COL,
    "N_MCED_FEATURES": len(feature_cols_mced),
    "N_TOO_FEATURES": len(feature_cols_too),
}])
config.to_csv(OUTDIR / "EF_run_config.csv", index=False)


# ============================================================
# 11. Zip results locally
# ============================================================

zip_path = SCRIPT_DIR / "classic_early_fusion_MCED_TOO_windows_results.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUTDIR.glob("*.csv")):
        zf.write(path, arcname=path.name)

print("\nDone.")
print("Results folder:", OUTDIR)
print("Zip file:", zip_path)
print("\nMCED method summary:")
print(mced_method_summary)
print("\nTOO method summary:")
print(too_method_summary)
