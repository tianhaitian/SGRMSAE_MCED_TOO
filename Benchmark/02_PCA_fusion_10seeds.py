
"""
PCA intermediate fusion + LR/RF/SVM/XGB for MCED and TOO.

Outputs
-------
The script creates:
    PCA_intermediate_fusion_MCED_TOO_windows_results/
    PCA_intermediate_fusion_MCED_TOO_windows_results.zip

The zip file contains fold-level results, seed summaries, method summaries,
predictions, PCA component information, and plot-friendly long-format CSV files.
"""

import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as e:
    XGBClassifier = None
    XGB_AVAILABLE = False
    print("WARNING: XGBoost could not be imported. PCA_XGB will be skipped.")
    print("Reason:", str(e).split("\n")[0])
    raise ImportError("Install xgboost before the final 16-method benchmark.") from e


# ============================================================
# 1. User settings
# ============================================================

# If None, the script automatically uses the first CSV in the same folder.
# Alternatively, set this directly, e.g. DATA_FILE = "elife_data.csv"
DATA_FILE = "elife_ready_to_use_dataset.csv"  # Edit to your exact input CSV filename

SEEDS = list(range(2021, 2031))
N_SPLITS = 5

# PCA choices: retain this proportion of training-set variance.
# For a faster run, use [0.95].
PCA_COMPONENT_OPTIONS = [0.95]  # Fixed in advance; do not select using test-fold results.

MCED_LABEL_COL = "Label"
TOO_LABEL_COL = "Diagnosis"

ID_CANDIDATES = [
    "Sample", "sample", "SampleID", "sample_id", "ID", "id",
    "Patient", "patient", "PatientID", "patient_id"
]

HEALTHY_NAMES = {
    "healthy", "normal", "control", "noncancer", "non-cancer",
    "non_cancer", "non cancer", "negative", "healthy_control",
    "normal_control", "benign", "no cancer", "nocancer", "0"
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "PCA_intermediate_fusion_MCED_TOO_windows_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
print("XGB available:", XGB_AVAILABLE)
print("First columns:", list(df.columns[:25]), "..." if len(df.columns) > 25 else "")


# ============================================================
# 3. Helper functions
# ============================================================

def detect_id_columns(df_input):
    return [c for c in ID_CANDIDATES if c in df_input.columns]


def is_healthy_value(x):
    return str(x).strip().lower() in HEALTHY_NAMES


def get_feature_matrix(df_input, label_cols_to_exclude):
    """Use all numeric columns as features, excluding labels and sample ID columns."""
    id_cols = detect_id_columns(df_input)
    exclude_cols = set(label_cols_to_exclude + id_cols)

    feature_cols = [
        c for c in df_input.columns
        if c not in exclude_cols and not c.startswith("_original_") and pd.api.types.is_numeric_dtype(df_input[c])
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns detected.")

    X = df_input[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return X, feature_cols, id_cols


def make_mced_label(df_input, mced_col=MCED_LABEL_COL, too_col=TOO_LABEL_COL):
    """Return binary MCED label: 0 = healthy/control, 1 = cancer."""
    if mced_col is not None and mced_col in df_input.columns:
        raw = df_input[mced_col]
        if pd.api.types.is_numeric_dtype(raw):
            y = raw.fillna(-999).astype(int).to_numpy()
            unique_vals = sorted(np.unique(y).tolist())
            if len(unique_vals) == 2:
                if unique_vals != [0, 1]:
                    raise ValueError("MCED Label must use 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {unique_vals}. Will try Diagnosis-based MCED.")
        else:
            vals = raw.astype(str).str.strip()
            unique_vals = sorted(vals.unique().tolist())
            if len(unique_vals) == 2:
                raise ValueError("MCED Label must be numeric 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {unique_vals[:20]}. Will try Diagnosis-based MCED.")

    if too_col not in df_input.columns:
        raise ValueError(f"Cannot create MCED labels because {too_col} was not found.")

    y = np.array([0 if is_healthy_value(v) else 1 for v in df_input[too_col]], dtype=int)
    if len(np.unique(y)) != 2:
        print("Diagnosis values:")
        print(df_input[too_col].value_counts(dropna=False).head(50))
        raise ValueError("MCED labels still have only one class. Please update HEALTHY_NAMES.")
    return y, f"derived_from_{too_col}"


def prepare_too_data(df_input, y_mced, too_label_col=TOO_LABEL_COL):
    """Prepare cancer-only TOO labels."""
    if too_label_col not in df_input.columns:
        raise ValueError(f"Cannot find TOO label column: {too_label_col}")

    df_too = df_input.loc[np.asarray(y_mced) == 1].copy()
    y_raw = df_too[too_label_col].astype(str).str.strip()
    keep_mask = ~y_raw.str.lower().isin(HEALTHY_NAMES)
    df_too = df_too.loc[keep_mask].copy()

    if df_too.shape[0] == 0:
        raise ValueError("After removing healthy/control samples, no TOO samples remain.")

    le = LabelEncoder()
    y = le.fit_transform(df_too[too_label_col].astype(str).str.strip())

    if len(le.classes_) < 2:
        raise ValueError(f"TOO requires at least two cancer types. Detected: {list(le.classes_)}")

    return df_too.reset_index(drop=True), y.astype(int), le


def safe_stratified_kfold(y, requested_splits=N_SPLITS, seed=1):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    n_splits = requested_splits
    if min_count < n_splits:
        raise ValueError(f"A class has only {min_count} samples; cannot run {n_splits} folds.")
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples per class for CV. Class counts:\n{counts}")
    if n_splits < requested_splits:
        print(f"WARNING: using {n_splits}-fold CV because the smallest class has only {min_count} samples.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits


def make_classifiers(seed, task, n_classes):
    """Create LR/RF/SVM/XGB classifiers for binary or multiclass tasks."""
    classifiers = {}

    if task == "binary":
        classifiers["PCA_LR"] = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=5000,
            class_weight="balanced", random_state=seed
        )
    else:
        classifiers["PCA_LR"] = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            class_weight="balanced", random_state=seed
        )

    classifiers["PCA_SVM"] = SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=True,
        class_weight="balanced", random_state=seed
    )

    classifiers["PCA_RF"] = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, class_weight="balanced",
        random_state=seed, n_jobs=-1
    )

    if XGB_AVAILABLE:
        if task == "binary":
            classifiers["PCA_XGB"] = XGBClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=seed, n_jobs=-1, tree_method="hist"
            )
        else:
            classifiers["PCA_XGB"] = XGBClassifier(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", random_state=seed, n_jobs=-1,
                tree_method="hist"
            )

    return classifiers


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "AP": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def multiclass_sensitivity_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    sens_list = []
    spec_list = []
    for k in range(len(labels)):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn + 1e-12))
        spec_list.append(tn / (tn + fp + 1e-12))
    return float(np.mean(sens_list)), float(np.mean(spec_list))


def multiclass_metrics(y_true, y_prob, labels):
    y_pred = np.argmax(y_prob, axis=1)
    y_bin = label_binarize(y_true, classes=list(labels))

    out = {
        "AUC": np.nan, "AUC_macro": np.nan, "AUC_weighted": np.nan,
        "AP": np.nan, "AP_macro": np.nan, "AP_weighted": np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        out["AUC_macro"] = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr", labels=labels)
        out["AUC_weighted"] = roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovr", labels=labels)
        out["AUC"] = out["AUC_macro"]
    except Exception:
        pass

    try:
        out["AP_macro"] = average_precision_score(y_bin, y_prob, average="macro")
        out["AP_weighted"] = average_precision_score(y_bin, y_prob, average="weighted")
        out["AP"] = out["AP_macro"]
    except Exception:
        pass

    sens_macro, spec_macro = multiclass_sensitivity_specificity(y_true, y_pred, labels)
    out["Sensitivity"] = sens_macro
    out["Specificity"] = spec_macro
    out["Sensitivity_macro"] = sens_macro
    out["Specificity_macro"] = spec_macro

    return out


def make_pipeline(pca_option, clf, seed):
    """Imputation, scaling, and PCA are fitted only on the training fold."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_option, random_state=seed)),
        ("clf", clf),
    ])


def summarize_by_seed_and_method(fold_results_df, metric_cols):
    group_cols = ["Task", "PCA_option", "Method", "Seed"]
    return fold_results_df.groupby(group_cols, as_index=False)[metric_cols].mean()


def summarize_by_method(seed_summary_df, metric_cols):
    group_cols = ["Task", "PCA_option", "Method"]
    mean_df = seed_summary_df.groupby(group_cols)[metric_cols].mean().add_suffix("_mean")
    std_df = seed_summary_df.groupby(group_cols)[metric_cols].std().add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()


# ============================================================
# 4. Task runner
# ============================================================

def run_task(task_name, X, y, sample_ids, class_names):
    n_classes = len(np.unique(y))
    task_type = "binary" if task_name == "MCED" else "multiclass"
    labels = np.arange(n_classes)

    all_fold_results = []
    all_predictions = []
    all_pca_info = []

    print("\n" + "#" * 80)
    print(f"Running task: {task_name}")
    print(f"Task type: {task_type}")
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, classes: {n_classes}")
    print(f"Class names: {list(class_names)}")
    print("#" * 80)

    for pca_option in PCA_COMPONENT_OPTIONS:
        print("\n" + "=" * 80)
        print(f"{task_name}: PCA option = {pca_option}")
        print("=" * 80)

        for seed in SEEDS:
            skf, actual_splits = safe_stratified_kfold(y, N_SPLITS, seed)
            classifiers = make_classifiers(seed, task_type, n_classes)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                for method_name, clf in classifiers.items():
                    pipe = make_pipeline(pca_option, clf, seed)
                    pipe.fit(X_train, y_train)
                    y_prob_all = pipe.predict_proba(X_test)

                    if task_type == "binary":
                        y_prob = y_prob_all[:, 1]
                        metrics = binary_metrics(y_test, y_prob)
                        y_pred = (y_prob >= 0.5).astype(int)
                    else:
                        y_prob = y_prob_all
                        metrics = multiclass_metrics(y_test, y_prob, labels)
                        y_pred = np.argmax(y_prob, axis=1)

                    metrics.update({
                        "Task": task_name,
                        "PCA_option": pca_option,
                        "Seed": seed,
                        "Fold": fold,
                        "N_splits": actual_splits,
                        "Method": method_name,
                    })
                    all_fold_results.append(metrics)

                    pca_model = pipe.named_steps["pca"]
                    all_pca_info.append({
                        "Task": task_name,
                        "PCA_option": pca_option,
                        "Seed": seed,
                        "Fold": fold,
                        "Method": method_name,
                        "n_components_selected": int(pca_model.n_components_),
                        "explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
                    })

                    pred_df = pd.DataFrame({
                        "Task": task_name,
                        "SampleID": np.asarray(sample_ids)[test_idx],
                        "TrueLabelCode": y_test,
                        "PredLabelCode": y_pred,
                        "TrueLabel": [class_names[i] for i in y_test],
                        "PredLabel": [class_names[i] for i in y_pred],
                        "PCA_option": pca_option,
                        "Seed": seed,
                        "Fold": fold,
                        "Method": method_name,
                    })

                    if task_type == "binary":
                        pred_df["PredProb_Cancer"] = y_prob
                    else:
                        for k, cls in enumerate(class_names):
                            safe_cls = str(cls).replace("/", "_").replace(" ", "_")
                            pred_df[f"PredProb_{safe_cls}"] = y_prob[:, k]

                    all_predictions.append(pred_df)

                    print(
                        f"{task_name}, PCA={pca_option}, seed={seed}, fold={fold}, "
                        f"{method_name}, AUC={metrics.get('AUC', np.nan):.4f}, "
                        f"AP={metrics.get('AP', np.nan):.4f}, "
                        f"Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, "
                        f"Sen={metrics['Sensitivity']:.4f}, Spe={metrics['Specificity']:.4f}"
                    )

    fold_results_df = pd.DataFrame(all_fold_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    pca_info_df = pd.DataFrame(all_pca_info)

    if task_name == "MCED":
        metric_cols = ["AUC", "AP", "Accuracy", "F1", "Sensitivity", "Specificity"]
    else:
        metric_cols = [
            "AUC", "AUC_macro", "AUC_weighted", "AP", "AP_macro", "AP_weighted",
            "Accuracy", "F1", "F1_macro", "F1_weighted",
            "Sensitivity", "Specificity", "Sensitivity_macro", "Specificity_macro"
        ]

    metric_cols = [c for c in metric_cols if c in fold_results_df.columns]
    seed_summary_df = summarize_by_seed_and_method(fold_results_df, metric_cols)
    method_summary_df = summarize_by_method(seed_summary_df, metric_cols)

    return fold_results_df, predictions_df, pca_info_df, seed_summary_df, method_summary_df


# ============================================================
# 5. Build MCED and TOO data
# ============================================================

label_cols_to_exclude = [c for c in [MCED_LABEL_COL, TOO_LABEL_COL, "Stage", "stage", "Cohort", "cohort", "Split", "split", "Set", "set", "Age", "age", "Gender", "gender", "Sex", "sex", "Batch", "batch", "PatientID", "patientid"] if c in df.columns]

y_mced, mced_source = make_mced_label(df, MCED_LABEL_COL, TOO_LABEL_COL)
X_mced, feature_cols_mced, id_cols_mced = get_feature_matrix(df, label_cols_to_exclude)
sample_ids_mced = df[id_cols_mced[0]].values if len(id_cols_mced) > 0 else np.arange(df.shape[0])
mced_class_names = np.array(["Healthy", "Cancer"])

print("\nMCED label source:", mced_source)
print("MCED feature number:", len(feature_cols_mced))
print("MCED class counts:", dict(zip(*np.unique(y_mced, return_counts=True))))

df_too, y_too, le_too = prepare_too_data(df, y_mced, TOO_LABEL_COL)
X_too, feature_cols_too, id_cols_too = get_feature_matrix(df_too, label_cols_to_exclude)
sample_ids_too = df_too[id_cols_too[0]].values if len(id_cols_too) > 0 else np.arange(df_too.shape[0])

print("\nTOO feature number:", len(feature_cols_too))
print("TOO class mapping:")
for cls, code in zip(le_too.classes_, le_too.transform(le_too.classes_)):
    print(f"  {cls} -> {code}")
print("TOO class counts:", dict(zip(le_too.classes_, np.bincount(y_too))))


# ============================================================
# 6. Run MCED and TOO
# ============================================================

mced_fold, mced_pred, mced_pca, mced_seed_summary, mced_method_summary = run_task(
    "MCED", X_mced, y_mced, sample_ids_mced, mced_class_names
)

too_fold, too_pred, too_pca, too_seed_summary, too_method_summary = run_task(
    "TOO", X_too, y_too, sample_ids_too, le_too.classes_
)


# ============================================================
# 7. Save all results
# ============================================================

mced_fold.to_csv(OUT_DIR / "PCA_MCED_fold_results.csv", index=False)
too_fold.to_csv(OUT_DIR / "PCA_TOO_fold_results.csv", index=False)
pd.concat([mced_fold, too_fold], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "PCA_all_fold_results.csv", index=False
)

mced_pred.to_csv(OUT_DIR / "PCA_MCED_predictions.csv", index=False)
too_pred.to_csv(OUT_DIR / "PCA_TOO_predictions.csv", index=False)
pd.concat([mced_pred, too_pred], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "PCA_all_predictions.csv", index=False
)

mced_pca.to_csv(OUT_DIR / "PCA_MCED_pca_info.csv", index=False)
too_pca.to_csv(OUT_DIR / "PCA_TOO_pca_info.csv", index=False)
pd.concat([mced_pca, too_pca], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "PCA_all_pca_info.csv", index=False
)

mced_seed_summary.to_csv(OUT_DIR / "PCA_MCED_seed_summary.csv", index=False)
too_seed_summary.to_csv(OUT_DIR / "PCA_TOO_seed_summary.csv", index=False)
pd.concat([mced_seed_summary, too_seed_summary], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "PCA_all_seed_summary.csv", index=False
)

mced_method_summary.to_csv(OUT_DIR / "PCA_MCED_method_summary.csv", index=False)
too_method_summary.to_csv(OUT_DIR / "PCA_TOO_method_summary.csv", index=False)
pd.concat([mced_method_summary, too_method_summary], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "PCA_all_method_summary.csv", index=False
)

plot_long = []
all_seed = pd.concat([mced_seed_summary, too_seed_summary], ignore_index=True, sort=False)
id_cols = ["Task", "PCA_option", "Method", "Seed"]
for _, row in all_seed.iterrows():
    for col in all_seed.columns:
        if col in id_cols:
            continue
        if pd.api.types.is_number(row[col]):
            plot_long.append({
                "Task": row["Task"],
                "PCA_option": row["PCA_option"],
                "Method": row["Method"],
                "Seed": row["Seed"],
                "Metric": col,
                "Value": row[col],
            })
pd.DataFrame(plot_long).to_csv(OUT_DIR / "PCA_plot_long_format.csv", index=False)

config = pd.DataFrame([{
    "DATA_PATH": str(DATA_PATH),
    "SEEDS": str(SEEDS),
    "N_SPLITS": N_SPLITS,
    "PCA_COMPONENT_OPTIONS": str(PCA_COMPONENT_OPTIONS),
    "XGB_AVAILABLE": XGB_AVAILABLE,
    "MCED_LABEL_SOURCE": mced_source,
    "TOO_LABEL_COL": TOO_LABEL_COL,
    "N_MCED_FEATURES": len(feature_cols_mced),
    "N_TOO_FEATURES": len(feature_cols_too),
}])
config.to_csv(OUT_DIR / "PCA_run_config.csv", index=False)

zip_path = SCRIPT_DIR / "PCA_intermediate_fusion_MCED_TOO_windows_results.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT_DIR.glob("*.csv")):
        zf.write(p, arcname=p.name)

print("\nAll analyses finished.")
print("Results folder:", OUT_DIR)
print("Zip file:", zip_path)
print("\nMCED method summary:")
print(mced_method_summary.sort_values(["PCA_option", "AUC_mean"], ascending=[True, False]).head(20))
print("\nTOO method summary:")
sort_col = "AUC_mean" if "AUC_mean" in too_method_summary.columns else "Accuracy_mean"
print(too_method_summary.sort_values(["PCA_option", sort_col], ascending=[True, False]).head(20))
