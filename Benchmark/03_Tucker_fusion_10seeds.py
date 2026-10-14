# ============================================================
# 1. Install and import packages
# ============================================================

# !pip -q install numpy pandas scikit-learn xgboost tensorly

import os
import re
import glob
import zipfile
import warnings
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
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix, recall_score
)

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import mode_dot

tl.set_backend("numpy")

# from google.colab import files

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    print("WARNING: XGBoost is not available. TC_XGB will be skipped.")
    print("XGBoost import error:", e)
    raise ImportError("Install xgboost before the final 16-method benchmark.") from e

print("XGB_OK =", XGB_OK)

# ============================================================
# 2. User settings
# ============================================================

# If None, the notebook automatically uses the first CSV file in the current Colab folder.
# Alternatively, set this directly, e.g. DATA_FILE = "/content/elife_data.csv"
DATA_FILE = "elife_ready_to_use_dataset.csv"  # Edit to your exact input CSV filename

# Candidate label columns.
MCED_LABEL_COL = "Label"        # binary cancer/control label if valid
TOO_LABEL_COL  = "Diagnosis"    # cancer type / tissue-of-origin label

# Repeated seeds and folds
SEEDS = list(range(2021, 2031))
# SEEDS=[1]
N_SPLITS = 5

# Tensor settings.
# MAX_FEATURES_PER_MODULE controls memory. Set to None to use all features per module.
MAX_FEATURES_PER_MODULE = 500
TUCKER_RANK_MODULE = 3
TUCKER_RANK_FEATURE = 20
TUCKER_RANK_SAMPLE_MAX = 30
TUCKER_N_ITER_MAX = 60

# Output folder
SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "tensor_fusion_MCED_TOO_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Healthy/control names for creating MCED label from Diagnosis when Label is not valid.
HEALTHY_NAMES = {
    "control", "healthy", "normal", "non-cancer", "non_cancer", "non cancer",
    "no cancer", "nocancer", "negative", "benign", "0"
}

# Columns that should not be used as numerical features if present.
EXCLUDE_COLS_EXTRA = {
    "Sample", "sample", "SampleID", "sampleid", "sample_id", "ID", "id",
    "Patient", "patient", "PatientID", "patientid",
    "Cohort", "cohort", "Split", "split", "Set", "set",
    "Stage", "stage"
}

print("Seeds:", SEEDS)
print("N_SPLITS:", N_SPLITS)
print("MAX_FEATURES_PER_MODULE:", MAX_FEATURES_PER_MODULE)

# ============================================================
# 3. Load data from Colab file folder
# ============================================================

import os
import glob
import pandas as pd

# Explicit data selection prevents accidentally loading a results CSV.
DATA_PATH = SCRIPT_DIR / (DATA_FILE or "elife_ready_to_use_dataset.csv")
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Set DATA_FILE to your eLife CSV filename: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
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
print("First columns:", list(df.columns[:20]), "..." if len(df.columns) > 20 else "")

df.head()



# ============================================================
# 4. Check label columns
# ============================================================

possible_label_cols = [
    "Label", "label", "Diagnosis", "diagnosis", "CancerType", "cancertype",
    "Cancer_type", "cancer_type", "Class", "class", "Group", "group"
]

for col in possible_label_cols:
    if col in df.columns:
        print("\nColumn:", col)
        print(df[col].value_counts(dropna=False).head(30))
        
# ============================================================
# 5. Label utilities
# ============================================================

def is_healthy_value(x):
    s = str(x).strip().lower()
    return s in HEALTHY_NAMES


def make_mced_label(df, mced_col=MCED_LABEL_COL, too_col=TOO_LABEL_COL):
    # Return binary MCED label: 0 = healthy/control, 1 = cancer.
    if mced_col is not None and mced_col in df.columns:
        raw = df[mced_col]
        if pd.api.types.is_numeric_dtype(raw):
            y = raw.fillna(-999).astype(int).to_numpy()
            uniq = sorted(np.unique(y).tolist())
            if len(uniq) == 2:
                if uniq != [0, 1]:
                    raise ValueError("MCED Label must use 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), f"{mced_col}"
            else:
                print(f"WARNING: {mced_col} is not binary. Detected labels: {uniq}. Will try Diagnosis-based MCED.")
        else:
            vals = raw.astype(str).str.strip()
            uniq = sorted(vals.unique().tolist())
            if len(uniq) == 2:
                raise ValueError("MCED Label must be numeric 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), f"{mced_col}"
            else:
                print(f"WARNING: {mced_col} is not binary. Detected labels: {uniq[:20]}. Will try Diagnosis-based MCED.")

    if too_col not in df.columns:
        raise ValueError(f"Cannot create MCED labels because {too_col} was not found.")

    y = np.array([0 if is_healthy_value(v) else 1 for v in df[too_col]], dtype=int)
    uniq = sorted(np.unique(y).tolist())
    if len(uniq) != 2:
        print("Diagnosis values:")
        print(df[too_col].value_counts(dropna=False).head(50))
        raise ValueError("MCED must be binary, but the derived Diagnosis-based MCED label has only one class.")
    return y, f"derived_from_{too_col}"


def make_too_labels(df, y_mced, too_col=TOO_LABEL_COL):
    if too_col not in df.columns:
        raise ValueError(f"Cannot find TOO_LABEL_COL = {too_col}.")

    too_mask = np.asarray(y_mced) == 1
    df_too = df.loc[too_mask].copy()

    y_raw = df_too[too_col].astype(str).str.strip()
    y_raw = y_raw[~y_raw.str.lower().isin(HEALTHY_NAMES)]
    df_too = df_too.loc[y_raw.index].copy()

    encoder = LabelEncoder()
    y_too = encoder.fit_transform(df_too[too_col].astype(str).str.strip())

    if len(encoder.classes_) < 2:
        raise ValueError("TOO needs at least two cancer classes.")

    return df_too, y_too, encoder
    
 # ============================================================
# 6. Feature modules and tensor utilities
# ============================================================

def infer_module_name(col):
    c = str(col)
    lc = c.lower()

    if lc in {x.lower() for x in EXCLUDE_COLS_EXTRA}:
        return None

    if lc.startswith("motif_end_") or lc.startswith("end_motif") or lc.startswith("motif_"):
        return "end_motif"
    if lc.startswith("flen_") or "fragment" in lc or lc.startswith("frag_"):
        return "fragment_length"
    if lc.startswith("gwmd") or re.match(r"^chr\d+_", lc):
        return "gwmd"
    if lc.startswith("cna"):
        return "cna"
    if lc.startswith("gwfp_long"):
        return "gwfp_long"
    if lc.startswith("gwfp_short"):
        return "gwfp_short"
    if lc.startswith("gwfp_ratio"):
        return "gwfp_ratio"
    if lc.startswith("gwfp_total"):
        return "gwfp_total"
    if lc.startswith("tmd"):
        return "tmd"

    return None  # Never include unknown numeric metadata as a genomic feature.


def get_numeric_feature_modules(df_use, label_cols_to_exclude):
    modules = {}
    skipped = []

    for col in df_use.columns:
        if col.startswith("_original_") or col in label_cols_to_exclude:
            skipped.append(col)
            continue
        if col in EXCLUDE_COLS_EXTRA:
            skipped.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(df_use[col]):
            skipped.append(col)
            continue
        module = infer_module_name(col)
        if module is None:
            skipped.append(col)
            continue
        modules.setdefault(module, []).append(col)

    modules = {m: cols for m, cols in modules.items() if len(cols) > 0}
    return modules, skipped


def select_fold_module_features(df_use, modules, train_idx, max_features_per_module=MAX_FEATURES_PER_MODULE):
    selected = {}
    for module, cols in modules.items():
        if max_features_per_module is None or len(cols) <= max_features_per_module:
            selected[module] = list(cols)
            continue

        train_values = df_use.iloc[train_idx][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        variances = np.nanvar(train_values, axis=0)
        variances = np.nan_to_num(variances, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        top_idx = np.argsort(-variances, kind="stable")[:max_features_per_module]
        selected[module] = [cols[i] for i in top_idx]

    return selected


def build_preprocessed_tensors(df_use, selected_modules, train_idx, test_idx):
    active_modules = [m for m, cols in selected_modules.items() if len(cols) > 0]
    max_width = max(len(selected_modules[m]) for m in active_modules)

    Xtr_tensor = np.zeros((len(train_idx), len(active_modules), max_width), dtype=np.float32)
    Xte_tensor = np.zeros((len(test_idx), len(active_modules), max_width), dtype=np.float32)

    preprocessors = {}
    for mi, module in enumerate(active_modules):
        cols = selected_modules[module]
        Xtr_raw = df_use.iloc[train_idx][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        Xte_raw = df_use.iloc[test_idx][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr_imp = imp.fit_transform(Xtr_raw)
        Xte_imp = imp.transform(Xte_raw)
        Xtr_scaled = sc.fit_transform(Xtr_imp)
        Xte_scaled = sc.transform(Xte_imp)

        width = len(cols)
        Xtr_tensor[:, mi, :width] = Xtr_scaled.astype(np.float32)
        Xte_tensor[:, mi, :width] = Xte_scaled.astype(np.float32)
        preprocessors[module] = {"imputer": imp, "scaler": sc, "n_features": width}

    return Xtr_tensor, Xte_tensor, active_modules, max_width, preprocessors


def fit_tucker_and_project(Xtr_tensor, Xte_tensor, seed):
    n_train, n_modules, width = Xtr_tensor.shape
    rank_sample = max(2, min(TUCKER_RANK_SAMPLE_MAX, n_train - 1))
    rank_module = max(1, min(TUCKER_RANK_MODULE, n_modules))
    rank_feature = max(2, min(TUCKER_RANK_FEATURE, width))
    ranks = [rank_sample, rank_module, rank_feature]

    core, factors = tucker(
        Xtr_tensor,
        rank=ranks,
        init="svd",
        n_iter_max=TUCKER_N_ITER_MAX,
        tol=1e-5,
        random_state=seed,
    )

    U_module = factors[1]
    U_feature = factors[2]

    def project(T):
        Z = mode_dot(T, U_module.T, mode=1)
        Z = mode_dot(Z, U_feature.T, mode=2)
        return Z.reshape(T.shape[0], -1)

    Ztr = project(Xtr_tensor)
    Zte = project(Xte_tensor)
    return Ztr, Zte, ranks
    
# ============================================================
# 7. Build MCED and TOO datasets
# ============================================================

y_mced, mced_source = make_mced_label(df, MCED_LABEL_COL, TOO_LABEL_COL)
print("MCED label source:", mced_source)
print("MCED distribution: 0=healthy/control, 1=cancer")
print(pd.Series(y_mced).value_counts().sort_index())

label_cols_to_exclude = {
    MCED_LABEL_COL, TOO_LABEL_COL,
    "Label", "label", "Diagnosis", "diagnosis", "CancerType", "cancertype",
    "Cancer_type", "cancer_type", "Class", "class", "Group", "group"
}
label_cols_to_exclude = {c for c in label_cols_to_exclude if c is not None and c in df.columns}

mced_modules, skipped_cols_mced = get_numeric_feature_modules(df, label_cols_to_exclude)
print("\nMCED modules:")
for m, cols in sorted(mced_modules.items()):
    print(f"  {m}: {len(cols)} features")
print("Total MCED features:", sum(len(cols) for cols in mced_modules.values()))

_df_too, y_too, too_encoder = make_too_labels(df, y_mced, TOO_LABEL_COL)
too_modules, skipped_cols_too = get_numeric_feature_modules(_df_too, label_cols_to_exclude)

print("\nTOO distribution:")
print(pd.Series(_df_too[TOO_LABEL_COL].astype(str).values).value_counts())
print("\nTOO encoded classes:")
for i, c in enumerate(too_encoder.classes_):
    print(i, "=", c)

print("\nTOO modules:")
for m, cols in sorted(too_modules.items()):
    print(f"  {m}: {len(cols)} features")
print("Total TOO features:", sum(len(cols) for cols in too_modules.values()))


# ============================================================
# 8. Metric utilities
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
    labels = np.asarray(labels)
    y_bin = label_binarize(y_true, classes=labels)

    auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)
    ap_macro = average_precision_score(y_bin, y_prob, average="macro")
    ap_weighted = average_precision_score(y_bin, y_prob, average="weighted")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    sens_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    sens_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    spec_macro, spec_weighted = multiclass_specificity(y_true, y_pred, labels)

    return {
        "AUC_macro": auc_macro, "AUC_weighted": auc_weighted,
        "AP_macro": ap_macro, "AP_weighted": ap_weighted,
        "Accuracy": acc,
        "F1_macro": f1_macro, "F1_weighted": f1_weighted,
        "Sensitivity_macro": sens_macro, "Sensitivity_weighted": sens_weighted,
        "Specificity_macro": spec_macro, "Specificity_weighted": spec_weighted,
    }


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
    
# ============================================================
# 9. Classifier definitions
# ============================================================

METHODS = ["TC_LR", "TC_RF", "TC_SVM"]
if XGB_OK:
   METHODS.append("TC_XGB")
# METHODS=["TC_RF"]

print("Methods:", METHODS)


def make_model(method, seed, task, n_classes=None):
    if method == "TC_LR":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if method == "TC_RF":
        clf = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=-1)
        return clf

    if method == "TC_SVM":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if method == "TC_XGB":
        if not XGB_OK:
            raise ImportError("XGBoost is not available.")
        if task == "binary":
            clf = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
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
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
        return clf

    raise ValueError(f"Unknown method: {method}")
    
# ============================================================
# 10. Tensor fold runner
# ============================================================

module_count_rows = []


def run_tensor_cv(task_name, df_use, y, modules, encoder=None):
    rows = []
    pred_rows = []
    prob_rows = []

    y = np.asarray(y)
    class_labels = np.arange(len(np.unique(y)))

    for seed in SEEDS:
        print(f"\n================ {task_name} seed {seed} ================")
        skf, actual_splits = safe_stratified_kfold(y, N_SPLITS, seed)

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            selected_modules = select_fold_module_features(df_use, modules, tr_idx, MAX_FEATURES_PER_MODULE)
            Xtr_tensor, Xte_tensor, active_modules, max_width, preprocessors = build_preprocessed_tensors(
                df_use, selected_modules, tr_idx, te_idx
            )
            Ztr, Zte, ranks = fit_tucker_and_project(Xtr_tensor, Xte_tensor, seed + 1000 * fold)

            for module in active_modules:
                module_count_rows.append({
                    "Task": task_name,
                    "Seed": seed,
                    "Fold": fold,
                    "Module": module,
                    "Selected_features": len(selected_modules[module]),
                    "Max_tensor_width": max_width,
                    "Rank_sample": ranks[0],
                    "Rank_module": ranks[1],
                    "Rank_feature": ranks[2],
                })

            ytr, yte = y[tr_idx], y[te_idx]

            for method in METHODS:
                print(f"{task_name} | seed={seed} fold={fold}/{actual_splits} method={method} tensor_shape={Xtr_tensor.shape} Z_dim={Ztr.shape[1]}")
                model = make_model(
                    method,
                    seed,
                    task="binary" if task_name == "MCED" else "multiclass",
                    n_classes=len(class_labels),
                )
                model.fit(Ztr, ytr)
                y_prob = model.predict_proba(Zte)

                if task_name == "MCED":
                    y_score = y_prob[:, 1]
                    y_pred = (y_score >= 0.5).astype(int)
                    met = binary_metrics(yte, y_score, threshold=0.5)
                    met.update({"Task": task_name, "Method": method, "Seed": seed, "Fold": fold, "N_splits": actual_splits, "Z_dim": Ztr.shape[1]})
                    rows.append(met)

                    for local_i, global_i in enumerate(te_idx):
                        pred_rows.append({
                            "Task": task_name,
                            "Method": method,
                            "Seed": seed,
                            "Fold": fold,
                            "Sample_index": int(global_i),
                            "y_true": int(yte[local_i]),
                            "y_pred": int(y_pred[local_i]),
                            "score_cancer": float(y_score[local_i]),
                        })
                else:
                    y_pred = np.argmax(y_prob, axis=1)
                    met = multiclass_metrics(yte, y_prob, y_pred, class_labels)
                    met.update({"Task": task_name, "Method": method, "Seed": seed, "Fold": fold, "N_splits": actual_splits, "Z_dim": Ztr.shape[1]})
                    rows.append(met)

                    for local_i, global_i in enumerate(te_idx):
                        pred_rows.append({
                            "Task": task_name,
                            "Method": method,
                            "Seed": seed,
                            "Fold": fold,
                            "Sample_index": int(global_i),
                            "y_true": int(yte[local_i]),
                            "y_true_name": str(encoder.classes_[yte[local_i]]) if encoder is not None else str(yte[local_i]),
                            "y_pred": int(y_pred[local_i]),
                            "y_pred_name": str(encoder.classes_[y_pred[local_i]]) if encoder is not None else str(y_pred[local_i]),
                        })
                        for ci, cname in enumerate(encoder.classes_ if encoder is not None else class_labels):
                            prob_rows.append({
                                "Task": task_name,
                                "Method": method,
                                "Seed": seed,
                                "Fold": fold,
                                "Sample_index": int(global_i),
                                "Class_id": int(ci),
                                "Class_name": str(cname),
                                "Probability": float(y_prob[local_i, ci]),
                            })

    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pd.DataFrame(prob_rows)
    
# ============================================================
# 11. Run MCED and TOO
# ============================================================

mced_results, mced_predictions, mced_probabilities = run_tensor_cv(
    task_name="MCED",
    df_use=df,
    y=y_mced,
    modules=mced_modules,
    encoder=None,
)

too_results, too_predictions, too_probabilities = run_tensor_cv(
    task_name="TOO",
    df_use=_df_too,
    y=y_too,
    modules=too_modules,
    encoder=too_encoder,
)

print("\nMCED results shape:", mced_results.shape)
print("TOO results shape:", too_results.shape)

# ============================================================
# 12. Summaries for paper tables and plotting
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
    "Specificity_macro", "Specificity_weighted"
]

mced_seed_summary = summarize_by_seed_and_method(mced_results, mced_metric_cols)
too_seed_summary = summarize_by_seed_and_method(too_results, too_metric_cols)
mced_method_summary = summarize_by_method(mced_seed_summary, mced_metric_cols)
too_method_summary = summarize_by_method(too_seed_summary, too_metric_cols)

all_fold_results = pd.concat([mced_results, too_results], ignore_index=True, sort=False)
all_seed_summary = pd.concat([mced_seed_summary, too_seed_summary], ignore_index=True, sort=False)
all_method_summary = pd.concat([mced_method_summary, too_method_summary], ignore_index=True, sort=False)
all_predictions = pd.concat([mced_predictions, too_predictions], ignore_index=True, sort=False)
all_probabilities = pd.concat([mced_probabilities, too_probabilities], ignore_index=True, sort=False)
module_feature_counts = pd.DataFrame(module_count_rows)

# Long format, convenient for MATLAB/R/Python plotting.
plot_long_parts = []
for _, row in all_seed_summary.iterrows():
    task = row["Task"]
    metric_cols = mced_metric_cols if task == "MCED" else too_metric_cols
    for metric in metric_cols:
        plot_long_parts.append({
            "Task": task,
            "Method": row["Method"],
            "Seed": row["Seed"],
            "Metric": metric,
            "Value": row[metric],
        })
plot_long = pd.DataFrame(plot_long_parts)

print("\nMCED method summary:")
print(mced_method_summary.to_string(index=False))
print("\nTOO method summary:")
print(too_method_summary.to_string(index=False))

# ============================================================
# 13. Save all result files
# ============================================================

# Fold-level results
mced_results.to_csv(OUTDIR / "TC_MCED_fold_results.csv", index=False)
too_results.to_csv(OUTDIR / "TC_TOO_fold_results.csv", index=False)
all_fold_results.to_csv(OUTDIR / "TC_all_fold_results.csv", index=False)

# Seed-level summaries
mced_seed_summary.to_csv(OUTDIR / "TC_MCED_seed_summary.csv", index=False)
too_seed_summary.to_csv(OUTDIR / "TC_TOO_seed_summary.csv", index=False)
all_seed_summary.to_csv(OUTDIR / "TC_all_seed_summary.csv", index=False)

# Method-level mean/std summaries
mced_method_summary.to_csv(OUTDIR / "TC_MCED_method_summary.csv", index=False)
too_method_summary.to_csv(OUTDIR / "TC_TOO_method_summary.csv", index=False)
all_method_summary.to_csv(OUTDIR / "TC_all_method_summary.csv", index=False)

# Predictions and probabilities
mced_predictions.to_csv(OUTDIR / "TC_MCED_predictions.csv", index=False)
too_predictions.to_csv(OUTDIR / "TC_TOO_predictions.csv", index=False)
# Attach stable identifiers to the original cancer-only subset and full dataset.
if ID_COLUMN:
    _mced_ids = df[ID_COLUMN].astype(str).to_numpy()
    _too_ids = df_too[ID_COLUMN].astype(str).to_numpy() if "df_too" in globals() else _df_too[ID_COLUMN].astype(str).to_numpy()
else:
    _mced_ids = df["_original_row_index"].astype(str).to_numpy()
    _too_source = df_too if "df_too" in globals() else _df_too
    _too_ids = _too_source["_original_row_index"].astype(str).to_numpy()
for _pred_df, _ids in [(mced_predictions, _mced_ids), (too_predictions, _too_ids)]:
    _pred_df["SampleID"] = _ids[_pred_df["Sample_index"].to_numpy(dtype=int)]
all_predictions = pd.concat([mced_predictions, too_predictions], ignore_index=True, sort=False)

all_predictions.to_csv(OUTDIR / "TC_all_predictions.csv", index=False)
all_probabilities.to_csv(OUTDIR / "TC_TOO_probabilities_long.csv", index=False)

# Plot-friendly and tensor diagnostic files
plot_long.to_csv(OUTDIR / "TC_plot_long_format.csv", index=False)
module_feature_counts.to_csv(OUTDIR / "TC_module_feature_counts_by_fold.csv", index=False)

config = pd.DataFrame([{
    "DATA_PATH": DATA_PATH,
    "SEEDS": str(SEEDS),
    "N_SPLITS": N_SPLITS,
    "MAX_FEATURES_PER_MODULE": MAX_FEATURES_PER_MODULE,
    "TUCKER_RANK_MODULE": TUCKER_RANK_MODULE,
    "TUCKER_RANK_FEATURE": TUCKER_RANK_FEATURE,
    "TUCKER_RANK_SAMPLE_MAX": TUCKER_RANK_SAMPLE_MAX,
    "TUCKER_N_ITER_MAX": TUCKER_N_ITER_MAX,
    "METHODS": str(METHODS),
    "MCED_LABEL_SOURCE": mced_source,
    "TOO_LABEL_COL": TOO_LABEL_COL,
}])
config.to_csv(OUTDIR / "TC_run_config.csv", index=False)

print("Saved files to:", OUTDIR)
print("\nFiles:")
for p in sorted(OUTDIR.glob("*.csv")):
    print(p.name)
    
# ============================================================
# 14. Zip and download all results
# ============================================================

zip_path = SCRIPT_DIR / "tensor_fusion_MCED_TOO_results.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUTDIR.glob("*.csv")):
        zf.write(p, arcname=p.name)

print("Created:", zip_path)
# files.download(str(zip_path))
print("Results saved to:", zip_path)

        
        
                       
