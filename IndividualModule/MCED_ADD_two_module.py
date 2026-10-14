
"""
Module-ablation experiments for the final PCA-AE + REAL graph framework.

Two tests are run in one program:
  1) SINGLE_MODULE:
       use one module only
  2) LEAVE_ONE_OUT:
       remove one module and use all remaining modules

Main downstream classifiers:
  LR / RF / SVM / XGB

"""

import re
import random
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, confusion_matrix, recall_score
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    print("WARNING: XGBoost unavailable; XGB will be skipped.")
    print(e)

import torch
import torch.nn as nn

TASK = "MCED"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = None
# Example:
# DATA_FILE = "elife_data.csv"

# ============================================================
# USER SETTINGS
# ============================================================

# For the final paper, 10 seeds are preferable if runtime allows.
# For a quick first run, change back to [2021,...,2025].
SEEDS = [
    2021, 2022, 2023, 2024, 2025,
    2026, 2027, 2028, 2029, 2030
]

N_SPLITS = 5

# Run both experiments.
# Pair selection is specified in MODULE_PAIRS near the experiment block.
# Set to False for a quicker preliminary run (full baseline then needs separate paired run).
INCLUDE_FULL_REFERENCE = True

# Patient-level predictions may be large across 10 seeds and 5 folds.
SAVE_PATIENT_PREDICTIONS = False

# IMPORTANT:
# Freeze hyperparameters across ALL module experiments.
# Do not choose a different best set for each module.
#
# MCED default below uses the strong full-model setting discussed:
#   m_i = 22, lambda = 10, alpha = 0.125, k = 8
#
# For TOO, edit this to your preferred frozen full-model setting if different.
PARAMETER_SETS = [
    {
        "name": "primary",
        "pca_components": 22,
        "lambda_cls": 10.0,
        "alpha_graph": 0.125,
        "k_graph": 8,
    },
]

# Optional robustness analysis example:
# PARAMETER_SETS = [
#     {"name":"P1", "pca_components":22, "lambda_cls":10.0, "alpha_graph":0.125, "k_graph":8},
#     {"name":"P2", "pca_components":24, "lambda_cls":10.0, "alpha_graph":0.10,  "k_graph":5},
#     {"name":"P3", "pca_components":28, "lambda_cls":5.0,  "alpha_graph":0.10,  "k_graph":8},
# ]

MCED_LABEL_COL = "Label"
TOO_LABEL_COL = "Diagnosis"

HEALTHY_NAMES = {
    "control","healthy","normal","non-cancer","non_cancer",
    "non cancer","no cancer","nocancer","negative","benign","0"
}

META_CANDIDATES = [
    "Sample","sample","SampleID","sampleid","sample_id",
    "ID","id","Patient","patient","PatientID","patientid",
    "Cohort","cohort","Split","split","Set","set",
    "Age","age","Sex","sex","Gender","gender",
    "Stage","stage","Diagnosis","diagnosis",
    "CancerType","cancer_type","Cancer_type"
]

MODULE_RULES = {
    "TMD":[r"^TMD_"],
    "GWMD":[r"^GWMD_"],
    "CNA":[r"^CNA_"],
    "GWFP_long":[r"^GWFP_long_"],
    "GWFP_short":[r"^GWFP_short_"],
    "GWFP_ratio":[r"^GWFP_ratio_"],
    "GWFP_total":[r"^GWFP_total_"],
    "flen":[r"^flen_"],
    "end_motif":[r"^MOTIF_END_", r"^END_MOTIF_", r"^[ACGT]{4}$"],
}

TOP_FEATURES_PER_MODULE = 1200

LATENT_DIM = 32
HIDDEN_DIM = 128
DROPOUT = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 250
PATIENCE = 40
VAL_FRAC = 0.15

GRAPH_SIMILARITY = "spearman"

LR_C = 1.0
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
SVM_C = 1.0
SVM_GAMMA = "scale"

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_seed(*items):
    s = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

# ============================================================
# LOAD DATA
# ============================================================

if DATA_FILE is None:
    csv_files = [
        p for p in sorted(SCRIPT_DIR.glob("*.csv"))
        if "result" not in p.name.lower()
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV found in the same folder.")
    DATA_PATH = csv_files[0]
else:
    DATA_PATH = SCRIPT_DIR / DATA_FILE

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print("Device:", DEVICE)

# ============================================================
# LABELS
# ============================================================

def is_healthy_value(x):
    return str(x).strip().lower() in HEALTHY_NAMES

def make_mced_label(frame):
    if MCED_LABEL_COL in frame.columns:
        raw = frame[MCED_LABEL_COL]

        if pd.api.types.is_numeric_dtype(raw):
            u = sorted(pd.Series(raw).dropna().unique().tolist())
            if set(u) == {0,1}:
                return raw.astype(int).to_numpy()

        vals = raw.astype(str).str.strip()
        uniq = sorted(vals.unique().tolist())

        if len(uniq) == 2:
            if any(u.lower() in HEALTHY_NAMES for u in uniq):
                return np.array(
                    [0 if is_healthy_value(v) else 1 for v in vals],
                    dtype=int
                )
            return LabelEncoder().fit_transform(vals).astype(int)

    if TOO_LABEL_COL not in frame.columns:
        raise ValueError("Cannot construct MCED labels.")

    y = np.array(
        [0 if is_healthy_value(v) else 1 for v in frame[TOO_LABEL_COL]],
        dtype=int
    )
    return y

# ============================================================
# DETECT MODULES
# ============================================================

exclude = set(META_CANDIDATES) | {
    MCED_LABEL_COL, TOO_LABEL_COL,
    "Label","label","Diagnosis","diagnosis",
    "CancerType","cancertype","Cancer_type","cancer_type",
    "Class","class","Group","group"
}

numeric_cols = [
    c for c in df.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
]

modules_all = {}
used = set()

for name, patterns in MODULE_RULES.items():
    cols = [
        c for c in numeric_cols
        if c not in used
        and any(re.search(p, str(c), flags=re.I) for p in patterns)
    ]
    if cols:
        modules_all[name] = cols
        used.update(cols)

if len(modules_all) < 2:
    raise ValueError("Too few modules detected.")

MODULE_NAMES = list(modules_all.keys())

print("Detected modules:")
for name in MODULE_NAMES:
    print(f"  {name:12s}: {len(modules_all[name])} features")

# ============================================================
# TASK-SPECIFIC DATA
# ============================================================

if TASK == "MCED":
    work = df.copy().reset_index(drop=True)
    y = make_mced_label(work)
    label_encoder = None
    n_classes = 2
    OUTDIR = SCRIPT_DIR / "MCED_ADD_two_module_results"
else:
    y_mced = make_mced_label(df)
    work = df.loc[y_mced == 1].copy().reset_index(drop=True)
    work = work[
        ~work[TOO_LABEL_COL].map(is_healthy_value)
    ].copy().reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(
        work[TOO_LABEL_COL].astype(str).str.strip()
    ).astype(int)
    n_classes = len(label_encoder.classes_)
    OUTDIR = SCRIPT_DIR / "ModuleAblation_TOO_results"

OUTDIR.mkdir(exist_ok=True)

meta_cols, seen = [], set()
for c in META_CANDIDATES:
    if c in work.columns and c not in seen:
        meta_cols.append(c)
        seen.add(c)

metadata = (
    work[meta_cols].copy()
    if meta_cols else pd.DataFrame(index=work.index)
)
metadata.insert(0, "Sample_index", np.arange(len(work)))

# ============================================================
# PREPROCESSOR FOR A SELECTED SET OF MODULES
# ============================================================

class ModulePreprocessor:
    def __init__(self, selected_modules, pca_components):
        self.selected_modules = list(selected_modules)
        self.pca_components = int(pca_components)
        self.fitted = {}
        self.actual_components = {}

    def fit(self, frame, y_train):
        for name in self.selected_modules:
            cols = modules_all[name]

            X = (
                frame[cols]
                .replace([np.inf,-np.inf], np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )

            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(X)

            k = min(TOP_FEATURES_PER_MODULE, X.shape[1])
            selector = None

            if k < X.shape[1]:
                selector = SelectKBest(f_classif, k=k)
                X = selector.fit_transform(X, y_train)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            ncomp = min(
                self.pca_components,
                X.shape[1],
                X.shape[0]-1
            )

            pca = PCA(
                n_components=ncomp,
                random_state=0
            )
            pca.fit(X)

            self.actual_components[name] = int(ncomp)
            self.fitted[name] = (
                cols, imp, selector, scaler, pca
            )

        return self

    def transform(self, frame):
        parts = []

        for name in self.selected_modules:
            cols, imp, selector, scaler, pca = self.fitted[name]

            X = (
                frame[cols]
                .replace([np.inf,-np.inf], np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )

            X = imp.transform(X)
            if selector is not None:
                X = selector.transform(X)
            X = scaler.transform(X)
            X = pca.transform(X)

            parts.append(X.astype(np.float32))

        return np.concatenate(parts, axis=1).astype(np.float32)

# ============================================================
# REAL GRAPH
# ============================================================

def build_real_graph(X, k):
    if GRAPH_SIMILARITY.lower() == "spearman":
        Z = np.apply_along_axis(rankdata, 1, X).astype(np.float32)
    else:
        Z = X

    knn = NearestNeighbors(
        n_neighbors=min(k+1, len(Z)),
        metric="correlation",
        n_jobs=-1
    ).fit(Z)

    dist, ind = knn.kneighbors(Z)

    W = np.zeros((len(Z),len(Z)), dtype=np.float32)

    for i in range(len(Z)):
        for d,j in zip(dist[i,1:], ind[i,1:]):
            sim = max(0.0, 1.0-float(d))
            W[i,j] = max(W[i,j], sim)
            W[j,i] = max(W[j,i], sim)

    return W

def graph_loss(z, W):
    degree = W.sum(dim=1)
    L = torch.diag(degree) - W
    return torch.trace(z.T @ L @ z) / (W.sum()+1e-8)

# ============================================================
# AUTOENCODER
# ============================================================

class PCAAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, LATENT_DIM)
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim)
        )

        self.head = nn.Linear(LATENT_DIM, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z), self.head(z)

def classification_loss(logits, yt):
    if TASK == "MCED":
        return nn.BCEWithLogitsLoss()(
            logits.view(-1), yt.float()
        )

    return nn.CrossEntropyLoss()(
        logits, yt.long()
    )

def choose_epoch(X, y_train, W, seed, lam, alpha):
    set_seed(seed)

    tr,va = train_test_split(
        np.arange(len(y_train)),
        test_size=VAL_FRAC,
        random_state=seed,
        stratify=y_train
    )

    output_dim = 1 if TASK=="MCED" else n_classes

    model = PCAAutoencoder(
        X.shape[1], output_dim
    ).to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    Xt = torch.tensor(
        X, dtype=torch.float32, device=DEVICE
    )

    yt = torch.tensor(
        y_train,
        dtype=torch.float32 if TASK=="MCED" else torch.long,
        device=DEVICE
    )

    Wt = torch.tensor(
        W, dtype=torch.float32, device=DEVICE
    )

    tri = torch.tensor(
        tr, dtype=torch.long, device=DEVICE
    )

    vai = torch.tensor(
        va, dtype=torch.long, device=DEVICE
    )

    Wtr = Wt[tri][:,tri]

    best = np.inf
    best_epoch = 1
    wait = 0

    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        opt.zero_grad()

        z,xh,logits = model(Xt[tri])

        rec = ((xh-Xt[tri])**2).mean()
        cls = classification_loss(logits, yt[tri])
        gl = graph_loss(z, Wtr)

        loss = rec + lam*cls + alpha*gl
        loss.backward()
        opt.step()

        model.eval()

        with torch.no_grad():
            z,xh,logits = model(Xt[vai])

            val = (
                ((xh-Xt[vai])**2).mean()
                + lam*classification_loss(logits, yt[vai])
            )

            val = float(val.cpu())

        if val < best-1e-7:
            best = val
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return best_epoch

def fit_ae(Xtr,ytr,Xte,W,seed,lam,alpha):
    ep = choose_epoch(
        Xtr,ytr,W,seed,lam,alpha
    )

    set_seed(seed)

    output_dim = 1 if TASK=="MCED" else n_classes

    model = PCAAutoencoder(
        Xtr.shape[1], output_dim
    ).to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    Xt = torch.tensor(
        Xtr, dtype=torch.float32, device=DEVICE
    )

    yt = torch.tensor(
        ytr,
        dtype=torch.float32 if TASK=="MCED" else torch.long,
        device=DEVICE
    )

    Wt = torch.tensor(
        W, dtype=torch.float32, device=DEVICE
    )

    for _ in range(ep):
        model.train()
        opt.zero_grad()

        z,xh,logits = model(Xt)

        rec = ((xh-Xt)**2).mean()
        cls = classification_loss(logits,yt)
        gl = graph_loss(z,Wt)

        loss = rec + lam*cls + alpha*gl
        loss.backward()
        opt.step()

    model.eval()

    with torch.no_grad():
        ztr = model.encoder(Xt).cpu().numpy()

        zte = model.encoder(
            torch.tensor(
                Xte,
                dtype=torch.float32,
                device=DEVICE
            )
        ).cpu().numpy()

    return ztr,zte,ep

# ============================================================
# DOWNSTREAM METHODS: LR/RF/SVM/XGB ONLY
# ============================================================

def reorder_proba(model, X):
    p = model.predict_proba(X)

    if TASK=="MCED":
        return p[:,list(model.classes_).index(1)]

    out = np.zeros((len(X),n_classes))

    for j,c in enumerate(model.classes_):
        out[:,int(c)] = p[:,j]

    return out

def downstream(ztr,ytr,zte,seed):
    out = {}

    lr = LogisticRegression(
        C=LR_C,
        max_iter=5000,
        class_weight="balanced",
        random_state=seed
    ).fit(ztr,ytr)
    out["Graph_LR"] = reorder_proba(lr,zte)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1
    ).fit(ztr,ytr)
    out["Graph_RF"] = reorder_proba(rf,zte)

    svm = SVC(
        C=SVM_C,
        gamma=SVM_GAMMA,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=seed
    ).fit(ztr,ytr)
    out["Graph_SVM"] = reorder_proba(svm,zte)

    if XGB_OK:
        params = dict(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist"
        )

        if TASK=="MCED":
            params.update(
                objective="binary:logistic",
                eval_metric="logloss"
            )
        else:
            params.update(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss"
            )

        xgb = XGBClassifier(**params).fit(ztr,ytr)
        out["Graph_XGB"] = reorder_proba(xgb,zte)

    return out

# ============================================================
# METRICS
# ============================================================

def evaluate(y_true,score):
    if TASK=="MCED":
        pred = (score>=0.5).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y_true,pred,labels=[0,1]
        ).ravel()

        return {
            "AUC":roc_auc_score(y_true,score),
            "AP":average_precision_score(y_true,score),
            "Accuracy":accuracy_score(y_true,pred),
            "F1":f1_score(y_true,pred,zero_division=0),
            "Sensitivity":tp/(tp+fn+1e-12),
            "Specificity":tn/(tn+fp+1e-12),
        },pred

    pred = np.argmax(score,axis=1)

    Y = label_binarize(
        y_true,
        classes=np.arange(n_classes)
    )

    if n_classes==2 and Y.shape[1]==1:
        Y = np.column_stack(
            [1-Y[:,0],Y[:,0]]
        )

    return {
        "AUC_macro":roc_auc_score(
            y_true,score,average="macro",
            multi_class="ovr",
            labels=np.arange(n_classes)
        ),
        "AUC_weighted":roc_auc_score(
            y_true,score,average="weighted",
            multi_class="ovr",
            labels=np.arange(n_classes)
        ),
        "AP_macro":average_precision_score(
            Y,score,average="macro"
        ),
        "AP_weighted":average_precision_score(
            Y,score,average="weighted"
        ),
        "Accuracy":accuracy_score(y_true,pred),
        "F1_macro":f1_score(
            y_true,pred,average="macro",zero_division=0
        ),
        "F1_weighted":f1_score(
            y_true,pred,average="weighted",zero_division=0
        ),
        "Sensitivity_macro":recall_score(
            y_true,pred,average="macro",zero_division=0
        ),
        "Sensitivity_weighted":recall_score(
            y_true,pred,average="weighted",zero_division=0
        ),
    },pred

# ============================================================
# PAIR EXPERIMENT DEFINITIONS — EDIT THE TUPLES AS NEEDED
# ============================================================

PAIR_MODE = 'ADD'
MODULE_PAIRS = [
    ('end_motif', 'flen'),
    ('end_motif', 'GWMD'),
    ('end_motif', 'TMD'),
    ('end_motif', 'CNA'),
    ('end_motif', 'GWFP_long'),
    ('end_motif', 'GWFP_total'),
    ('flen', 'GWMD'),
    ('GWMD', 'TMD'),
]

if len(MODULE_PAIRS) != 8:
    raise ValueError("Expected 8 pair settings; edit this check if you change the list.")

for pair in MODULE_PAIRS:
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("Every pair must contain two distinct modules: " + str(pair))
    missing = set(pair) - set(MODULE_NAMES)
    if missing:
        raise ValueError("Requested modules not found: " + str(missing) + "; detected=" + str(MODULE_NAMES))

experiments = []
if INCLUDE_FULL_REFERENCE:
    experiments.append({
        "TestMode": "FULL_REFERENCE", "TargetModule": "FULL",
        "SelectedModules": MODULE_NAMES.copy(),
    })

for pair in MODULE_PAIRS:
    selected = (list(pair) if PAIR_MODE == "ADD" else
                [m for m in MODULE_NAMES if m not in pair])
    experiments.append({
        "TestMode": "ADD_TWO" if PAIR_MODE == "ADD" else "REMOVE_TWO",
        "TargetModule": "+".join(pair),
        "SelectedModules": selected,
    })

# ============================================================
# MAIN
# ============================================================

rows = []
predrows = []

for param in PARAMETER_SETS:

    param_name = param["name"]
    pca_components = int(param["pca_components"])
    lambda_cls = float(param["lambda_cls"])
    alpha_graph = float(param["alpha_graph"])
    k_graph = int(param["k_graph"])

    print("\n" + "="*100)
    print("PARAMETER SET:",param_name)
    print(param)
    print("="*100)

    for exp in experiments:

        mode = exp["TestMode"]
        target_module = exp["TargetModule"]
        selected_modules = exp["SelectedModules"]

        print("\n" + "-"*100)
        print(
            f"{TASK} | {mode} | target={target_module} | "
            f"selected={selected_modules}"
        )
        print("-"*100)

        for seed in SEEDS:

            skf = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed
            )

            for fold,(tr,te) in enumerate(
                skf.split(work,y),
                start=1
            ):

                dtr = work.iloc[tr].reset_index(drop=True)
                dte = work.iloc[te].reset_index(drop=True)

                ytr = y[tr]
                yte = y[te]

                prep = ModulePreprocessor(
                    selected_modules,
                    pca_components
                ).fit(dtr,ytr)

                Xtr = prep.transform(dtr)
                Xte = prep.transform(dte)

                W = build_real_graph(
                    Xtr,
                    k_graph
                )

                # Matched initialization/inner split across module configurations.
                run_seed = stable_seed(TASK, param_name, seed, fold)

                ztr,zte,best_epoch = fit_ae(
                    Xtr,ytr,Xte,W,
                    run_seed,
                    lambda_cls,
                    alpha_graph
                )

                scores = downstream(
                    ztr,ytr,zte,seed
                )

                for method,score in scores.items():

                    metrics,pred = evaluate(
                        yte,score
                    )

                    rows.append({
                        "Task":TASK,
                        "ParameterSet":param_name,
                        "TestMode":mode,
                        "TargetModule":target_module,
                        "Pair":target_module,
                        "NSelectedModules":len(selected_modules),
                        "SelectedModules":"|".join(selected_modules),
                        "PCAComponents":pca_components,
                        "LambdaCls":lambda_cls,
                        "AlphaGraph":alpha_graph,
                        "KGraph":k_graph,
                        "Method":method,
                        "Seed":seed,
                        "Fold":fold,
                        "BestEpoch":best_epoch,
                        **metrics
                    })

                    tmp = metadata.iloc[te].copy().reset_index(drop=True)

                    tmp.insert(0,"Task",TASK)
                    tmp.insert(1,"ParameterSet",param_name)
                    tmp.insert(2,"TestMode",mode)
                    tmp.insert(3,"TargetModule",target_module)
                    tmp.insert(4,"Method",method)
                    tmp.insert(5,"Seed",seed)
                    tmp.insert(6,"Fold",fold)

                    if TASK=="MCED":
                        tmp["y_true"] = yte
                        tmp["y_pred"] = pred
                        tmp["score_cancer"] = score

                    else:
                        tmp["y_true_encoded"] = yte
                        tmp["y_true_label"] = label_encoder.inverse_transform(yte)
                        tmp["y_pred_encoded"] = pred
                        tmp["y_pred_label"] = label_encoder.inverse_transform(pred)

                        for j,cls in enumerate(label_encoder.classes_):
                            safe = re.sub(
                                r"[^A-Za-z0-9_]+",
                                "_",
                                str(cls)
                            ).strip("_")

                            tmp[f"prob_{safe}"] = score[:,j]

                    if SAVE_PATIENT_PREDICTIONS:
                        predrows.append(tmp)

# ============================================================
# SAVE — FOLD, SEED, METHOD AND MATCHED-FULL DIFFERENCES
# ============================================================

results = pd.DataFrame(rows)
metric_columns = [c for c in results.columns if c in (
    "AUC", "AP", "Accuracy", "F1", "Sensitivity", "Specificity",
    "AUC_macro", "AUC_weighted", "AP_macro", "AP_weighted",
    "F1_macro", "F1_weighted", "Sensitivity_macro", "Sensitivity_weighted"
)]

results.to_csv(OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_fold_results.csv", index=False)

if SAVE_PATIENT_PREDICTIONS and predrows:
    pd.concat(predrows, ignore_index=True).to_csv(
        OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_patient_predictions.csv", index=False)

seed_group = ["Task", "ParameterSet", "TestMode", "TargetModule", "Method", "Seed"]
seed_summary = results.groupby(seed_group, as_index=False)[metric_columns].mean()
seed_summary.to_csv(OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_seed_summary.csv", index=False)

method_group = ["Task", "ParameterSet", "TestMode", "TargetModule", "Method"]
means = seed_summary.groupby(method_group)[metric_columns].mean().add_suffix("_mean")
stds = seed_summary.groupby(method_group)[metric_columns].std().add_suffix("_std")
summary = pd.concat([means, stds], axis=1).reset_index()
summary.to_csv(OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_method_summary.csv", index=False)

if INCLUDE_FULL_REFERENCE:
    # Merge only matching parameter setting, seed, fold and classifier.
    full = results.loc[results["TestMode"] == "FULL_REFERENCE"].copy()
    tests = results.loc[results["TestMode"] != "FULL_REFERENCE"].copy()
    matched_keys = ["Task", "ParameterSet", "Method", "Seed", "Fold"]
    full = full[matched_keys + metric_columns].rename(
        columns={metric: metric + "_full" for metric in metric_columns})
    paired = tests.merge(full, on=matched_keys, how="left", validate="many_to_one")
    for metric in metric_columns:
        paired["Delta_" + metric] = paired[metric] - paired[metric + "_full"]
    paired.to_csv(OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_paired_fold_deltas.csv", index=False)

    # One average delta per seed, then mean +/- SD across seeds.
    delta_columns = ["Delta_" + metric for metric in metric_columns]
    delta_group = ["Task", "ParameterSet", "TestMode", "TargetModule", "Method", "Seed"]
    seed_deltas = paired.groupby(delta_group, as_index=False)[delta_columns].mean()
    seed_deltas.to_csv(OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_paired_seed_deltas.csv", index=False)
    delta_method_group = ["Task", "ParameterSet", "TestMode", "TargetModule", "Method"]
    delta_mean = seed_deltas.groupby(delta_method_group)[delta_columns].mean().add_suffix("_mean")
    delta_sd = seed_deltas.groupby(delta_method_group)[delta_columns].std().add_suffix("_std")
    pd.concat([delta_mean, delta_sd], axis=1).reset_index().to_csv(
        OUTDIR / f"{TASK}_{PAIR_MODE.lower()}_two_paired_delta_summary.csv", index=False)

main_metric = "AUC_mean" if TASK == "MCED" else "Accuracy_mean"
print("\nDONE: output folder", OUTDIR)
print(summary.sort_values(main_metric, ascending=False).to_string(index=False))
