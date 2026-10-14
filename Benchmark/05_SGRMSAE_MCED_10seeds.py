
"""
PCA-AE + graph with fixed per-module PCA dimension.

Graph conditions:
    NO_GRAPH
    RANDOM_GRAPH
    REAL_GRAPH

All samples are evaluated using repeated stratified 5-fold CV.
All preprocessing is fitted inside the training fold only.
"""

import re, random, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
    print("WARNING: XGBoost unavailable; Graph_XGB will be skipped.")
    print(e)

import torch
import torch.nn as nn

TASK = "MCED"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = None

# ============================================================
# USER-EDITABLE VECTORS
# ============================================================

SEEDS = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
N_SPLITS = 5

# Fixed number of PCA components PER MODULE.
# PCA_COMPONENT_VALUES = [24]
# Examples:
# PCA_COMPONENT_VALUES = [16, 24, 32]
PCA_COMPONENT_VALUES = [22]

LAMBDA_CLS_VALUES = [10.0]
# Example: [5.0, 10.0, 20.0]

ALPHA_GRAPH_VALUES = [0.125]
# Example: [0.01, 0.03, 0.05, 0.07, 0.10]

K_GRAPH_VALUES = [8]
# Example: [5, 7, 8, 9]

# GRAPH_CONDITIONS = ["NO_GRAPH", "RANDOM_GRAPH", "REAL_GRAPH"]
GRAPH_CONDITIONS = ["REAL_GRAPH"]
# ============================================================
# OTHER SETTINGS
# ============================================================

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
    "end_motif":[r"^MOTIF_END_",r"^END_MOTIF_",r"^[ACGT]{4}$"],
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

INTERNAL_NN_K = 5
LR_C = 1.0
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
SVM_C = 1.0
SVM_GAMMA = "scale"
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    csv_files = [p for p in sorted(SCRIPT_DIR.glob("*.csv")) if "result" not in p.name.lower()]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in same folder.")
    DATA_PATH = csv_files[0]
else:
    DATA_PATH = SCRIPT_DIR / DATA_FILE

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded:", DATA_PATH)
print("Data shape:", df.shape)
print("Device:", DEVICE)

def is_healthy_value(x):
    return str(x).strip().lower() in HEALTHY_NAMES

def make_mced_label(frame):
    if MCED_LABEL_COL in frame.columns:
        raw = frame[MCED_LABEL_COL]
        if pd.api.types.is_numeric_dtype(raw):
            vals = sorted(pd.Series(raw).dropna().unique().tolist())
            if set(vals) == {0, 1}:
                return raw.astype(int).to_numpy()

        vals = raw.astype(str).str.strip()
        uniq = sorted(vals.unique().tolist())
        if len(uniq) == 2:
            if any(u.lower() in HEALTHY_NAMES for u in uniq):
                return np.array([0 if is_healthy_value(v) else 1 for v in vals], dtype=int)
            return LabelEncoder().fit_transform(vals).astype(int)

    if TOO_LABEL_COL not in frame.columns:
        raise ValueError("Cannot create MCED labels.")

    y = np.array([0 if is_healthy_value(v) else 1 for v in frame[TOO_LABEL_COL]], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("MCED label does not contain two classes.")
    return y

exclude = set(META_CANDIDATES) | {
    MCED_LABEL_COL, TOO_LABEL_COL, "Label","label","Diagnosis","diagnosis",
    "CancerType","cancertype","Cancer_type","cancer_type","Class","class","Group","group"
}

numeric_cols = [
    c for c in df.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
]

modules = {}
used = set()
for name, pats in MODULE_RULES.items():
    cols = [
        c for c in numeric_cols
        if c not in used and any(re.search(p, str(c), flags=re.I) for p in pats)
    ]
    if cols:
        modules[name] = cols
        used.update(cols)

if len(modules) < 2:
    raise ValueError("Too few modules detected. Check MODULE_RULES.")

print("Detected modules:", {k:len(v) for k,v in modules.items()})

# ============================================================
# PREPROCESSING
# ============================================================

class ModulePreprocessor:
    def __init__(self, pca_components):
        self.pca_components = int(pca_components)
        self.fitted = {}
        self.actual_components = {}

    def fit(self, frame, y):
        for name, cols in modules.items():
            X = (
                frame[cols]
                .replace([np.inf, -np.inf], np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )

            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(X)

            k = min(TOP_FEATURES_PER_MODULE, X.shape[1])
            sel = None
            if k < X.shape[1]:
                sel = SelectKBest(f_classif, k=k)
                X = sel.fit_transform(X, y)

            sc = StandardScaler()
            X = sc.fit_transform(X)

            ncomp = min(self.pca_components, X.shape[1], X.shape[0]-1)
            if ncomp < 1:
                raise ValueError(f"Module {name} has insufficient dimensions for PCA.")

            pca = PCA(n_components=ncomp, random_state=0)
            pca.fit(X)

            self.actual_components[name] = int(ncomp)
            self.fitted[name] = (cols, imp, sel, sc, pca)

        return self

    def transform(self, frame):
        parts = []
        for name, (cols, imp, sel, sc, pca) in self.fitted.items():
            X = (
                frame[cols]
                .replace([np.inf, -np.inf], np.nan)
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float32)
            )
            X = imp.transform(X)
            if sel is not None:
                X = sel.transform(X)
            X = sc.transform(X)
            X = pca.transform(X)
            parts.append(X.astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

# ============================================================
# GRAPH
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
    W = np.zeros((len(Z), len(Z)), dtype=np.float32)

    for i in range(len(Z)):
        for d, j in zip(dist[i,1:], ind[i,1:]):
            w = max(0.0, 1.0-float(d))
            W[i,j] = max(W[i,j], w)
            W[j,i] = max(W[j,i], w)
    return W

def make_random_graph(W_real, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(W_real))
    return W_real[np.ix_(perm, perm)].copy()

def graph_loss(z, W):
    D = W.sum(dim=1)
    L = torch.diag(D) - W
    return torch.trace(z.T @ L @ z) / (W.sum() + 1e-8)

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

# ============================================================
# TASK SETUP
# ============================================================

if TASK == "MCED":
    work = df.copy().reset_index(drop=True)
    y = make_mced_label(work)
    label_encoder = None
    n_classes = 2
    OUTDIR = SCRIPT_DIR / "FixedPCA_ThreeGraphs_MCED_results"
else:
    ym = make_mced_label(df)
    work = df.loc[ym == 1].copy().reset_index(drop=True)
    work = work[~work[TOO_LABEL_COL].map(is_healthy_value)].copy().reset_index(drop=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(work[TOO_LABEL_COL].astype(str).str.strip()).astype(int)
    n_classes = len(label_encoder.classes_)
    OUTDIR = SCRIPT_DIR / "FixedPCA_ThreeGraphs_TOO_results"

OUTDIR.mkdir(exist_ok=True)

meta_cols, seen = [], set()
for c in META_CANDIDATES:
    if c in work.columns and c not in seen:
        meta_cols.append(c)
        seen.add(c)

metadata = work[meta_cols].copy() if meta_cols else pd.DataFrame(index=work.index)
metadata.insert(0, "Sample_index", np.arange(len(work)))

def classification_loss(logits, y_tensor):
    if TASK == "MCED":
        return nn.BCEWithLogitsLoss()(logits.view(-1), y_tensor.float())
    return nn.CrossEntropyLoss()(logits, y_tensor.long())

def choose_epoch(X, y_train, W, seed, lam, alpha):
    set_seed(seed)
    tr, va = train_test_split(
        np.arange(len(y_train)),
        test_size=VAL_FRAC,
        random_state=seed,
        stratify=y_train
    )

    model = PCAAutoencoder(X.shape[1], 1 if TASK=="MCED" else n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32 if TASK=="MCED" else torch.long, device=DEVICE)
    Wt = torch.tensor(W, dtype=torch.float32, device=DEVICE)

    tri = torch.tensor(tr, dtype=torch.long, device=DEVICE)
    vai = torch.tensor(va, dtype=torch.long, device=DEVICE)
    Wtr = Wt[tri][:,tri]

    best, best_ep, wait = np.inf, 1, 0

    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        opt.zero_grad()

        z, xh, logits = model(Xt[tri])
        rec = ((xh-Xt[tri])**2).mean()
        cls = classification_loss(logits, yt[tri])
        gl = graph_loss(z, Wtr) if alpha > 0 else torch.tensor(0.0, device=DEVICE)

        loss = rec + lam*cls + alpha*gl
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z, xh, logits = model(Xt[vai])
            v = ((xh-Xt[vai])**2).mean() + lam*classification_loss(logits, yt[vai])
            v = float(v.cpu())

        if v < best - 1e-7:
            best, best_ep, wait = v, ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return best_ep

def fit_ae(Xtr, ytr, Xte, W, seed, lam, alpha):
    ep = choose_epoch(Xtr, ytr, W, seed, lam, alpha)
    set_seed(seed)

    model = PCAAutoencoder(Xtr.shape[1], 1 if TASK=="MCED" else n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr, dtype=torch.float32 if TASK=="MCED" else torch.long, device=DEVICE)
    Wt = torch.tensor(W, dtype=torch.float32, device=DEVICE)

    for _ in range(ep):
        model.train()
        opt.zero_grad()
        z, xh, logits = model(Xt)
        rec = ((xh-Xt)**2).mean()
        cls = classification_loss(logits, yt)
        gl = graph_loss(z, Wt) if alpha > 0 else torch.tensor(0.0, device=DEVICE)
        loss = rec + lam*cls + alpha*gl
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ztr = model.encoder(Xt).cpu().numpy()
        zte = model.encoder(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    return ztr, zte, ep

def reorder_proba(model, X):
    p = model.predict_proba(X)
    if TASK == "MCED":
        return p[:, list(model.classes_).index(1)]

    out = np.zeros((len(X), n_classes))
    for j,c in enumerate(model.classes_):
        out[:, int(c)] = p[:,j]
    return out

def downstream(ztr, ytr, zte, seed):
    out = {}

    knn = KNeighborsClassifier(n_neighbors=INTERNAL_NN_K, weights="distance").fit(ztr, ytr)
    out["Graph_Internal_NN"] = reorder_proba(knn, zte)

    lr = LogisticRegression(C=LR_C, max_iter=5000, class_weight="balanced", random_state=seed).fit(ztr, ytr)
    out["Graph_LR"] = reorder_proba(lr, zte)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1
    ).fit(ztr, ytr)
    out["Graph_RF"] = reorder_proba(rf, zte)

    svm = SVC(
        C=SVM_C, gamma=SVM_GAMMA, kernel="rbf",
        probability=True, class_weight="balanced", random_state=seed
    ).fit(ztr, ytr)
    out["Graph_SVM"] = reorder_proba(svm, zte)

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
        if TASK == "MCED":
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")

        xgb = XGBClassifier(**params).fit(ztr, ytr)
        out["Graph_XGB"] = reorder_proba(xgb, zte)

    return out

def evaluate(y_true, score):
    if TASK == "MCED":
        pred = (score >= 0.5).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        return {
            "AUC": roc_auc_score(y_true, score),
            "AP": average_precision_score(y_true, score),
            "Accuracy": accuracy_score(y_true, pred),
            "F1": f1_score(y_true, pred, zero_division=0),
            "Sensitivity": tp/(tp+fn+1e-12),
            "Specificity": tn/(tn+fp+1e-12)
        }, pred

    pred = np.argmax(score, axis=1)
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    if n_classes == 2 and Y.shape[1] == 1:
        Y = np.column_stack([1-Y[:,0], Y[:,0]])

    return {
        "AUC_macro": roc_auc_score(y_true, score, average="macro", multi_class="ovr", labels=np.arange(n_classes)),
        "AUC_weighted": roc_auc_score(y_true, score, average="weighted", multi_class="ovr", labels=np.arange(n_classes)),
        "AP_macro": average_precision_score(Y, score, average="macro"),
        "AP_weighted": average_precision_score(Y, score, average="weighted"),
        "Accuracy": accuracy_score(y_true, pred),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "Sensitivity_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "Sensitivity_weighted": recall_score(y_true, pred, average="weighted", zero_division=0)
    }, pred

# ============================================================
# MAIN LOOPS
# ============================================================

rows = []
predrows = []

for pca_m in PCA_COMPONENT_VALUES:
    print("\n" + "="*80)
    print("PCA_COMPONENTS PER MODULE =", pca_m)
    print("="*80)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold, (tr,te) in enumerate(skf.split(work,y), start=1):
            dtr = work.iloc[tr].reset_index(drop=True)
            dte = work.iloc[te].reset_index(drop=True)
            ytr = y[tr]
            yte = y[te]

            prep = ModulePreprocessor(pca_m).fit(dtr, ytr)
            Xtr = prep.transform(dtr)
            Xte = prep.transform(dte)

            print(
                f"{TASK} PCA={pca_m} seed={seed} fold={fold} "
                f"actual={prep.actual_components} total_dim={Xtr.shape[1]}"
            )

            real_graphs = {
                k: build_real_graph(Xtr, k)
                for k in K_GRAPH_VALUES
            }

            settings = []

            if "NO_GRAPH" in GRAPH_CONDITIONS:
                W0 = np.zeros((len(Xtr), len(Xtr)), dtype=np.float32)
                for lam in LAMBDA_CLS_VALUES:
                    settings.append(("NO_GRAPH", lam, 0.0, None, W0))

            for k in K_GRAPH_VALUES:
                Wr = real_graphs[k]

                if "RANDOM_GRAPH" in GRAPH_CONDITIONS:
                    Wrand = make_random_graph(
                        Wr,
                        stable_seed(TASK,pca_m,seed,fold,k,"random_graph")
                    )
                    for lam in LAMBDA_CLS_VALUES:
                        for alpha in ALPHA_GRAPH_VALUES:
                            settings.append(("RANDOM_GRAPH", lam, alpha, k, Wrand))

                if "REAL_GRAPH" in GRAPH_CONDITIONS:
                    for lam in LAMBDA_CLS_VALUES:
                        for alpha in ALPHA_GRAPH_VALUES:
                            settings.append(("REAL_GRAPH", lam, alpha, k, Wr))

            for condition,lam,alpha,k,W in settings:
                run_seed = stable_seed(TASK,pca_m,seed,fold,condition,lam,alpha,k)

                print(
                    f"  {condition}: lambda={lam}, alpha={alpha}, k={k}"
                )

                ztr,zte,ep = fit_ae(
                    Xtr,ytr,Xte,W,run_seed,lam,alpha
                )

                for method,score in downstream(ztr,ytr,zte,seed).items():
                    met,pred = evaluate(yte,score)

                    rows.append({
                        "Task": TASK,
                        "PCAComponents": pca_m,
                        "GraphCondition": condition,
                        "LambdaCls": lam,
                        "AlphaGraph": alpha,
                        "KGraph": k,
                        "Method": method,
                        "Seed": seed,
                        "Fold": fold,
                        "BestEpoch": ep,
                        **met
                    })

                    tmp = metadata.iloc[te].copy().reset_index(drop=True)
                    tmp.insert(0,"Task",TASK)
                    tmp.insert(1,"PCAComponents",pca_m)
                    tmp.insert(2,"GraphCondition",condition)
                    tmp.insert(3,"LambdaCls",lam)
                    tmp.insert(4,"AlphaGraph",alpha)
                    tmp.insert(5,"KGraph",k)
                    tmp.insert(6,"Method",method)
                    tmp.insert(7,"Seed",seed)
                    tmp.insert(8,"Fold",fold)

                    if TASK == "MCED":
                        tmp["y_true"] = yte
                        tmp["y_pred"] = pred
                        tmp["score_cancer"] = score
                    else:
                        tmp["y_true_encoded"] = yte
                        tmp["y_true_label"] = label_encoder.inverse_transform(yte)
                        tmp["y_pred_encoded"] = pred
                        tmp["y_pred_label"] = label_encoder.inverse_transform(pred)

                        for j,cls in enumerate(label_encoder.classes_):
                            safe = re.sub(r"[^A-Za-z0-9_]+","_",str(cls)).strip("_")
                            tmp[f"prob_{safe}"] = score[:,j]

                    tmp["BestEpoch"] = ep
                    predrows.append(tmp)

# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(rows)
predictions = pd.concat(predrows, ignore_index=True)

results.to_csv(
    OUTDIR / f"{TASK}_fixedPCA_threeGraphs_fold_results.csv",
    index=False
)

predictions.to_csv(
    OUTDIR / f"{TASK}_fixedPCA_threeGraphs_patient_predictions.csv",
    index=False
)

idcols = [
    "Task","PCAComponents","GraphCondition","LambdaCls",
    "AlphaGraph","KGraph","Method","Seed"
]

metriccols = [
    c for c in results.columns
    if c not in idcols + ["Fold","BestEpoch"]
]

seed_summary = (
    results.groupby(
        idcols,
        dropna=False,
        as_index=False
    )[metriccols].mean()
)

seed_summary.to_csv(
    OUTDIR / f"{TASK}_fixedPCA_threeGraphs_seed_summary.csv",
    index=False
)

grp = [
    "Task","PCAComponents","GraphCondition",
    "LambdaCls","AlphaGraph","KGraph","Method"
]

mean_df = seed_summary.groupby(
    grp, dropna=False
)[metriccols].mean().add_suffix("_mean")

std_df = seed_summary.groupby(
    grp, dropna=False
)[metriccols].std().add_suffix("_std")

summary = pd.concat([mean_df,std_df],axis=1).reset_index()

summary.to_csv(
    OUTDIR / f"{TASK}_fixedPCA_threeGraphs_method_summary.csv",
    index=False
)

rank_metric = "AUC_mean" if TASK=="MCED" else "Accuracy_mean"

summary.sort_values(
    rank_metric,
    ascending=False
).to_csv(
    OUTDIR / f"{TASK}_fixedPCA_threeGraphs_ranked_summary.csv",
    index=False
)

print("\nTOP SETTINGS:")
print(summary.sort_values(rank_metric,ascending=False).head(30))
print("\nResults folder:", OUTDIR)
