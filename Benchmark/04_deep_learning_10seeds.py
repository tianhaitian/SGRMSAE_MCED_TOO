
"""
Four deep-learning models for MCED and TOO.

Outputs
-------
The script creates:
    DL4_MCED_TOO_windows_results/
    DL4_MCED_TOO_windows_results.zip

The zip file contains fold-level results, seed summaries, method summaries,
predictions, TOO probabilities, and plot-friendly long-format CSV files.
"""

import gc
import os
import random
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sklearn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
)


# ============================================================
# 1. User settings
# ============================================================

# If None, the script uses the first CSV file in the same folder as this code.
# Alternatively, set the file name directly, e.g. DATA_FILE = "elife_data.csv"
DATA_FILE = "elife_ready_to_use_dataset.csv"  # Edit to your exact input CSV filename

MCED_LABEL_COL = "Label"
TOO_LABEL_COL = "Diagnosis"

SEEDS = list(range(2021, 2031))

print("\n" + "=" * 70)
print("EXECUTED SCRIPT:", Path(__file__).resolve())
print("SEEDS:", SEEDS)
print("NUMBER OF SEEDS:", len(SEEDS))
print("=" * 70 + "\n")

assert len(SEEDS) == 10, f"Expected 10 seeds, but found {len(SEEDS)}: {SEEDS}"

# SEEDS=[1]
N_SPLITS = 5
DL_METHODS = ["DL0", "DL1", "DL2", "DL3"]
# DL_METHODS = ["DL0"]

# Training settings. Increase EPOCHS if you want stronger DL results.
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Reduces the genomic bin length for memory safety on Windows.
POOL_SIZE = 8

HIDDEN_DIM = 128
LATENT_DIM = 64
DROPOUT = 0.30

HEALTHY_NAMES = {
    "control", "healthy", "normal", "non-cancer", "non_cancer", "non cancer",
    "no cancer", "nocancer", "negative", "benign", "0"
}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "DL4_MCED_TOO_windows_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ============================================================
# 2. Reproducibility and small utilities
# ============================================================

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_healthy_value(x):
    return str(x).strip().lower() in HEALTHY_NAMES


def clean_array(X):
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def row_normalize(X, eps=1e-12):
    X = clean_array(X)
    s = X.sum(axis=1, keepdims=True)
    return np.divide(X, s, out=np.zeros_like(X), where=np.abs(s) > eps)


def scale_train_test(Xtr, Xte):
    Xtr = clean_array(Xtr)
    Xte = clean_array(Xte)
    if Xtr.shape[1] == 0:
        return Xtr.astype(np.float32), Xte.astype(np.float32), None
    scaler = StandardScaler()
    Xtr2 = scaler.fit_transform(Xtr).astype(np.float32)
    Xte2 = scaler.transform(Xte).astype(np.float32)
    return Xtr2, Xte2, scaler


def safe_stratified_kfold(y, requested_splits, seed):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    n_splits = requested_splits
    if min_count < n_splits:
        raise ValueError(f"A class has only {min_count} samples; cannot run {n_splits} folds.")
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples per class for CV. Counts:\n{counts}")
    if n_splits < requested_splits:
        print(f"WARNING: using {n_splits}-fold CV because the smallest class has {min_count} samples.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), n_splits


def safe_inner_train_valid_indices(y_train, seed, valid_size=0.15):
    idx = np.arange(len(y_train))
    counts = pd.Series(y_train).value_counts()
    if len(counts) >= 2 and counts.min() >= 2:
        return train_test_split(
            idx,
            test_size=valid_size,
            stratify=y_train,
            random_state=seed,
        )
    return train_test_split(
        idx,
        test_size=valid_size,
        stratify=None,
        random_state=seed,
    )


# ============================================================
# 3. Load data
# ============================================================

if DATA_FILE is None:
    csv_files = sorted(SCRIPT_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file was found in the same folder as this script.")
    if len(csv_files) > 1:
        print("Multiple CSV files found:")
        for i, p in enumerate(csv_files):
            print(f"{i}: {p.name}")
        print("\nUsing the first CSV file by default. Edit DATA_FILE if this is not correct.")
    DATA_PATH = csv_files[0]
else:
    DATA_PATH = SCRIPT_DIR / DATA_FILE

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


# ============================================================
# 4. Labels
# ============================================================

def make_mced_label(df_use, mced_col=MCED_LABEL_COL, too_col=TOO_LABEL_COL):
    if mced_col is not None and mced_col in df_use.columns:
        raw = df_use[mced_col]
        if pd.api.types.is_numeric_dtype(raw):
            y = raw.fillna(-999).astype(int).to_numpy()
            vals = sorted(np.unique(y).tolist())
            if len(vals) == 2:
                if vals != [0, 1]:
                    raise ValueError("MCED Label must use 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {vals}.")
        else:
            vals_raw = raw.astype(str).str.strip()
            vals = sorted(vals_raw.unique().tolist())
            if len(vals) == 2:
                raise ValueError("MCED Label must be numeric 0=healthy, 1=cancer; recode the CSV explicitly.")
                return y.astype(int), mced_col
            print(f"WARNING: {mced_col} is not binary. Detected labels: {vals[:20]}.")

    if too_col not in df_use.columns:
        raise ValueError(f"Cannot create MCED labels because {too_col} was not found.")

    y = np.array([0 if is_healthy_value(v) else 1 for v in df_use[too_col]], dtype=int)
    if len(np.unique(y)) != 2:
        print(df_use[too_col].value_counts(dropna=False).head(50))
        raise ValueError("Derived MCED labels have only one class. Please check HEALTHY_NAMES.")
    return y, f"derived_from_{too_col}"


def make_too_dataset(df_use, y_mced, too_col=TOO_LABEL_COL):
    if too_col not in df_use.columns:
        raise ValueError(f"Cannot find TOO_LABEL_COL = {too_col}.")
    mask = np.asarray(y_mced) == 1
    df_too = df_use.loc[mask].copy()
    df_too = df_too.loc[~df_too[too_col].astype(str).str.strip().str.lower().isin(HEALTHY_NAMES)].copy()
    encoder = LabelEncoder()
    y_too = encoder.fit_transform(df_too[too_col].astype(str).str.strip())
    if len(encoder.classes_) < 2:
        raise ValueError("TOO needs at least two cancer classes.")
    return df_too.reset_index(drop=True), y_too.astype(int), encoder


y_mced, mced_source = make_mced_label(df)
df_too, y_too, too_encoder = make_too_dataset(df, y_mced)

print("\nMCED label source:", mced_source)
print(pd.Series(y_mced).value_counts().sort_index())
print("\nTOO classes:")
for i, c in enumerate(too_encoder.classes_):
    print(i, "=", c)


# ============================================================
# 5. Feature parsing and tensor construction
# ============================================================

def _norm(c):
    c = str(c).strip()
    c = c.replace("GWNMD_", "GWMD_")
    c = c.replace("fien_", "flen_")
    c = c.replace("Motif_end_", "MOTIF_END_")
    return c


RE_TMD = re.compile(r"^TMD_(\d+)$", re.IGNORECASE)
RE_GWMD = re.compile(r"^GWMD_chr(\d+)_(\d+)$", re.IGNORECASE)
RE_CNA = re.compile(r"^CNA_(\d+):(\d+)-(\d+)$", re.IGNORECASE)
RE_GWFP = re.compile(r"^GWFP_(long|short|total|ratio)_(\d+):(\d+)-(\d+)$", re.IGNORECASE)
RE_FLEN = re.compile(r"^flen_(\d+)$", re.IGNORECASE)
RE_MOTIF = re.compile(r"^MOTIF_END_([ACGT]{4})$", re.IGNORECASE)


def detect_feature_columns(df_ref):
    cols_tmd, cols_gwmd, cols_cna = [], [], []
    cols_gwfp = {"long": [], "short": [], "total": [], "ratio": []}
    cols_flen, cols_motif = [], []

    for col in df_ref.columns:
        c = _norm(col)
        if RE_TMD.match(c):
            cols_tmd.append(col)
        elif RE_GWMD.match(c):
            cols_gwmd.append(col)
        elif RE_CNA.match(c):
            cols_cna.append(col)
        elif RE_GWFP.match(c):
            cols_gwfp[RE_GWFP.match(c).group(1).lower()].append(col)
        elif RE_FLEN.match(c):
            cols_flen.append(col)
        elif RE_MOTIF.match(c):
            cols_motif.append(col)

    return cols_tmd, cols_gwmd, cols_cna, cols_gwfp, cols_flen, cols_motif


cols_tmd, cols_gwmd, cols_cna, cols_gwfp, cols_flen, cols_motif = detect_feature_columns(df)

print("\nDetected feature counts:")
print("TMD:", len(cols_tmd))
print("GWMD:", len(cols_gwmd))
print("CNA:", len(cols_cna))
print(
    "GWFP long/short/total/ratio:",
    len(cols_gwfp["long"]),
    len(cols_gwfp["short"]),
    len(cols_gwfp["total"]),
    len(cols_gwfp["ratio"]),
)
print("FLEN:", len(cols_flen))
print("MOTIF:", len(cols_motif))


def cna_to_bin_index(start):
    return int((int(start) - 1) // 1_000_000 + 1)


bin_keys = set()
for col in cols_gwmd:
    m = RE_GWMD.match(_norm(col))
    chr_k, b = int(m.group(1)), int(m.group(2))
    if 1 <= chr_k <= 22:
        bin_keys.add((chr_k, b))

for col in cols_cna:
    m = RE_CNA.match(_norm(col))
    chr_k, start = int(m.group(1)), int(m.group(2))
    if 1 <= chr_k <= 22:
        bin_keys.add((chr_k, cna_to_bin_index(start)))

bin_list = sorted(bin_keys, key=lambda x: (x[0], x[1]))
bin_to_p = {k: i for i, k in enumerate(bin_list)}
CHANNELS = ["GWMD", "CNA", "GWFP_long", "GWFP_short", "GWFP_total", "GWFP_ratio"]

if len(bin_list) == 0:
    raise ValueError("No genomic-bin features were detected. Please check GWMD/CNA column names.")

print("P genomic bins:", len(bin_list))


def build_tensor(df_use):
    N = df_use.shape[0]
    P = len(bin_list)
    C = len(CHANNELS)
    X = np.full((N, P, C), np.nan, dtype=np.float32)

    ch = CHANNELS.index("GWMD")
    for col in cols_gwmd:
        m = RE_GWMD.match(_norm(col))
        key = (int(m.group(1)), int(m.group(2)))
        if key in bin_to_p:
            X[:, bin_to_p[key], ch] = pd.to_numeric(df_use[col], errors="coerce").to_numpy(dtype=np.float32)

    ch = CHANNELS.index("CNA")
    for col in cols_cna:
        m = RE_CNA.match(_norm(col))
        key = (int(m.group(1)), cna_to_bin_index(m.group(2)))
        if key in bin_to_p:
            X[:, bin_to_p[key], ch] = pd.to_numeric(df_use[col], errors="coerce").to_numpy(dtype=np.float32)

    def mb_range(start, end):
        mb_s = int((int(start) - 1) // 1_000_000 + 1)
        mb_e = int((int(end) - 1) // 1_000_000 + 1)
        return mb_s, mb_e

    def fill_gwfp(kind, channel_name):
        ch_idx = CHANNELS.index(channel_name)
        for col in cols_gwfp[kind]:
            m = RE_GWFP.match(_norm(col))
            chr_k = int(m.group(2))
            mb_s, mb_e = mb_range(m.group(3), m.group(4))
            vals = pd.to_numeric(df_use[col], errors="coerce").to_numpy(dtype=np.float32)
            for b in range(mb_s, mb_e + 1):
                key = (chr_k, b)
                if key in bin_to_p:
                    X[:, bin_to_p[key], ch_idx] = vals

    fill_gwfp("long", "GWFP_long")
    fill_gwfp("short", "GWFP_short")
    fill_gwfp("total", "GWFP_total")
    fill_gwfp("ratio", "GWFP_ratio")

    cols_tmd_sorted = sorted(cols_tmd, key=lambda c: int(RE_TMD.match(_norm(c)).group(1)))
    cols_flen_sorted = sorted(cols_flen, key=lambda c: int(RE_FLEN.match(_norm(c)).group(1)))
    cols_motif_sorted = sorted(cols_motif, key=lambda c: RE_MOTIF.match(_norm(c)).group(1))

    Xtmd = df_use[cols_tmd_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) if cols_tmd_sorted else np.zeros((N, 0), dtype=np.float32)
    Xflen = df_use[cols_flen_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) if cols_flen_sorted else np.zeros((N, 0), dtype=np.float32)
    Xmotif = df_use[cols_motif_sorted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) if cols_motif_sorted else np.zeros((N, 0), dtype=np.float32)

    if Xflen.shape[1] > 0:
        Xflen = row_normalize(Xflen)
    if Xmotif.shape[1] > 0:
        Xmotif = row_normalize(Xmotif)

    return X, Xtmd, Xflen, Xmotif


Xgen_mced, Xtmd_mced, Xflen_mced, Xmotif_mced = build_tensor(df)
Xgen_too, Xtmd_too, Xflen_too, Xmotif_too = build_tensor(df_too)


# ============================================================
# 6. Fold preprocessing
# ============================================================

def impute_tensor_train(X):
    X2 = X.copy().astype(np.float32)
    med = np.zeros(X2.shape[2], dtype=np.float32)
    for c in range(X2.shape[2]):
        vals = X2[:, :, c]
        finite = np.isfinite(vals)
        m = np.nanmedian(vals[finite]) if finite.any() else 0.0
        med[c] = m
        vals[~finite] = m
        X2[:, :, c] = vals
    return X2, med


def impute_tensor_apply(X, med):
    X2 = X.copy().astype(np.float32)
    for c in range(X2.shape[2]):
        vals = X2[:, :, c]
        vals[~np.isfinite(vals)] = med[c]
        X2[:, :, c] = vals
    return X2


def pool_genomic_bins(Xgen, pool_size=8):
    Xgen = clean_array(Xgen)
    if pool_size <= 1:
        return Xgen
    N, P, C = Xgen.shape
    P_new = P // pool_size
    if P_new < 1:
        return Xgen
    X_trim = Xgen[:, :P_new * pool_size, :]
    return X_trim.reshape(N, P_new, pool_size, C).mean(axis=2).astype(np.float32)


def prepare_fold_data(Xgen, Xtmd, Xflen, Xmotif, y, train_idx, test_idx, seed):
    Xgen_train_raw, Xgen_test_raw = Xgen[train_idx], Xgen[test_idx]
    Xtmd_train_raw, Xtmd_test_raw = Xtmd[train_idx], Xtmd[test_idx]
    Xflen_train_raw, Xflen_test_raw = Xflen[train_idx], Xflen[test_idx]
    Xmotif_train_raw, Xmotif_test_raw = Xmotif[train_idx], Xmotif[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    Xgen_train_imp, med = impute_tensor_train(Xgen_train_raw)
    Xgen_test_imp = impute_tensor_apply(Xgen_test_raw, med)

    Xgen_train = pool_genomic_bins(Xgen_train_imp, POOL_SIZE)
    Xgen_test = pool_genomic_bins(Xgen_test_imp, POOL_SIZE)

    Xtmd_train, Xtmd_test, _ = scale_train_test(Xtmd_train_raw, Xtmd_test_raw)
    Xflen_train, Xflen_test, _ = scale_train_test(Xflen_train_raw, Xflen_test_raw)
    Xmotif_train, Xmotif_test, _ = scale_train_test(Xmotif_train_raw, Xmotif_test_raw)

    tr_inner, va_inner = safe_inner_train_valid_indices(y_train_full, seed, valid_size=0.15)

    return {
        "train": {
            "Xgen": Xgen_train[tr_inner],
            "Xtmd": Xtmd_train[tr_inner],
            "Xflen": Xflen_train[tr_inner],
            "Xmotif": Xmotif_train[tr_inner],
            "y": y_train_full[tr_inner],
        },
        "valid": {
            "Xgen": Xgen_train[va_inner],
            "Xtmd": Xtmd_train[va_inner],
            "Xflen": Xflen_train[va_inner],
            "Xmotif": Xmotif_train[va_inner],
            "y": y_train_full[va_inner],
        },
        "test": {
            "Xgen": Xgen_test,
            "Xtmd": Xtmd_test,
            "Xflen": Xflen_test,
            "Xmotif": Xmotif_test,
            "y": y_test,
        },
    }


# ============================================================
# 7. Dataset and model definitions
# ============================================================

class MultiInputDataset(Dataset):
    def __init__(self, Xgen, Xtmd, Xflen, Xmotif, y, task):
        self.Xgen = torch.tensor(Xgen, dtype=torch.float32)
        self.Xtmd = torch.tensor(Xtmd, dtype=torch.float32)
        self.Xflen = torch.tensor(Xflen, dtype=torch.float32)
        self.Xmotif = torch.tensor(Xmotif, dtype=torch.float32)
        if task == "binary":
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xgen[idx], self.Xtmd[idx], self.Xflen[idx], self.Xmotif[idx], self.y[idx]


class SmallEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DL0_EarlyFusionMLP(nn.Module):
    def __init__(self, p_gen_flat, dim_tmd, dim_flen, dim_motif, out_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        input_dim = p_gen_flat + dim_tmd + dim_flen + dim_motif
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, xg, xt, xf, xm):
        x = torch.cat([xg.reshape(xg.shape[0], -1), xt, xf, xm], dim=1)
        return self.net(x)


class DL1_MultiBranchMLP(nn.Module):
    def __init__(self, p_gen_flat, dim_tmd, dim_flen, dim_motif, out_dim, latent_dim=64, dropout=0.3):
        super().__init__()
        self.enc_gen = SmallEncoder(p_gen_flat, latent_dim, dropout)
        self.enc_tmd = SmallEncoder(dim_tmd, latent_dim, dropout)
        self.enc_flen = SmallEncoder(dim_flen, latent_dim, dropout)
        self.enc_motif = SmallEncoder(dim_motif, latent_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(4 * latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, xg, xt, xf, xm):
        z = torch.cat([
            self.enc_gen(xg.reshape(xg.shape[0], -1)),
            self.enc_tmd(xt),
            self.enc_flen(xf),
            self.enc_motif(xm),
        ], dim=1)
        return self.classifier(z)


class DL2_CNNFusion(nn.Module):
    def __init__(self, p_bins, n_channels, dim_tmd, dim_flen, dim_motif, out_dim, latent_dim=64, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gen_proj = nn.Sequential(nn.Linear(64, latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.aux_encoder = nn.Sequential(
            nn.Linear(dim_tmd + dim_flen + dim_motif, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, xg, xt, xf, xm):
        zg = self.cnn(xg.permute(0, 2, 1)).squeeze(-1)
        zg = self.gen_proj(zg)
        zaux = self.aux_encoder(torch.cat([xt, xf, xm], dim=1))
        return self.classifier(torch.cat([zg, zaux], dim=1))


class DL3_TransformerFusion(nn.Module):
    def __init__(self, p_bins, n_channels, dim_tmd, dim_flen, dim_motif, out_dim, embed_dim=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.token_embed = nn.Linear(n_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.gen_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.aux_encoder = nn.Sequential(
            nn.Linear(dim_tmd + dim_flen + dim_motif, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, xg, xt, xf, xm):
        h = self.transformer(self.token_embed(xg))
        zg = self.gen_proj(h.mean(dim=1))
        zaux = self.aux_encoder(torch.cat([xt, xf, xm], dim=1))
        return self.classifier(torch.cat([zg, zaux], dim=1))


def build_dl_model(method_name, data, out_dim):
    Xg = data["train"]["Xgen"]
    Xt = data["train"]["Xtmd"]
    Xf = data["train"]["Xflen"]
    Xm = data["train"]["Xmotif"]

    p_bins, n_channels = Xg.shape[1], Xg.shape[2]
    p_gen_flat = p_bins * n_channels
    dim_tmd, dim_flen, dim_motif = Xt.shape[1], Xf.shape[1], Xm.shape[1]

    if method_name == "DL0":
        return DL0_EarlyFusionMLP(p_gen_flat, dim_tmd, dim_flen, dim_motif, out_dim, HIDDEN_DIM, DROPOUT)
    if method_name == "DL1":
        return DL1_MultiBranchMLP(p_gen_flat, dim_tmd, dim_flen, dim_motif, out_dim, LATENT_DIM, DROPOUT)
    if method_name == "DL2":
        return DL2_CNNFusion(p_bins, n_channels, dim_tmd, dim_flen, dim_motif, out_dim, LATENT_DIM, DROPOUT)
    if method_name == "DL3":
        return DL3_TransformerFusion(p_bins, n_channels, dim_tmd, dim_flen, dim_motif, out_dim, 64, 4, 2, DROPOUT)
    raise ValueError(f"Unknown method: {method_name}")


# ============================================================
# 8. Training and prediction
# ============================================================

def make_loaders(data, task):
    train_ds = MultiInputDataset(**data["train"], task=task)
    valid_ds = MultiInputDataset(**data["valid"], task=task)
    test_ds = MultiInputDataset(**data["test"], task=task)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, valid_loader, test_loader


def train_one_epoch(model, loader, optimizer, task, pos_weight=None):
    model.train()
    total_loss = 0.0
    for xg, xt, xf, xm, y in loader:
        xg, xt, xf, xm, y = xg.to(DEVICE), xt.to(DEVICE), xf.to(DEVICE), xm.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xg, xt, xf, xm)
        if task == "binary":
            logits = logits.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        else:
            loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict_model(model, loader, task):
    model.eval()
    probs, labels = [], []
    for xg, xt, xf, xm, y in loader:
        xg, xt, xf, xm = xg.to(DEVICE), xt.to(DEVICE), xf.to(DEVICE), xm.to(DEVICE)
        logits = model(xg, xt, xf, xm)
        if task == "binary":
            p = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
        else:
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(p)
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)


def validation_score(y_true, prob, task, labels):
    try:
        if task == "binary":
            return roc_auc_score(y_true, prob)
        return roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=labels)
    except Exception:
        return 0.0


def fit_dl_model(model, train_loader, valid_loader, y_train, task, labels):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    pos_weight = None
    if task == "binary":
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos > 0:
            pos_weight = torch.tensor(n_neg / max(1, n_pos), dtype=torch.float32).to(DEVICE)

    best_score = -np.inf
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, task, pos_weight)
        y_val, p_val = predict_model(model, valid_loader, task)
        score = validation_score(y_val, p_val, task, labels)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | loss={loss:.4f} | valid_score={score:.4f}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best valid score={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 9. Metrics
# ============================================================

def binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "AP": average_precision_score(y_true, y_score),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn + 1e-12),
        "Specificity": tn / (tn + fp + 1e-12),
        "Threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def multiclass_specificity(y_true, y_pred, labels):
    specs, supports = [], []
    for lab in labels:
        y_bin_true = (y_true == lab).astype(int)
        y_bin_pred = (y_pred == lab).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()
        specs.append(tn / (tn + fp + 1e-12))
        supports.append(np.sum(y_true == lab))
    specs = np.asarray(specs)
    supports = np.asarray(supports)
    return float(np.mean(specs)), float(np.average(specs, weights=supports))


def multiclass_metrics(y_true, y_prob, y_pred, labels):
    y_bin = label_binarize(y_true, classes=labels)
    spec_macro, spec_weighted = multiclass_specificity(y_true, y_pred, labels)
    return {
        "AUC_macro": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels),
        "AUC_weighted": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels),
        "AP_macro": average_precision_score(y_bin, y_prob, average="macro"),
        "AP_weighted": average_precision_score(y_bin, y_prob, average="weighted"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Sensitivity_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Sensitivity_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Specificity_macro": spec_macro,
        "Specificity_weighted": spec_weighted,
    }


# ============================================================
# 10. Run experiments
# ============================================================

def run_task(task_name, Xgen, Xtmd, Xflen, Xmotif, y, class_names=None):
    task = "binary" if task_name == "MCED" else "multiclass"
    labels = np.arange(len(np.unique(y)))
    out_dim = 1 if task == "binary" else len(labels)

    result_rows = []
    pred_rows = []
    prob_rows = []

    for seed in SEEDS:
        set_all_seeds(seed)
        skf, actual_splits = safe_stratified_kfold(y, N_SPLITS, seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            print(f"\n================ {task_name} seed={seed} fold={fold}/{actual_splits} ================")
            data = prepare_fold_data(Xgen, Xtmd, Xflen, Xmotif, y, train_idx, test_idx, seed + fold)

            for method in DL_METHODS:
                set_all_seeds(seed * 1000 + fold * 10 + DL_METHODS.index(method))
                t0 = time.time()
                print(f"\n{task_name} | {method} | seed={seed} | fold={fold}")
                set_all_seeds(seed + 100 * fold)

                train_loader, valid_loader, test_loader = make_loaders(data, task)
                model = build_dl_model(method, data, out_dim)
                model = fit_dl_model(model, train_loader, valid_loader, data["train"]["y"], task, labels)
                y_test, prob = predict_model(model, test_loader, task)

                if task == "binary":
                    y_score = prob
                    y_pred = (y_score >= 0.5).astype(int)
                    metrics = binary_metrics(y_test, y_score, threshold=0.5)
                    for local_i, global_i in enumerate(test_idx):
                        pred_rows.append({
                            "Task": task_name,
                            "Method": method,
                            "Seed": seed,
                            "Fold": fold,
                            "Sample_index": int(global_i),
                            "y_true": int(y_test[local_i]),
                            "y_pred": int(y_pred[local_i]),
                            "score_cancer": float(y_score[local_i]),
                        })
                else:
                    y_pred = np.argmax(prob, axis=1)
                    metrics = multiclass_metrics(y_test, prob, y_pred, labels)
                    for local_i, global_i in enumerate(test_idx):
                        true_name = class_names[y_test[local_i]] if class_names is not None else str(y_test[local_i])
                        pred_name = class_names[y_pred[local_i]] if class_names is not None else str(y_pred[local_i])
                        pred_rows.append({
                            "Task": task_name,
                            "Method": method,
                            "Seed": seed,
                            "Fold": fold,
                            "Sample_index": int(global_i),
                            "y_true": int(y_test[local_i]),
                            "y_true_name": str(true_name),
                            "y_pred": int(y_pred[local_i]),
                            "y_pred_name": str(pred_name),
                        })
                        for ci, cname in enumerate(class_names if class_names is not None else labels):
                            prob_rows.append({
                                "Task": task_name,
                                "Method": method,
                                "Seed": seed,
                                "Fold": fold,
                                "Sample_index": int(global_i),
                                "Class_id": int(ci),
                                "Class_name": str(cname),
                                "Probability": float(prob[local_i, ci]),
                            })

                metrics.update({
                    "Task": task_name,
                    "Method": method,
                    "Seed": seed,
                    "Fold": fold,
                    "N_splits": actual_splits,
                    "Train_pred_sec": time.time() - t0,
                })
                result_rows.append(metrics)
#                print(metrics)

                del model, train_loader, valid_loader, test_loader
                clear_mem()

            del data
            clear_mem()

    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows), pd.DataFrame(prob_rows)


mced_results, mced_predictions, mced_probabilities = run_task(
    "MCED", Xgen_mced, Xtmd_mced, Xflen_mced, Xmotif_mced, y_mced
)

too_results, too_predictions, too_probabilities = run_task(
    "TOO", Xgen_too, Xtmd_too, Xflen_too, Xmotif_too, y_too, class_names=too_encoder.classes_
)


# ============================================================
# 11. Summaries and saving
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
all_probabilities = pd.concat([mced_probabilities, too_probabilities], ignore_index=True, sort=False)

plot_rows = []
for _, row in all_seed_summary.iterrows():
    metric_cols = mced_metric_cols if row["Task"] == "MCED" else too_metric_cols
    for metric in metric_cols:
        plot_rows.append({
            "Task": row["Task"],
            "Method": row["Method"],
            "Seed": row["Seed"],
            "Metric": metric,
            "Value": row[metric],
        })
plot_long = pd.DataFrame(plot_rows)

mced_results.to_csv(OUTDIR / "DL4_MCED_fold_results.csv", index=False)
too_results.to_csv(OUTDIR / "DL4_TOO_fold_results.csv", index=False)
all_fold_results.to_csv(OUTDIR / "DL4_all_fold_results.csv", index=False)

mced_seed_summary.to_csv(OUTDIR / "DL4_MCED_seed_summary.csv", index=False)
too_seed_summary.to_csv(OUTDIR / "DL4_TOO_seed_summary.csv", index=False)
all_seed_summary.to_csv(OUTDIR / "DL4_all_seed_summary.csv", index=False)

mced_method_summary.to_csv(OUTDIR / "DL4_MCED_method_summary.csv", index=False)
too_method_summary.to_csv(OUTDIR / "DL4_TOO_method_summary.csv", index=False)
all_method_summary.to_csv(OUTDIR / "DL4_all_method_summary.csv", index=False)

mced_predictions.to_csv(OUTDIR / "DL4_MCED_predictions.csv", index=False)
too_predictions.to_csv(OUTDIR / "DL4_TOO_predictions.csv", index=False)
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

all_predictions.to_csv(OUTDIR / "DL4_all_predictions.csv", index=False)
all_probabilities.to_csv(OUTDIR / "DL4_TOO_probabilities_long.csv", index=False)
plot_long.to_csv(OUTDIR / "DL4_plot_long_format.csv", index=False)

config = pd.DataFrame([{
    "DATA_PATH": str(DATA_PATH),
    "SEEDS": str(SEEDS),
    "N_SPLITS": N_SPLITS,
    "DL_METHODS": str(DL_METHODS),
    "EPOCHS": EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "POOL_SIZE": POOL_SIZE,
    "MCED_LABEL_SOURCE": mced_source,
    "TOO_LABEL_COL": TOO_LABEL_COL,
}])
config.to_csv(OUTDIR / "DL4_run_config.csv", index=False)

zip_path = SCRIPT_DIR / "DL4_MCED_TOO_windows_results.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUTDIR.glob("*.csv")):
        zf.write(p, arcname=p.name)

print("\nDone.")
print("Results folder:", OUTDIR)
print("Zip file:", zip_path)
print("\nMCED method summary:")
print(mced_method_summary)
print("\nTOO method summary:")
print(too_method_summary)
