#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Triple-ensemble survival ranking (IPCW-C-index @ tau=7), optimized weight tuning.

Models:
- Coxnet (Elastic-Net)
- GBSA1
- GBSA2 (different hyperparams)

Ensembles:
- Ensemble2: Coxnet + GBSA1 (rank-average)
- Ensemble3: Coxnet + GBSA1 + GBSA2 (rank-weighted)

Key improvement:
- --tune_weights now caches fold predictions ONCE (no refit per weight).

Usage:
1) CV only + tune weights (fast):
   python train_and_submit.py --data_dir data --out_dir outputs --cv_only --tune_weights \
     --gbsa_learning_rate 0.03 --gbsa_n_estimators 1200 --gbsa_max_depth 2 \
     --gbsa2_learning_rate 0.02 --gbsa2_n_estimators 2000 --gbsa2_max_depth 2

2) Create submission with triple ensemble:
   python train_and_submit.py --data_dir data --out_dir outputs --use_triple_ensemble \
     --weights "0.60,0.25,0.15" \
     --gbsa_learning_rate 0.03 --gbsa_n_estimators 1200 --gbsa_max_depth 2 \
     --gbsa2_learning_rate 0.02 --gbsa2_n_estimators 2000 --gbsa2_max_depth 2
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


# -----------------------------
# Utils
# -----------------------------

def safe_log1p(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x = x.clip(lower=0)
    return np.log1p(x)

def compute_top_categories(series: pd.Series, k: int) -> List[str]:
    s = series.dropna().astype(str)
    if s.empty:
        return []
    return s.value_counts().head(k).index.tolist()

def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x).reshape(-1))
    return (s.rank(method="average").values / (len(s) + 1.0)).astype(np.float64)

def _make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=50, sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -----------------------------
# Cytogenetics features
# -----------------------------

CYTO_PATTERNS = {
    "has_monosomy_7": r"(?<!\d)-7(?!\d)",
    "has_trisomy_8": r"(?<!\d)\+8(?!\d)",
    "has_del_5q": r"del\(\s*5\s*\)\s*\(\s*q|del\(\s*5q",
    "has_del_7q": r"del\(\s*7\s*\)\s*\(\s*q|del\(\s*7q",
    "has_inv_16": r"inv\(\s*16\s*\)|t\(\s*16\s*;\s*16\s*\)",
    "has_t_8_21": r"t\(\s*8\s*;\s*21\s*\)",
    "has_t_15_17": r"t\(\s*15\s*;\s*17\s*\)",
    "has_inv_3": r"inv\(\s*3\s*\)|t\(\s*3\s*;\s*3\s*\)",
}

def make_cytogenetics_features(cyto_series: pd.Series) -> pd.DataFrame:
    s = cyto_series.fillna("").astype(str)
    feats = pd.DataFrame(index=cyto_series.index)

    feats["cyto_len"] = s.str.len().astype(float)
    feats["cyto_n_semicolons"] = s.str.count(";").astype(float)
    feats["cyto_n_commas"] = s.str.count(",").astype(float)
    feats["cyto_n_plus"] = s.str.count(r"\+").astype(float)
    feats["cyto_n_minus"] = s.str.count(r"-").astype(float)
    feats["cyto_n_del"] = s.str.count("del").astype(float)
    feats["cyto_n_t"] = s.str.count(r"t\(").astype(float)
    feats["cyto_n_inv"] = s.str.count(r"inv\(").astype(float)
    feats["cyto_n_add"] = s.str.count("add").astype(float)
    feats["cyto_n_der"] = s.str.count("der").astype(float)
    feats["cyto_n_dup"] = s.str.count("dup").astype(float)
    feats["cyto_n_ins"] = s.str.count("ins").astype(float)

    normal_like = s.str.match(r"^\s*46\s*,\s*(XX|XY)\s*$", case=False)
    feats["cyto_is_normal_like"] = normal_like.astype(float)

    for name, pat in CYTO_PATTERNS.items():
        feats[name] = s.str.contains(pat, flags=re.IGNORECASE, regex=True).astype(float)

    abn_score = (
        feats["cyto_n_plus"]
        + feats["cyto_n_minus"]
        + feats["cyto_n_del"]
        + feats["cyto_n_t"]
        + feats["cyto_n_inv"]
        + feats["cyto_n_der"]
        + feats["cyto_n_add"]
        + feats["cyto_n_dup"]
        + feats["cyto_n_ins"]
    )
    feats["cyto_is_complex_3plus"] = (abn_score >= 3).astype(float)
    return feats


# -----------------------------
# Clinical features
# -----------------------------

def enrich_clinical(clin: pd.DataFrame) -> pd.DataFrame:
    df = clin.copy()

    blood_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    for c in blood_cols:
        if c in df.columns:
            df[f"{c}_isna"] = df[c].isna().astype(float)

    log_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "PLT"]
    for c in log_cols:
        if c in df.columns:
            df[f"log1p_{c}"] = safe_log1p(df[c])

    eps = 1e-6
    if {"ANC", "WBC"}.issubset(df.columns):
        df["anc_wbc"] = pd.to_numeric(df["ANC"], errors="coerce") / (pd.to_numeric(df["WBC"], errors="coerce") + eps)
    if {"MONOCYTES", "WBC"}.issubset(df.columns):
        df["mono_wbc"] = pd.to_numeric(df["MONOCYTES"], errors="coerce") / (pd.to_numeric(df["WBC"], errors="coerce") + eps)
    if {"HB", "PLT"}.issubset(df.columns):
        df["hb_plt"] = pd.to_numeric(df["HB"], errors="coerce") / (pd.to_numeric(df["PLT"], errors="coerce") + eps)

    return df


# -----------------------------
# Molecular effect groups
# -----------------------------

def effect_group(effect) -> str:
    if effect is None or (isinstance(effect, float) and np.isnan(effect)):
        e = ""
    else:
        e = str(effect).lower()
    if re.search(r"frameshift|stop_gained|nonsense|start_lost|stop_lost", e):
        return "trunc"
    if "splice" in e:
        return "splice"
    if re.search(r"missense|non_synonymous|nonsynonymous", e):
        return "missense"
    if re.search(r"synonymous|silent", e):
        return "synonymous"
    if re.search(r"inframe|in_frame", e):
        return "inframe_indel"
    if re.search(r"insertion|deletion|indel", e):
        return "indel_other"
    return "other"


# -----------------------------
# Driver genes + co-mutations
# -----------------------------

def compute_co_pairs(mol_train: pd.DataFrame, driver_genes: List[str], top_k: int = 20, min_count: int = 10) -> List[Tuple[str, str]]:
    if not driver_genes or "GENE" not in mol_train.columns:
        return []
    df = mol_train.copy()
    df["GENE"] = df["GENE"].astype(str)

    gene_ct = pd.crosstab(df["ID"], df["GENE"]).reindex(columns=driver_genes, fill_value=0)
    pres = (gene_ct > 0).astype(int).values
    co = pres.T @ pres

    pairs = []
    for i in range(len(driver_genes)):
        for j in range(i + 1, len(driver_genes)):
            c = int(co[i, j])
            if c >= min_count:
                pairs.append((driver_genes[i], driver_genes[j], c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:top_k]]


# -----------------------------
# Molecular aggregation
# -----------------------------

def aggregate_molecular(
    mol_df: pd.DataFrame,
    top_genes: List[str],
    top_effects: List[str],
    driver_genes: List[str],
    co_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    df = mol_df.copy()
    for col in ["GENE", "EFFECT", "VAF", "DEPTH", "PROTEIN_CHANGE"]:
        if col not in df.columns:
            df[col] = np.nan

    df["GENE"] = df["GENE"].astype(str)
    df["EFFECT"] = df["EFFECT"].astype(str)
    df["_VAF_NUM"] = pd.to_numeric(df["VAF"], errors="coerce")
    df["_DEPTH_NUM"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    g = df.groupby("ID", sort=False)
    out = pd.DataFrame(index=g.size().index)

    out["mut_n_mutations"] = g.size().astype(float)
    out["mut_n_genes"] = g["GENE"].nunique(dropna=True).astype(float)
    out["mut_n_effects"] = g["EFFECT"].nunique(dropna=True).astype(float)
    out["mut_n_protein_changes"] = g["PROTEIN_CHANGE"].nunique(dropna=True).astype(float)

    gv = df.groupby("ID")["_VAF_NUM"]
    out["mut_vaf_mean"] = gv.mean()
    out["mut_vaf_max"] = gv.max()
    out["mut_vaf_sum"] = gv.sum()
    out["mut_vaf_std"] = gv.std()

    gd = df.groupby("ID")["_DEPTH_NUM"]
    out["mut_depth_mean"] = gd.mean()
    out["mut_depth_max"] = gd.max()
    out["mut_depth_sum"] = gd.sum()
    out["mut_depth_std"] = gd.std()

    df["_VAF_NUM2"] = df["_VAF_NUM"].fillna(0.0)
    for thr in [0.02, 0.05, 0.10, 0.20, 0.30]:
        out[f"mut_n_vaf_ge_{int(thr*100):02d}"] = g.apply(lambda x, t=thr: float((x["_VAF_NUM2"] >= t).sum()))
    out["mut_frac_vaf_ge_10"] = out["mut_n_vaf_ge_10"] / (out["mut_n_mutations"] + 1e-6)
    out["mut_frac_vaf_ge_20"] = out["mut_n_vaf_ge_20"] / (out["mut_n_mutations"] + 1e-6)

    df["_EFF_GRP"] = df["EFFECT"].map(effect_group)
    effg = pd.crosstab(df["ID"], df["_EFF_GRP"]).astype(float)
    for c in ["trunc", "splice", "missense", "synonymous", "inframe_indel", "indel_other", "other"]:
        if c not in effg.columns:
            effg[c] = 0.0
    effg = effg[["trunc", "splice", "missense", "synonymous", "inframe_indel", "indel_other", "other"]]
    effg.columns = [f"effgrp_{c}_count" for c in effg.columns]
    out = out.join(effg, how="left").fillna(0.0)
    out["effgrp_trunc_frac"] = out["effgrp_trunc_count"] / (out["mut_n_mutations"] + 1e-6)
    out["effgrp_splice_frac"] = out["effgrp_splice_count"] / (out["mut_n_mutations"] + 1e-6)

    if top_genes:
        gene_ct = pd.crosstab(df["ID"], df["GENE"]).reindex(columns=top_genes, fill_value=0)
        gene_bin = (gene_ct > 0).astype(float)
        gene_bin.columns = [f"gene_{c}_present" for c in gene_bin.columns]
        out = out.join(gene_bin, how="left")

        vaf_by_gene = df.pivot_table(index="ID", columns="GENE", values="_VAF_NUM", aggfunc="max").reindex(columns=top_genes)
        vaf_by_gene.columns = [f"gene_{c}_vaf_max" for c in vaf_by_gene.columns]
        out = out.join(vaf_by_gene, how="left")

    if top_effects:
        eff_ct = pd.crosstab(df["ID"], df["EFFECT"]).reindex(columns=top_effects, fill_value=0).astype(float)
        eff_ct.columns = [f"eff_{c}_count" for c in eff_ct.columns]
        out = out.join(eff_ct, how="left")

    if driver_genes:
        gene_ct_drv = pd.crosstab(df["ID"], df["GENE"]).reindex(columns=driver_genes, fill_value=0)
        pres_drv = (gene_ct_drv > 0).astype(float)
        out["drv_present_count"] = pres_drv.sum(axis=1).astype(float)

        vaf_drv = df.pivot_table(index="ID", columns="GENE", values="_VAF_NUM", aggfunc="max").reindex(columns=driver_genes)
        out["drv_vaf_max"] = vaf_drv.max(axis=1).fillna(0.0).astype(float)
        out["drv_vaf_sum"] = vaf_drv.sum(axis=1).fillna(0.0).astype(float)

        if co_pairs:
            pres_int = (gene_ct_drv > 0).astype(int)
            for a, b in co_pairs:
                col = f"co_{a}__{b}"
                if a in pres_int.columns and b in pres_int.columns:
                    out[col] = ((pres_int[a] == 1) & (pres_int[b] == 1)).astype(float)
                else:
                    out[col] = 0.0

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# -----------------------------
# Build patient-level features
# -----------------------------

def build_patient_features(
    clinical_df: pd.DataFrame,
    molecular_df: pd.DataFrame,
    top_genes: List[str],
    top_effects: List[str],
    driver_genes: List[str],
    co_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    clin = clinical_df.copy()

    if "CYTOGENETICS" in clin.columns:
        cyto_feats = make_cytogenetics_features(clin["CYTOGENETICS"])
        clin = clin.drop(columns=["CYTOGENETICS"])
        clin = pd.concat([clin, cyto_feats], axis=1)

    clin = enrich_clinical(clin)
    clin = clin.drop_duplicates(subset=["ID"]).set_index("ID")

    mol_ag = aggregate_molecular(molecular_df, top_genes, top_effects, driver_genes, co_pairs)
    X = clin.join(mol_ag, how="left")

    for c in X.columns:
        if c.startswith(("mut_", "gene_", "eff_", "effgrp_", "drv_", "co_")):
            X[c] = X[c].fillna(0.0)

    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].fillna("UNK").astype(str)

    return X.replace([np.inf, -np.inf], np.nan)


# -----------------------------
# Pipelines
# -----------------------------

def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

def make_coxnet_pipeline(X: pd.DataFrame, l1_ratio: float, alpha: float) -> Pipeline:
    pre = make_preprocessor(X, scale_numeric=True)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        alphas=[float(alpha)],
        tol=1e-5,
        max_iter=400000,
    )
    return Pipeline([("preprocess", pre), ("model", model)])

def make_gbsa_pipeline(X: pd.DataFrame, params: Dict) -> Pipeline:
    pre = make_preprocessor(X, scale_numeric=False)
    model = GradientBoostingSurvivalAnalysis(**params)
    return Pipeline([("preprocess", pre), ("model", model)])


# -----------------------------
# CV scoring (single models)
# -----------------------------

def cv_ipcw_score_single(pipe_builder, X: pd.DataFrame, y: np.ndarray, tau: float, n_splits: int, seed: int) -> float:
    event = y["event"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, event):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        pipe = pipe_builder(X_tr)
        pipe.fit(X_tr, y_tr)
        pred = np.asarray(pipe.predict(X_va)).reshape(-1)
        scores.append(float(concordance_index_ipcw(y_tr, y_va, pred, tau=tau)[0]))
    return float(np.mean(scores))


# -----------------------------
# CV weight tuning (FAST): cache per-fold rank preds once
# -----------------------------

def precompute_fold_rank_preds(
    cox_builder,
    gb1_builder,
    gb2_builder,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of folds with:
      (y_tr, y_va, r_cox, r_gb1, r_gb2) for validation set.
    """
    event = y["event"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for tr, va in tqdm(list(skf.split(X, event)), desc="Precompute folds"):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        cox = cox_builder(X_tr); cox.fit(X_tr, y_tr)
        gb1 = gb1_builder(X_tr); gb1.fit(X_tr, y_tr)
        gb2 = gb2_builder(X_tr); gb2.fit(X_tr, y_tr)

        p1 = np.asarray(cox.predict(X_va)).reshape(-1)
        p2 = np.asarray(gb1.predict(X_va)).reshape(-1)
        p3 = np.asarray(gb2.predict(X_va)).reshape(-1)

        r1, r2, r3 = rank01(p1), rank01(p2), rank01(p3)
        folds.append((y_tr, y_va, r1, r2, r3))
    return folds

def score_weights_on_folds(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    weights: Tuple[float, float, float],
    tau: float
) -> float:
    w1, w2, w3 = weights
    s = w1 + w2 + w3
    if s <= 0:
        return -1.0
    w1, w2, w3 = w1/s, w2/s, w3/s

    scores = []
    for y_tr, y_va, r1, r2, r3 in folds:
        ens = w1 * r1 + w2 * r2 + w3 * r3
        scores.append(float(concordance_index_ipcw(y_tr, y_va, ens, tau=tau)[0]))
    return float(np.mean(scores))


def parse_weights(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError('weights must be "w_cox,w_gb1,w_gb2"')
    return (float(parts[0]), float(parts[1]), float(parts[2]))

def default_weight_grid() -> List[Tuple[float, float, float]]:
    return [
        (0.50, 0.25, 0.25),
        (0.55, 0.25, 0.20),
        (0.60, 0.25, 0.15),
        (0.60, 0.20, 0.20),
        (0.65, 0.20, 0.15),
        (0.70, 0.20, 0.10),
        (0.70, 0.15, 0.15),
        (0.75, 0.15, 0.10),
        (0.80, 0.10, 0.10),
    ]


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="outputs")

    ap.add_argument("--tau", type=float, default=7.0)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--top_genes", type=int, default=250)
    ap.add_argument("--top_effects", type=int, default=25)

    ap.add_argument("--driver_genes_n", type=int, default=30)
    ap.add_argument("--co_pairs_k", type=int, default=20)

    ap.add_argument("--cv_only", action="store_true")

    ap.add_argument("--l1_ratio", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.025)

    ap.add_argument("--use_triple_ensemble", action="store_true")
    ap.add_argument("--weights", type=str, default="0.60,0.25,0.15")
    ap.add_argument("--tune_weights", action="store_true")

    ap.add_argument("--gbsa_learning_rate", type=float, default=0.03)
    ap.add_argument("--gbsa_n_estimators", type=int, default=1200)
    ap.add_argument("--gbsa_max_depth", type=int, default=2)

    ap.add_argument("--gbsa2_learning_rate", type=float, default=0.02)
    ap.add_argument("--gbsa2_n_estimators", type=int, default=2000)
    ap.add_argument("--gbsa2_max_depth", type=int, default=2)

    ap.add_argument("--clinical_train", type=str, default="clinical_train.csv")
    ap.add_argument("--molecular_train", type=str, default="molecular_train.csv")
    ap.add_argument("--clinical_test", type=str, default="clinical_test.csv")
    ap.add_argument("--molecular_test", type=str, default="molecular_test.csv")
    ap.add_argument("--target_train", type=str, default="target_train.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    p_clin_tr = os.path.join(args.data_dir, args.clinical_train)
    p_mol_tr  = os.path.join(args.data_dir, args.molecular_train)
    p_clin_te = os.path.join(args.data_dir, args.clinical_test)
    p_mol_te  = os.path.join(args.data_dir, args.molecular_test)
    p_y       = os.path.join(args.data_dir, args.target_train)

    print("Loading CSVs...")
    clin_tr = pd.read_csv(p_clin_tr)
    mol_tr  = pd.read_csv(p_mol_tr)
    clin_te = pd.read_csv(p_clin_te)
    mol_te  = pd.read_csv(p_mol_te)
    y_df    = pd.read_csv(p_y)

    top_genes = compute_top_categories(mol_tr["GENE"], args.top_genes) if "GENE" in mol_tr.columns else []
    top_effects = compute_top_categories(mol_tr["EFFECT"], args.top_effects) if "EFFECT" in mol_tr.columns else []
    driver_genes = top_genes[:min(args.driver_genes_n, len(top_genes))]
    co_pairs = compute_co_pairs(mol_tr, driver_genes=driver_genes, top_k=args.co_pairs_k, min_count=10)

    print(f"Top genes: {len(top_genes)} | Top effects: {len(top_effects)}")
    print(f"Driver genes: {len(driver_genes)} | Co-pairs: {len(co_pairs)}")

    print("Building features...")
    X_train = build_patient_features(clin_tr, mol_tr, top_genes, top_effects, driver_genes, co_pairs)
    X_test  = build_patient_features(clin_te, mol_te, top_genes, top_effects, driver_genes, co_pairs)

    y_df = y_df.drop_duplicates(subset=["ID"]).set_index("ID")
    y_df["OS_STATUS"] = pd.to_numeric(y_df["OS_STATUS"], errors="coerce")
    y_df["OS_YEARS"] = pd.to_numeric(y_df["OS_YEARS"], errors="coerce")
    y_df = y_df.loc[y_df["OS_STATUS"].isin([0, 1]) & y_df["OS_YEARS"].notna()].copy()

    common = X_train.index.intersection(y_df.index)
    X_train = X_train.loc[common].copy()
    y_df = y_df.loc[common].copy()

    event = y_df["OS_STATUS"].astype(int).values.astype(bool)
    time = y_df["OS_YEARS"].values.astype(float)
    y = Surv.from_arrays(event=event, time=time)

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Event rate: {event.mean():.3f}")

    gbsa1_params = dict(
        learning_rate=float(args.gbsa_learning_rate),
        n_estimators=int(args.gbsa_n_estimators),
        max_depth=int(args.gbsa_max_depth),
        random_state=int(args.seed),
    )
    gbsa2_params = dict(
        learning_rate=float(args.gbsa2_learning_rate),
        n_estimators=int(args.gbsa2_n_estimators),
        max_depth=int(args.gbsa2_max_depth),
        random_state=int(args.seed + 1),
    )

    cox_builder = lambda Xtr: make_coxnet_pipeline(Xtr, l1_ratio=args.l1_ratio, alpha=args.alpha)
    gb1_builder = lambda Xtr: make_gbsa_pipeline(Xtr, params=gbsa1_params)
    gb2_builder = lambda Xtr: make_gbsa_pipeline(Xtr, params=gbsa2_params)

    if args.cv_only:
        print("\nCV local (IPCW-C-index):")
        s_cox = cv_ipcw_score_single(cox_builder, X_train, y, tau=args.tau, n_splits=args.n_splits, seed=args.seed)
        print(f"  Coxnet (l1={args.l1_ratio}, alpha={args.alpha}) -> {s_cox:.4f}")

        s_gb1 = cv_ipcw_score_single(gb1_builder, X_train, y, tau=args.tau, n_splits=args.n_splits, seed=args.seed)
        print(f"  GBSA1 (lr={gbsa1_params['learning_rate']}, n={gbsa1_params['n_estimators']}, d={gbsa1_params['max_depth']}) -> {s_gb1:.4f}")

        s_gb2 = cv_ipcw_score_single(gb2_builder, X_train, y, tau=args.tau, n_splits=args.n_splits, seed=args.seed)
        print(f"  GBSA2 (lr={gbsa2_params['learning_rate']}, n={gbsa2_params['n_estimators']}, d={gbsa2_params['max_depth']}) -> {s_gb2:.4f}")

        if args.tune_weights:
            print("\nTuning triple-ensemble weights (FAST: cache fold preds once)...")
            folds = precompute_fold_rank_preds(cox_builder, gb1_builder, gb2_builder, X_train, y, n_splits=args.n_splits, seed=args.seed)
            grid = default_weight_grid()

            results = []
            for w in grid:
                s = score_weights_on_folds(folds, w, tau=args.tau)
                results.append((s, w))
            results.sort(key=lambda t: t[0], reverse=True)

            print("Top 5 weights:")
            for s, w in results[:5]:
                print(f"  weights={w} -> {s:.4f}")
            print(f"\nBest weights={results[0][1]} -> {results[0][0]:.4f}")

        else:
            w = parse_weights(args.weights)
            folds = precompute_fold_rank_preds(cox_builder, gb1_builder, gb2_builder, X_train, y, n_splits=args.n_splits, seed=args.seed)
            s = score_weights_on_folds(folds, w, tau=args.tau)
            print(f"\nTriple ensemble weights={w} -> {s:.4f}")

        print("\n(aucune soumission générée en mode --cv_only)")
        return

    # ---- Train full and generate submission ----

    # Fit models on full train
    cox = cox_builder(X_train); cox.fit(X_train, y)
    gb1 = gb1_builder(X_train); gb1.fit(X_train, y)
    gb2 = gb2_builder(X_train); gb2.fit(X_train, y)

    p_cox = np.asarray(cox.predict(X_test)).reshape(-1)
    p_gb1 = np.asarray(gb1.predict(X_test)).reshape(-1)
    p_gb2 = np.asarray(gb2.predict(X_test)).reshape(-1)

    if args.use_triple_ensemble:
        w = parse_weights(args.weights)
        w1, w2, w3 = w
        s = w1 + w2 + w3
        w1, w2, w3 = w1 / s, w2 / s, w3 / s
        risk = w1 * rank01(p_cox) + w2 * rank01(p_gb1) + w3 * rank01(p_gb2)
        print(f"Using TRIPLE ENSEMBLE weights={(w1, w2, w3)}")
    else:
        # default: keep your best current approach (Cox+GBSA1 50/50)
        risk = 0.5 * rank01(p_cox) + 0.5 * rank01(p_gb1)
        print("Using DOUBLE ENSEMBLE (Cox+GBSA1) 50/50")

    sub = pd.DataFrame({"ID": X_test.index.values, "risk_score": risk}).set_index("ID")
    sub_path = os.path.join(args.out_dir, "submission.csv")
    sub.to_csv(sub_path)

    model_path = os.path.join(args.out_dir, "model.joblib")
    dump(
        {
            "coxnet": cox,
            "gbsa1": gb1,
            "gbsa2": gb2,
            "weights": args.weights,
            "top_genes": top_genes,
            "top_effects": top_effects,
            "driver_genes": driver_genes,
            "co_pairs": co_pairs,
            "gbsa1_params": gbsa1_params,
            "gbsa2_params": gbsa2_params,
        },
        model_path,
    )

    print(f"Saved submission: {sub_path}")
    print(f"Saved model: {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()