"""Small reproducible survival baseline; distinct from the original competition ensemble."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from aml.data import NUMERIC


def survival_curve(target):
    """Estimate Kaplan–Meier survival, accounting for right censoring."""
    if target.empty:
        raise ValueError("No follow-up data for this cohort.")
    times, survival = kaplan_meier_estimator(
        target["OS_STATUS"].to_numpy(dtype=bool),
        target["OS_YEARS"].to_numpy(dtype=float),
    )
    return pd.DataFrame({"years": np.r_[0.0, times], "survival": np.r_[1.0, survival]})


def evaluate_baseline(clinical, target, seed=42):
    """Fit on 75%, evaluate on held-out 25%; imputation/scaling learn only from training."""
    data = clinical.merge(target, on="ID", validate="one_to_one")
    if len(data) < 40 or data["OS_STATUS"].value_counts().reindex([0, 1], fill_value=0).min() < 8:
        raise ValueError("At least 40 labelled patients and 8 in each status are required.")
    train, test = train_test_split(data, test_size=0.25, random_state=seed, stratify=data["OS_STATUS"])
    if (train[NUMERIC].notna().sum() == 0).any():
        raise ValueError("Each clinical feature needs at least one training value.")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        CoxPHSurvivalAnalysis(alpha=1.0),
    )
    model.fit(train[NUMERIC], Surv.from_arrays(train["OS_STATUS"].astype(bool), train["OS_YEARS"]))
    risk = model.predict(test[NUMERIC])
    score = concordance_index_censored(test["OS_STATUS"].astype(bool), test["OS_YEARS"], risk)[0]
    predictions = test[["ID", "OS_YEARS", "OS_STATUS"]].copy()
    predictions["relative_risk_score"] = risk
    return {"c_index": float(score), "n_train": len(train), "n_test": len(test),
            "predictions": predictions.sort_values("relative_risk_score", ascending=False)}
