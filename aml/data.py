"""CSV contracts: reject malformed values, preserve missing clinical measurements."""
import numpy as np
import pandas as pd

NUMERIC = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
SCHEMAS = {
    "clinical": ["ID", "CENTER", *NUMERIC],
    "molecular": ["ID", "GENE"],
    "target": ["ID", "OS_YEARS", "OS_STATUS"],
}


def load_csv(source, kind, allow_incomplete_target=False):
    """Read a CSV path or file-like object and validate its explicit schema."""
    if kind not in SCHEMAS:
        raise ValueError("Unknown CSV type.")
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError("Invalid or empty CSV file.") from exc
    missing = set(SCHEMAS[kind]) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("The CSV must contain at least one row.")
    frame = frame.copy()
    frame["ID"] = frame["ID"].str.strip()
    if frame["ID"].eq("").any():
        raise ValueError("Patient IDs must not be empty.")
    if kind != "molecular" and frame["ID"].duplicated().any():
        raise ValueError("Patient IDs must be unique in clinical and target files.")
    columns = NUMERIC if kind == "clinical" else ["OS_YEARS", "OS_STATUS"] if kind == "target" else []
    for col in columns:
        raw = frame[col].str.strip()
        is_missing = raw.str.lower().isin(["", "na", "nan", "null"])
        values = pd.to_numeric(raw.mask(is_missing), errors="coerce")
        if ((~is_missing & values.isna()) | (values.notna() & ~np.isfinite(values))).any():
            raise ValueError(f"{col}: values must be finite numbers or missing.")
        if (values.dropna() < 0).any():
            raise ValueError(f"{col}: negative values are not allowed.")
        frame[col] = values
    if kind == "clinical":
        frame["CENTER"] = frame["CENTER"].str.strip().replace("", "Unknown")
        if (frame["BM_BLAST"].dropna() > 100).any():
            raise ValueError("BM_BLAST must be between 0 and 100.")
    if kind == "target":
        incomplete = frame[["OS_YEARS", "OS_STATUS"]].isna().any(axis=1)
        if incomplete.any():
            if not allow_incomplete_target:
                raise ValueError("Target values cannot be missing.")
            removed = int(incomplete.sum())
            frame = frame.loc[~incomplete].copy()
            frame.attrs["excluded_followup"] = removed
            if frame.empty:
                raise ValueError("No complete follow-up rows remain.")
        if not frame["OS_STATUS"].isin([0, 1]).all():
            raise ValueError("OS_STATUS must be 0 (censored) or 1 (event).")
        frame["OS_STATUS"] = frame["OS_STATUS"].astype(int)
    if kind == "molecular":
        frame["GENE"] = frame["GENE"].str.strip().replace("", pd.NA)
    return frame


def check_alignment(clinical, molecular, target):
    """Reject orphan rows instead of silently losing them in joins."""
    ids = set(clinical["ID"])
    for name, frame in (("molecular", molecular), ("target", target)):
        if frame is not None and not set(frame["ID"]).issubset(ids):
            raise ValueError(f"{name}: some IDs do not exist in the clinical file.")


def filter_clinical(clinical, centers=None, blast_range=None, include_missing=True):
    """Filter without mutating inputs; empty center selection means no patients."""
    result = clinical.copy()
    if centers is not None:
        result = result[result["CENTER"].isin(centers)]
    if blast_range is not None:
        low, high = blast_range
        if not 0 <= low <= high <= 100:
            raise ValueError("Invalid blast range.")
        mask = result["BM_BLAST"].between(low, high, inclusive="both")
        if include_missing:
            mask |= result["BM_BLAST"].isna()
        result = result[mask]
    return result.reset_index(drop=True)


def gene_prevalence(molecular, patient_ids):
    """Count distinct patients per gene; denominator includes patients without mutations."""
    ids = set(patient_ids)
    selected = molecular[molecular["ID"].isin(ids)].dropna(subset=["GENE"])
    counts = selected.drop_duplicates(["ID", "GENE"]).groupby("GENE")["ID"].nunique()
    out = counts.rename("patients").reset_index()
    out["percent"] = out["patients"] / len(ids) * 100 if ids else 0.0
    return out.sort_values(["patients", "GENE"], ascending=[False, True]).reset_index(drop=True)
