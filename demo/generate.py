"""Generate independent fictional data; no real data is read. Seed: 42."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 160
    ids = [f"DEMO-{i:03d}" for i in range(n)]
    blast = rng.uniform(5, 95, n)
    clinical = pd.DataFrame({
        "ID": ids, "CENTER": rng.choice(["Demo A", "Demo B", "Demo C"], n),
        "BM_BLAST": blast.round(2), "WBC": rng.lognormal(2, 0.6, n).round(2),
        "ANC": rng.uniform(0.1, 10, n).round(2), "MONOCYTES": rng.uniform(0, 4, n).round(2),
        "HB": rng.uniform(6, 15, n).round(2), "PLT": rng.uniform(10, 300, n).round(2),
    })
    clinical.loc[::13, "BM_BLAST"] = np.nan
    death = rng.exponential(4 * np.exp(-blast / 100), n)
    censor = rng.uniform(0.5, 8, n)
    target = pd.DataFrame({"ID": ids, "OS_YEARS": np.minimum(death, censor).round(4),
                           "OS_STATUS": (death <= censor).astype(int)})
    rows = [(patient, gene) for patient in ids
            for gene in rng.choice(["GENE_A", "GENE_B", "GENE_C", "GENE_D"], size=rng.integers(0, 4), replace=False)]
    molecular = pd.DataFrame(rows, columns=["ID", "GENE"])
    for name, frame in [("clinical", clinical), ("molecular", molecular), ("target", target)]:
        frame.to_csv(folder / f"{name}_train.csv", index=False)

if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
