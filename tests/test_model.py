from pathlib import Path
import pandas as pd
import pytest
from aml.data import load_csv
from aml.model import evaluate_baseline, survival_curve
ROOT = Path(__file__).resolve().parents[1]

def test_kaplan_meier_accounts_for_censoring_and_ties():
    target = pd.DataFrame({"OS_YEARS": [1., 2., 2., 3.], "OS_STATUS": [1, 1, 0, 1]})
    curve = survival_curve(target)
    assert curve.survival.tolist() == pytest.approx([1., .75, .5, 0.])

def test_all_censored_curve():
    curve = survival_curve(pd.DataFrame({"OS_YEARS": [1., 2.], "OS_STATUS": [0, 0]}))
    assert (curve.survival == 1).all()

def test_baseline_is_reproducible():
    clinical = load_csv(ROOT / "demo/clinical_train.csv", "clinical")
    target = load_csv(ROOT / "demo/target_train.csv", "target")
    first = evaluate_baseline(clinical, target)
    second = evaluate_baseline(clinical, target)
    assert 0 <= first["c_index"] <= 1
    assert first["n_train"] == 120 and first["n_test"] == 40
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    with pytest.raises(ValueError, match="40"):
        evaluate_baseline(clinical.head(10), target)
