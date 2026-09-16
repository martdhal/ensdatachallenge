from io import StringIO
from pathlib import Path
import pandas as pd
import pytest
from aml.data import load_csv, filter_clinical, gene_prevalence, check_alignment
ROOT = Path(__file__).resolve().parents[1]

def clinical():
    return load_csv(ROOT / "demo/clinical_train.csv", "clinical")

def test_preserves_ids_and_missing_values():
    data = load_csv(StringIO("ID,CENTER,BM_BLAST,WBC,ANC,MONOCYTES,HB,PLT\n001,A,,1,2,3,4,5"), "clinical")
    assert data.ID.tolist() == ["001"]
    assert pd.isna(data.BM_BLAST.iloc[0])

@pytest.mark.parametrize("text", [
    "", "wrong\n1", "ID,OS_YEARS,OS_STATUS\n",
    "ID,OS_YEARS,OS_STATUS\na,1,1\na,2,0",
    "ID,OS_YEARS,OS_STATUS\n,1,1",
    "ID,OS_YEARS,OS_STATUS\na,-1,1",
    "ID,OS_YEARS,OS_STATUS\na,inf,1",
    "ID,OS_YEARS,OS_STATUS\na,oops,1",
    "ID,OS_YEARS,OS_STATUS\na,1,2",
    "ID,OS_YEARS,OS_STATUS\na,,1",
])
def test_rejects_invalid_csv(text):
    with pytest.raises(ValueError):
        load_csv(StringIO(text), "target")

def test_filters_inclusive_missing_and_nonmutating():
    data = clinical().iloc[:3].copy()
    data["BM_BLAST"] = [10, 20, float("nan")]
    before = data.copy(deep=True)
    assert len(filter_clinical(data, blast_range=(10, 20))) == 3
    assert len(filter_clinical(data, blast_range=(10, 20), include_missing=False)) == 2
    assert filter_clinical(data, centers=[]).empty
    assert filter_clinical(data, centers=["absent"]).empty
    pd.testing.assert_frame_equal(data, before)

def test_center_filter_and_invalid_range():
    data = clinical()
    selected = filter_clinical(data, centers=["Demo A"])
    assert set(selected.CENTER) == {"Demo A"}
    with pytest.raises(ValueError):
        filter_clinical(data, blast_range=(90, 10))

def test_unique_patient_gene_prevalence():
    mol = pd.DataFrame({"ID": ["a", "a", "b", "outside"], "GENE": ["G", "G", "G", "G"]})
    result = gene_prevalence(mol, ["a", "b", "c", "d"])
    assert result.patients.tolist() == [2]
    assert result.percent.tolist() == [50]
    assert gene_prevalence(mol, []).empty

def test_alignment_rejects_orphans_but_allows_missing_followup():
    data = clinical()
    check_alignment(data, None, pd.DataFrame({"ID": data.ID.iloc[:2]}))
    with pytest.raises(ValueError, match="IDs"):
        check_alignment(data, pd.DataFrame({"ID": ["outsider"]}), None)

def test_incomplete_followup_exclusion_is_explicit():
    text = "ID,OS_YEARS,OS_STATUS\na,1,1\nb,,0\nc,2,"
    result = load_csv(StringIO(text), "target", allow_incomplete_target=True)
    assert result.ID.tolist() == ["a"]
    assert result.attrs["excluded_followup"] == 2
    with pytest.raises(ValueError, match="No complete"):
        load_csv(StringIO("ID,OS_YEARS,OS_STATUS\na,,1"), "target", allow_incomplete_target=True)
