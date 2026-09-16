"""Streamlit entry point. Run from the repository root."""
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st
from aml.data import NUMERIC, check_alignment, filter_clinical, gene_prevalence, load_csv
from aml.model import evaluate_baseline, survival_curve

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AML · Cohort Explorer", page_icon="🧬", layout="wide")
st.caption("ENS DATA CHALLENGE · RESEARCH EXPLORER")
st.title("Understand the cohort. Explore survival.")
st.write("Explore clinical measurements and mutations, then evaluate a reproducible survival baseline.")
st.caption("Educational research project. Outputs are not clinical recommendations.")

source = st.sidebar.radio("Data source", ["Synthetic demo", "Repository data", "Upload CSV files"])
try:
    if source == "Upload CSV files":
        uploads = {kind: st.sidebar.file_uploader(label, type="csv", key=kind)
                   for kind, label in [("clinical", "Clinical CSV (required)"),
                                       ("molecular", "Molecular CSV (optional)"),
                                       ("target", "Follow-up CSV (optional)")]}
        if uploads["clinical"] is None:
            st.info("Upload a clinical CSV. Download example files from the demo folder in the repository.")
            st.stop()
        frames = {k: load_csv(v, k) if v is not None else None for k, v in uploads.items()}
    else:
        folder = ROOT / ("demo" if source == "Synthetic demo" else "data")
        frames = {k: load_csv(folder / name, k, allow_incomplete_target=True) for k, name in [
            ("clinical", "clinical_train.csv"), ("molecular", "molecular_train.csv"),
            ("target", "target_train.csv")]}
    clinical, molecular, target = (frames[k] for k in ["clinical", "molecular", "target"])
    check_alignment(clinical, molecular, target)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

if target is not None and target.attrs.get("excluded_followup", 0):
    st.warning(f"{target.attrs['excluded_followup']} incomplete follow-up rows excluded from survival analyses; clinical records are retained.")

if source == "Synthetic demo":
    st.info("Synthetic demonstration data: no real patient records. Results do not represent the ENS cohort.")

centers = sorted(clinical["CENTER"].unique())
selected = st.sidebar.multiselect("Centers", centers, default=centers)
bounds = st.sidebar.slider("Bone marrow blasts (%)", 0, 100, (0, 100))
missing = st.sidebar.checkbox("Include missing blast measurements", value=True)
cohort = filter_clinical(clinical, selected, bounds, missing)
st.sidebar.caption("All charts and the baseline use the selected cohort.")
a, b, c = st.columns(3)
a.metric("Selected patients", len(cohort))
b.metric("Centers", cohort["CENTER"].nunique())
c.metric("Missing clinical values", f"{cohort[NUMERIC].isna().mean().mean():.1%}" if len(cohort) else "—")
if cohort.empty:
    st.info("No patients match these filters. Broaden the selection.")
    st.stop()

overview, genetics, followup, model_tab = st.tabs(["Clinical overview", "Mutations", "Survival", "Model baseline"])
with overview:
    feature = st.selectbox("Clinical measurement", NUMERIC)
    st.altair_chart(alt.Chart(cohort).mark_bar(color="#16847b").encode(
        x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30)), y=alt.Y("count():Q", title="Patients")),
        width="stretch")
    st.caption("Missing values are excluded from the histogram.")
    st.dataframe(cohort, hide_index=True, width="stretch")
    st.download_button("Download filtered clinical data", cohort.to_csv(index=False),
                       "filtered_clinical.csv", "text/csv")
with genetics:
    if molecular is None:
        st.info("Upload a molecular CSV to explore mutations.")
    else:
        prevalence = gene_prevalence(molecular, cohort["ID"])
        if prevalence.empty:
            st.info("No recorded genes for the selected patients.")
        else:
            st.bar_chart(prevalence.head(15).set_index("GENE")["percent"], horizontal=True)
            st.caption("Percentage of selected patients with a recorded mutation in each gene; each patient counts once per gene.")
            st.dataframe(prevalence, hide_index=True)
with followup:
    labelled = None if target is None else target[target["ID"].isin(cohort["ID"])]
    if labelled is None or labelled.empty:
        st.info("No follow-up records available for this cohort.")
    else:
        st.caption(f"Follow-up available for {len(labelled)} / {len(cohort)} patients. OS_STATUS=0 denotes right censoring.")
        curve = survival_curve(labelled)
        st.altair_chart(alt.Chart(curve).mark_line(interpolate="step-after", color="#16847b").encode(
            x=alt.X("years:Q", title="Follow-up (years)"),
            y=alt.Y("survival:Q", title="Kaplan–Meier estimate", scale=alt.Scale(domain=[0, 1]))),
            width="stretch")
        st.caption("Descriptive cohort estimate, without confidence intervals. Not an individual prediction.")
with model_tab:
    st.write("Regularized Cox baseline using six clinical measurements, with a fixed 75/25 train/test split (seed 42).")
    st.caption("This is a new lightweight baseline, not the original Coxnet + gradient boosting competition ensemble. "
               "Its Harrell C-index is not directly comparable with the competition IPCW C-index.")
    if target is None:
        st.info("Upload follow-up data to evaluate the baseline.")
    elif st.button("Train and evaluate baseline"):
        try:
            with st.spinner("Training on the selected cohort…"):
                result = evaluate_baseline(cohort, target)
            st.metric("Held-out Harrell C-index", f"{result['c_index']:.3f}")
            st.write(f"Training: {result['n_train']} patients · Test: {result['n_test']} patients")
            st.caption("Higher scores indicate higher relative risk, not probabilities. Evaluation uses one holdout split.")
            st.dataframe(result["predictions"], hide_index=True)
            st.download_button("Download held-out scores", result["predictions"].to_csv(index=False),
                               "heldout_scores.csv", "text/csv", on_click="ignore")
        except (ValueError, ArithmeticError) as exc:
            st.error(f"Unable to evaluate this cohort: {exc}")
