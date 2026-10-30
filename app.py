"""Streamlit demo — postprandial glucose-response (PPGR) prediction.

Research demo for the MetaNutri project's research module. Trained on CGMacros
(45 subjects, 1,557 meals) with leave-one-subject-out cross-validation.
Not a clinical tool.

The layout follows Streamlit's own theming and layout primitives: a
``.streamlit/config.toml`` theme for colour and radius, native containers,
tabs and metrics for structure, and a small stylesheet for typography only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
from inference import (
    MEAL_FIELDS,
    TARGET_LABELS,
    PPGRModel,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
EXTERNAL_CSV = os.path.join(ASSETS, "external_results.csv")
EXTERNAL_FIG = os.path.join(ASSETS, "fig5_external_validation.png")

st.set_page_config(
    page_title="PPGR Predictor — MetaNutri research demo",
    page_icon="🩸",
    layout="wide",
)

# Rows shown in the external-validation panel: (display label, cohort, model).
EXTERNAL_ROWS = [
    ("CGMacros (LOPO, internal)", "CGMacros (LOPO, internal)", "xgboost"),
    ("BIG IDEAs (LOPO, internal)", "BIG IDEAs (LOPO, internal)", "xgboost"),
    ("BIG IDEAs (external, XGBoost)", "BIG IDEAs (external transfer)", "xgboost"),
    ("BIG IDEAs (external, carb-only Ridge)", "BIG IDEAs (external transfer)", "carb-only linear"),
]

MEAL_LABELS = {
    "carbs_g": "Carbohydrates (g)",
    "protein_g": "Protein (g)",
    "fat_g": "Fat (g)",
    "fiber_g": "Fiber (g)",
    "baseline_glucose": "Meal-time glucose (mg/dL)",
}
SHAP_LABELS = {
    "Meal-time glucose (mg/dL)": "Meal-time glucose",
    "Fasting insulin (µIU/mL)": "Fasting insulin",
    "Sex (M=1/F=−1)": "Sex",
}
SUBJECT_LABELS = {
    "age": "Age (years)",
    "bmi": "BMI (kg/m²)",
    "a1c": "HbA1c (%)",
    "fasting_bg": "Fasting glucose (mg/dL)",
    "insulin": "Fasting insulin (µIU/mL)",
    "tg": "Triglycerides (mg/dL)",
    "cholesterol": "Total cholesterol (mg/dL)",
    "hdl": "HDL (mg/dL)",
    "non_hdl": "Non-HDL (mg/dL)",
    "ldl": "LDL (mg/dL)",
    "vldl": "VLDL (mg/dL)",
    "cho_hdl_ratio": "Cholesterol / HDL ratio",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #17191B;
  --ink-soft: #2B2B29;
  --paper: #F8F5EF;
  --card: #FFFDF9;
  --line: #DCD5C8;
  --oxblood: #8E2231;
  --sage: #5A6B5C;
  --muted: #6E6A62;
}

.stApp { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] input,
[data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] button {
  font-family: "IBM Plex Sans", "Source Sans", sans-serif;
}

.block-container {
  max-width: 1160px;
  padding-top: 2.4rem;
  padding-bottom: 3.5rem;
}

h1, h2, h3 {
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.015em;
}
h1 {
  font-size: clamp(2rem, 3.4vw, 2.9rem) !important;
  line-height: 1.08 !important;
  margin-bottom: 0.4rem !important;
}
h2, h3 { font-size: 1.3rem !important; }

p, li, [data-testid="stMarkdownContainer"] { color: var(--ink-soft); font-size: 0.95rem; }
[data-testid="stMarkdownContainer"] a,
[data-testid="stCaptionContainer"] a {
  color: var(--oxblood) !important;
  text-decoration: underline;
  text-decoration-color: rgba(142, 34, 49, 0.35);
  text-underline-offset: 3px;
}
[data-testid="stMarkdownContainer"] a:hover { text-decoration-color: var(--oxblood); }
[data-testid="stMarkdownContainer"] code {
  background: #F1ECE2 !important;
  color: #7A2230 !important;
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 0.05rem 0.3rem;
  font-size: 0.8rem;
}
[data-testid="stMarkdownContainer"] ul { padding-left: 1.15rem; margin: 0.2rem 0 0.6rem; }
[data-testid="stMarkdownContainer"] li { margin-bottom: 0.32rem; }
[data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
  font-size: 0.82rem;
  line-height: 1.55;
}

[data-testid="stImage"] { display: flex; justify-content: center; margin: 1rem 0 0.3rem; }
[data-testid="stImage"] img {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 6px;
  background: #FFFFFF;
}

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--oxblood);
}
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.55rem 0 0; }
.chip {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #3A3A36;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.28rem 0.62rem;
}
.rule { height: 1px; background: var(--line); margin: 1.6rem 0 1.1rem; }

.section-head { margin: 1.5rem 0 0.6rem; }
.section-head .n {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--oxblood);
}
.section-head h3 { margin: 0.12rem 0 0; font-size: 1.25rem !important; }
.group-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.64rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--sage);
  margin: 0.1rem 0 0.5rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card);
  border: 1px solid var(--line) !important;
  border-radius: 14px;
  box-shadow: 0 1px 0 rgba(23, 25, 27, 0.03), 0 18px 40px -34px rgba(23, 25, 27, 0.4);
}
[data-testid="stForm"] { border: 0; padding: 0; }

[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}
[data-testid="stMetricLabel"] p {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.66rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] p {
  font-family: "Source Serif 4", Georgia, serif !important;
  font-size: 1.85rem !important;
  line-height: 1.12 !important;
  font-variant-numeric: tabular-nums;
  color: var(--ink) !important;
  margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 0.6rem 0.1rem;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; }

.stButton > button, [data-testid="stFormSubmitButton"] button {
  width: 100%;
  background: var(--ink) !important;
  color: #FBF8F2 !important;
  border: 1px solid var(--ink) !important;
  border-radius: 8px;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  padding: 0.72rem 1rem;
}
.stButton > button p, [data-testid="stFormSubmitButton"] button p,
.stButton > button [data-testid="stMarkdownContainer"],
[data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] {
  color: inherit !important;
  font-family: inherit !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.18em;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  background: var(--oxblood) !important;
  border-color: var(--oxblood) !important;
  color: #FFFFFF !important;
}

[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label p {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.64rem !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
}

[data-testid="stNumberInput"] input {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.84rem;
  font-variant-numeric: tabular-nums;
}

[data-testid="stAlert"] { background: transparent !important; border: none !important; }
[data-testid="stAlertContainer"] {
  background: #FBF4EC !important;
  border: 1px solid #E7D6CF !important;
  border-left: 3px solid var(--oxblood) !important;
  border-radius: 10px;
}
[data-testid="stAlertContainer"] p { color: #4A3330 !important; font-size: 0.86rem; }

[data-testid="stTable"] table {
  width: 100%;
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}
[data-testid="stTable"] thead th {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  border-bottom: 1px solid var(--ink);
  padding: 0.45rem 0.6rem;
}
[data-testid="stTable"] tbody td {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.8rem;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--line);
  padding: 0.5rem 0.6rem;
}
[data-testid="stTable"] tbody td:first-child { font-family: "IBM Plex Sans", sans-serif; }
[data-testid="stTable"] tbody td:not(:first-child),
[data-testid="stTable"] thead th:not(:first-child) { text-align: right; }

.hint {
  border: 1px dashed var(--line);
  background: var(--card);
  border-radius: 10px;
  padding: 0.8rem 1rem;
  font-size: 0.86rem;
  color: var(--muted);
}
.hint b { color: var(--ink); }

.footer {
  border-top: 1px solid var(--line);
  margin-top: 2.4rem;
  padding-top: 1rem;
  font-size: 0.78rem;
  color: var(--muted);
}
.footer a { color: var(--oxblood); text-decoration: none; }
</style>
"""


@st.cache_resource
def load_model() -> PPGRModel:
    return PPGRModel()


@st.cache_data
def load_external() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL_CSV)


def _external_table(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out Pearson r by protocol × target, on the five shared features."""
    core = df[df["feature_set"] == "core (5 features)"]
    data: dict[str, dict[str, float]] = {}
    for label, cohort, model in EXTERNAL_ROWS:
        sub = core[(core["cohort"] == cohort) & (core["model"] == model)]
        row: dict[str, float] = {}
        for target in ("auc", "iauc", "peak_rise"):
            vals = sub.loc[sub["target"] == target, "pearson_r"]
            row[TARGET_LABELS[target].split(" (")[0]] = (
                float(vals.iloc[0]) if len(vals) else float("nan")
            )
        data[label] = row
    frame = pd.DataFrame.from_dict(data, orient="index")
    return frame.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}")


def _section(number: str, title: str) -> None:
    st.markdown(
        f'<div class="section-head"><div class="n">{number}</div>'
        f"<h3>{title}</h3></div>",
        unsafe_allow_html=True,
    )


def _group(label: str) -> None:
    st.markdown(f'<div class="group-label">{label}</div>', unsafe_allow_html=True)


def render_external() -> None:
    """Read-only summary of the cross-cohort (BIG IDEAs) validation."""
    if not os.path.exists(EXTERNAL_CSV):
        st.info("External-validation results are not bundled with this build.")
        return

    st.markdown(
        "The CGMacros model above was **frozen** — no retraining — and applied to "
        "**BIG IDEAs**, an independent cohort (16 subjects, 656 meals, a different "
        "CGM device and free-living food logs). Only the five features shared by "
        "both datasets are used (carbohydrate, protein, baseline glucose, sex, HbA1c)."
    )
    st.table(_external_table(load_external()), border="horizontal", width="stretch")
    st.caption("Pearson r between observed and predicted values. Higher is better.")
    if os.path.exists(EXTERNAL_FIG):
        st.image(
            EXTERNAL_FIG,
            caption="Figure 5 — external validation (technical report §4.4).",
        )
    st.markdown(
        "Read it honestly: **AUC still transfers** (r ≈ 0.57), but **iAUC and peak "
        "rise drop** to r ≈ 0.22, and a carbohydrate-only Ridge regression transfers "
        "*better* on those two targets (r ≈ 0.36). The BIG IDEAs internal LOPO row "
        "(0.46) shows the drop is a shift between cohorts, not noise in the external "
        "data — exactly the kind of result only an external test can expose."
    )


def _number_input(name: str, label: str, bounds: dict, medians: dict, step: float) -> float:
    lo, hi = bounds.get(name, [0.0, 100.0])
    lo = float(np.floor(lo))
    hi = float(np.ceil(hi))
    if hi <= lo:
        hi = lo + 1.0
    default = float(np.clip(medians.get(name, lo), lo, hi))
    return st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)


def main() -> None:
    model = load_model()
    meta = model.meta
    lopo = meta["lopo_metrics"]
    bounds = meta["ui"]["bounds"]
    medians = meta["medians"]

    st.markdown(CSS, unsafe_allow_html=True)

    # ---- masthead --------------------------------------------------------
    st.markdown(
        '<div class="eyebrow">MetaNutri · Research module — PPGR</div>',
        unsafe_allow_html=True,
    )
    st.title("Postprandial glucose-response prediction")
    st.markdown(
        '<div class="chips">'
        f'<span class="chip">{meta["n_subjects"]} subjects</span>'
        f'<span class="chip">{meta["n_meals"]:,} meals</span>'
        '<span class="chip">Leave-one-subject-out</span>'
        '<span class="chip">XGBoost · depth 1</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.warning(
        "**Not a medical device.** This is a research/portfolio demo on a small "
        "cohort. Predictions are population-level estimates, not personal "
        "medical advice, and the reconstructed curve is illustrative only — the "
        "models predict the three scalar metrics, not a glucose trace."
    )

    tab_predict, tab_validation, tab_method = st.tabs(
        ["Predict", "Validation", "Method & data"]
    )

    # ---- predict ---------------------------------------------------------
    with tab_predict:
        with st.form("inputs"):
            with st.container(border=True):
                _section("01", "Meal composition")
                meal_cols = st.columns(5)
                meal = {}
                for col, name in zip(meal_cols, MEAL_FIELDS, strict=False):
                    with col:
                        meal[name] = _number_input(
                            name, MEAL_LABELS[name], bounds, medians, 1.0
                        )

            with st.container(border=True):
                _section("02", "Subject panel")
                s1, s2, s3 = st.columns(3)
                subject = {}
                with s1:
                    _group("Anthropometry")
                    subject["age"] = _number_input(
                        "age", SUBJECT_LABELS["age"], bounds, medians, 1.0
                    )
                    sex = st.selectbox("Sex", ["Female", "Male"])
                    subject["gender"] = -1.0 if sex == "Female" else 1.0
                    subject["bmi"] = _number_input(
                        "bmi", SUBJECT_LABELS["bmi"], bounds, medians, 0.1
                    )
                    subject["a1c"] = _number_input(
                        "a1c", SUBJECT_LABELS["a1c"], bounds, medians, 0.1
                    )
                with s2:
                    _group("Glycaemia")
                    subject["fasting_bg"] = _number_input(
                        "fasting_bg", SUBJECT_LABELS["fasting_bg"], bounds, medians, 1.0
                    )
                    subject["insulin"] = _number_input(
                        "insulin", SUBJECT_LABELS["insulin"], bounds, medians, 0.1
                    )
                    subject["tg"] = _number_input(
                        "tg", SUBJECT_LABELS["tg"], bounds, medians, 1.0
                    )
                    subject["cholesterol"] = _number_input(
                        "cholesterol", SUBJECT_LABELS["cholesterol"], bounds, medians, 1.0
                    )
                with s3:
                    _group("Lipid panel")
                    subject["hdl"] = _number_input(
                        "hdl", SUBJECT_LABELS["hdl"], bounds, medians, 1.0
                    )
                    subject["non_hdl"] = _number_input(
                        "non_hdl", SUBJECT_LABELS["non_hdl"], bounds, medians, 1.0
                    )
                    subject["ldl"] = _number_input(
                        "ldl", SUBJECT_LABELS["ldl"], bounds, medians, 1.0
                    )
                    subject["vldl"] = _number_input(
                        "vldl", SUBJECT_LABELS["vldl"], bounds, medians, 0.1
                    )
                    subject["cho_hdl_ratio"] = _number_input(
                        "cho_hdl_ratio",
                        SUBJECT_LABELS["cho_hdl_ratio"],
                        bounds,
                        medians,
                        0.1,
                    )

            submitted = st.form_submit_button(
                "Predict", type="primary", width="stretch"
            )

        if not submitted:
            st.markdown(
                '<div class="hint">Adjust the inputs and press <b>Predict</b> '
                "to run the frozen model.</div>",
                unsafe_allow_html=True,
            )
        else:
            values = PPGRModel.build_inputs(meal, subject)
            preds = model.predict(values)

            st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
            _section("03", "Predicted response")
            c1, c2, c3 = st.columns(3)
            for col, target in zip((c1, c2, c3), ("auc", "iauc", "peak_rise"), strict=False):
                col.metric(
                    TARGET_LABELS[target].split(" (")[0],
                    f"{preds[target]:,.0f}",
                    help=f"Held-out-subject Pearson r = {lopo[target]['pearson_r']:.2f}.",
                )
            st.caption(f"HOMA-IR (derived) = {values['homa_ir']:.2f}")

            t, glucose, tp_eff = model.curve(
                baseline=float(meal["baseline_glucose"]),
                peak_rise=preds["peak_rise"],
                iauc=preds["iauc"],
            )
            frame = pd.DataFrame(
                {
                    "Predicted glucose": glucose,
                    "Meal baseline": np.full_like(
                        glucose, float(meal["baseline_glucose"])
                    ),
                },
                index=t.astype(int),
            )
            frame.index.name = "minutes after meal"
            _section("04", "Illustrative 2-hour response")
            st.line_chart(frame, color=["#8E2231", "#9AA0A6"], height=320)
            st.caption(
                "**Illustrative reconstruction, not a model output.** The models predict "
                "the three scalars above; this curve is a population-average gamma shape "
                f"scaled to the predicted peak rise and iAUC (implied peak ≈ {tp_eff:.0f} min). "
                "Use it to read the magnitude and shape, not as a forecast."
            )

            _section("05", "Why this prediction — TreeSHAP")
            target = st.radio(
                "Explanation for",
                options=list(TARGET_LABELS),
                format_func=lambda t: TARGET_LABELS[t].split(" (")[0],
                horizontal=True,
            )
            contribs = model.explain(values, target)
            top = contribs[:12]
            shap_frame = pd.DataFrame(
                {"contribution": [v for _, v in top]},
                index=[SHAP_LABELS.get(n, n) for n, _ in top],
            )
            st.bar_chart(shap_frame, horizontal=True, color="#8E2231", height=360)
            st.caption(
                "Exact TreeSHAP values in the target's units. Positive contributions push "
                "the prediction up; negative push it down. Ranked by absolute magnitude."
            )

    # ---- validation ------------------------------------------------------
    with tab_validation:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        _section("06", "Held-out-subject performance")
        c1, c2, c3 = st.columns(3)
        for col, target in zip((c1, c2, c3), ("auc", "iauc", "peak_rise"), strict=False):
            m = lopo[target]
            col.metric(
                TARGET_LABELS[target].split(" (")[0],
                f"r = {m['pearson_r']:.2f}",
                help=f"Pearson r on {m['n']} held-out meals (R² = {m['r2']:.2f}).",
            )
        st.markdown(
            "Every number comes from **leave-one-subject-out** cross-validation: "
            "the model is always tested on a person it has never seen. AUC is "
            "predicted much better than iAUC, because AUC is dominated by the "
            "person's overall glucose level, while iAUC isolates the meal-driven "
            "excursion."
        )

        _section("07", "External cohort — BIG IDEAs")
        render_external()

    # ---- method ----------------------------------------------------------
    with tab_method:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        _section("08", "Data, model and code")
        st.markdown(
            "- **Cohort.** CGMacros (PhysioNet, DOI 10.13026/3z8q-x658), CC BY-NC-SA 4.0; "
            "45 subjects, 1,557 meals, AbbVie Libre Pro CGM with study-recorded meal notes.\n"
            "- **Model.** XGBoost, `max_depth=1`, `n_estimators=80`, `learning_rate=0.2`; "
            "frozen JSON weight bundles shipped in `model/`.\n"
            "- **Protocol.** Leave-one-subject-out for the internal estimate; "
            "no retraining for the external test on BIG IDEAs.\n"
            "- **Scope.** Three scalar targets only (2-h AUC, 2-h iAUC, peak rise). "
            "The 2-hour curve on the Predict tab is an illustrative reconstruction, "
            "not a model output.\n"
            "- **Licence.** Code MIT; CGMacros data CC BY-NC-SA 4.0 (non-commercial)."
        )
        st.markdown(
            "Repositories: "
            "[`MetaNutri-Ai-ppgr-predictor`](https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor) · "
            "[`MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-) · "
            "[technical report](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/research/reports/technical_report.md) · "
            "[preprint PDF](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/preprint/medrxiv_manuscript.pdf)"
        )

    st.markdown(
        '<div class="footer">MetaNutri ecosystem · PPGR research module. '
        "Model: XGBoost on CGMacros. Data: "
        '<a href="https://doi.org/10.13026/3z8q-x658">PhysioNet DOI 10.13026/3z8q-x658</a>, '
        "CC BY-NC-SA 4.0. Not a medical device.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
