"""Streamlit demo — postprandial glucose-response (PPGR) prediction.

Research demo for the MetaNutri project's research module. Trained on CGMacros
(45 subjects, 1,557 meals) with leave-one-subject-out cross-validation.
Not a clinical tool.

The layout follows Streamlit's own theming and layout primitives: a
``.streamlit/config.toml`` theme for colour and radius, native containers,
tabs and metrics for structure, and a small stylesheet for typography only.
The interface is bilingual (English / Simplified Chinese): the switch lives in
the sidebar and every string is routed through ``STRINGS``.
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

REPO_THIS = "https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor"
REPO_HUB = "https://github.com/ElijahZhao/MetaNutri---AI-"
REPO_RAG = "https://github.com/ElijahZhao/MetaNutri-Ai-RAG"
URL_REPORT = (
    "https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/research/reports/"
    "technical_report.md"
)
URL_PREPRINT = (
    "https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/preprint/"
    "medrxiv_manuscript.pdf"
)

st.set_page_config(
    page_title="PPGR Predictor — MetaNutri research demo",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="auto",
)

LANGS = ("EN", "中文")
LANG = {"code": "en"}

STRINGS = {
    "masthead_eyebrow": {
        "en": "MetaNutri · Research module — PPGR",
        "zh": "MetaNutri · 研究模块 — PPGR",
    },
    "title": {
        "en": "Postprandial glucose-response prediction",
        "zh": "餐后血糖响应预测",
    },
    "chip_subjects": {"en": "subjects", "zh": "名受试者"},
    "chip_meals": {"en": "meals", "zh": "餐"},
    "chip_lopo": {"en": "Leave-one-subject-out", "zh": "留一受试者交叉验证"},
    "chip_model": {"en": "XGBoost · depth 1", "zh": "XGBoost · 深度 1"},
    "warning": {
        "en": "**Not a medical device.** This is a research/portfolio demo on a "
        "small cohort. Predictions are population-level estimates, not personal "
        "medical advice, and the reconstructed curve is illustrative only — the "
        "models predict the three scalar metrics, not a glucose trace.",
        "zh": "**非医疗器械。** 这是基于小型队列的研究／作品集演示。预测为人群层面的"
        "估计，不构成个人医疗建议；重建曲线仅作示意——模型预测的是三个标量指标，"
        "而非血糖轨迹。",
    },
    "tab_predict": {"en": "Predict", "zh": "预测"},
    "tab_validation": {"en": "Validation", "zh": "验证"},
    "tab_method": {"en": "Method & data", "zh": "方法与数据"},
    "sec_meal": {"en": "Meal composition", "zh": "餐食构成"},
    "sec_subject": {"en": "Subject panel", "zh": "受试者面板"},
    "sec_predicted": {"en": "Predicted response", "zh": "预测响应"},
    "sec_curve": {"en": "Illustrative 2-hour response", "zh": "示意性 2 小时响应"},
    "sec_shap": {"en": "Why this prediction — TreeSHAP", "zh": "预测依据 — TreeSHAP"},
    "sec_validation": {"en": "Held-out-subject performance", "zh": "留一受试者表现"},
    "sec_external": {"en": "External cohort — BIG IDEAs", "zh": "外部队列 — BIG IDEAs"},
    "sec_method": {"en": "Data, model and code", "zh": "数据、模型与代码"},
    "group_anthro": {"en": "Anthropometry", "zh": "人体测量"},
    "group_glyc": {"en": "Glycaemia", "zh": "血糖指标"},
    "group_lipid": {"en": "Lipid panel", "zh": "血脂面板"},
    "sex": {"en": "Sex", "zh": "性别"},
    "btn_predict": {"en": "Predict", "zh": "预测"},
    "hint": {
        "en": "Adjust the inputs and press <b>Predict</b> to run the frozen model.",
        "zh": "调整输入后点击 <b>预测</b>，运行冻结模型。",
    },
    "metric_help_pred": {
        "en": "Held-out-subject Pearson r = {r:.2f}.",
        "zh": "留一受试者 Pearson r = {r:.2f}。",
    },
    "homa": {
        "en": "HOMA-IR (derived) = {v:.2f}",
        "zh": "HOMA-IR（推导值）= {v:.2f}",
    },
    "chart_pred": {"en": "Predicted glucose", "zh": "预测血糖"},
    "chart_base": {"en": "Meal baseline", "zh": "餐前基线"},
    "chart_index": {"en": "minutes after meal", "zh": "餐后分钟"},
    "curve_caption": {
        "en": "**Illustrative reconstruction, not a model output.** The models "
        "predict the three scalars above; this curve is a population-average "
        "gamma shape scaled to the predicted peak rise and iAUC (implied peak ≈ "
        "{tp:.0f} min). Use it to read the magnitude and shape, not as a forecast.",
        "zh": "**示意性重建，并非模型输出。** 模型预测的是上方三个标量；此曲线为人群"
        "平均 gamma 形状，按预测的峰值升幅与 iAUC 缩放（隐含峰值 ≈ {tp:.0f} 分钟）。"
        "可用它读取幅度与形状，但不可当作预测。",
    },
    "radio_label": {"en": "Explanation for", "zh": "解释目标"},
    "shap_caption": {
        "en": "Exact TreeSHAP values in the target's units. Positive contributions "
        "push the prediction up; negative push it down. Ranked by absolute "
        "magnitude.",
        "zh": "目标量单位下的精确 TreeSHAP 值。正贡献推高预测，负贡献拉低预测；"
        "按绝对值排序。",
    },
    "val_help": {
        "en": "Pearson r on {n} held-out meals (R² = {r2:.2f}).",
        "zh": "基于 {n} 餐留出样本的 Pearson r（R² = {r2:.2f}）。",
    },
    "val_para": {
        "en": "Every number comes from **leave-one-subject-out** cross-validation: "
        "the model is always tested on a person it has never seen. AUC is "
        "predicted much better than iAUC, because AUC is dominated by the "
        "person's overall glucose level, while iAUC isolates the meal-driven "
        "excursion.",
        "zh": "所有数字均来自**留一受试者交叉验证**：模型始终在它从未见过的人身上"
        "测试。AUC 的预测效果远好于 iAUC，因为 AUC 主要由个人整体血糖水平决定，"
        "而 iAUC 分离出餐食驱动的波动。",
    },
    "ext_intro": {
        "en": "The CGMacros model above was **frozen** — no retraining — and applied "
        "to **BIG IDEAs**, an independent cohort (16 subjects, 656 meals, a "
        "different CGM device and free-living food logs). Only the five features "
        "shared by both datasets are used (carbohydrate, protein, baseline "
        "glucose, sex, HbA1c).",
        "zh": "上方的 CGMacros 模型被**冻结**——不重新训练——直接应用于独立队列 "
        "**BIG IDEAs**（16 名受试者、656 餐，使用不同的 CGM 设备与自由生活饮食"
        "记录）。仅使用两个数据集共有的五个特征（碳水化合物、蛋白质、基线血糖、"
        "性别、HbA1c）。",
    },
    "ext_caption": {
        "en": "Pearson r between observed and predicted values. Higher is better.",
        "zh": "观测值与预测值的 Pearson r，越高越好。",
    },
    "ext_fig_caption": {
        "en": "Figure 5 — external validation (technical report §4.4).",
        "zh": "图 5 — 外部验证（技术报告 §4.4）。",
    },
    "ext_honesty": {
        "en": "Read it honestly: **AUC still transfers** (r ≈ 0.57), but **iAUC and "
        "peak rise drop** to r ≈ 0.22, and a carbohydrate-only Ridge regression "
        "transfers *better* on those two targets (r ≈ 0.36). The BIG IDEAs "
        "internal LOPO row (0.46) shows the drop is a shift between cohorts, not "
        "noise in the external data — exactly the kind of result only an external "
        "test can expose.",
        "zh": "如实解读：**AUC 仍可迁移**（r ≈ 0.57），但 **iAUC 与峰值升幅下降**至 "
        "r ≈ 0.22，而仅用碳水化合物的 Ridge 回归在这两个目标上迁移得*更好*"
        "（r ≈ 0.36）。BIG IDEAs 内部 LOPO 一行（0.46）说明这种下降来自队列间的"
        "分布偏移，而非外部数据的噪声——这正是只有外部测试才能暴露的结果。",
    },
    "ext_missing": {
        "en": "External-validation results are not bundled with this build.",
        "zh": "本次构建未包含外部验证结果。",
    },
    "method_bullets": {
        "en": "- **Cohort.** CGMacros (PhysioNet, DOI 10.13026/3z8q-x658), CC "
        "BY-NC-SA 4.0; 45 subjects, 1,557 meals, AbbVie Libre Pro CGM with "
        "study-recorded meal notes.\n"
        "- **Model.** XGBoost, `max_depth=1`, `n_estimators=80`, "
        "`learning_rate=0.2`; frozen JSON weight bundles shipped in `model/`.\n"
        "- **Protocol.** Leave-one-subject-out for the internal estimate; "
        "no retraining for the external test on BIG IDEAs.\n"
        "- **Scope.** Three scalar targets only (2-h AUC, 2-h iAUC, peak rise). "
        "The 2-hour curve on the Predict tab is an illustrative reconstruction, "
        "not a model output.\n"
        "- **Licence.** Code MIT; CGMacros data CC BY-NC-SA 4.0 (non-commercial).",
        "zh": "- **队列。** CGMacros（PhysioNet，DOI 10.13026/3z8q-x658），CC "
        "BY-NC-SA 4.0；45 名受试者、1,557 餐，AbbVie Libre Pro CGM 配合研究记录"
        "的餐食备注。\n"
        "- **模型。** XGBoost，`max_depth=1`、`n_estimators=80`、"
        "`learning_rate=0.2`；权重以冻结的 JSON 包随 `model/` 一起发布。\n"
        "- **协议。** 内部估计采用留一受试者交叉验证；在 BIG IDEAs 上的外部测试"
        "不重新训练。\n"
        "- **范围。** 仅三个标量目标（2 小时 AUC、2 小时 iAUC、峰值升幅）。"
        "预测页的 2 小时曲线是示意性重建，不是模型输出。\n"
        "- **许可。** 代码 MIT；CGMacros 数据 CC BY-NC-SA 4.0（非商业）。",
    },
    "repos_line": {
        "en": "Repositories: "
        "[`MetaNutri-Ai-ppgr-predictor`](https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor) · "
        "[`MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-) · "
        "[technical report](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/research/reports/technical_report.md) · "
        "[preprint PDF](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/preprint/medrxiv_manuscript.pdf)",
        "zh": "仓库："
        "[`MetaNutri-Ai-ppgr-predictor`](https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor) · "
        "[`MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-) · "
        "[技术报告](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/research/reports/technical_report.md) · "
        "[预印本 PDF](https://github.com/ElijahZhao/MetaNutri---AI-/blob/main/preprint/medrxiv_manuscript.pdf)",
    },
    "footer": {
        "en": '<div class="footer">MetaNutri ecosystem · PPGR research module. '
        "Model: XGBoost on CGMacros. Data: "
        '<a href="https://doi.org/10.13026/3z8q-x658">PhysioNet DOI 10.13026/3z8q-x658</a>, '
        "CC BY-NC-SA 4.0. Not a medical device.</div>",
        "zh": '<div class="footer">MetaNutri 生态 · PPGR 研究模块。'
        "模型：基于 CGMacros 的 XGBoost。数据："
        '<a href="https://doi.org/10.13026/3z8q-x658">PhysioNet DOI 10.13026/3z8q-x658</a>，'
        "CC BY-NC-SA 4.0。非医疗器械。</div>",
    },
    "side_brand_sub": {"en": "PPGR research module", "zh": "PPGR 研究模块"},
    "side_lang_label": {"en": "Language · 语言", "zh": "Language · 语言"},
    "side_repos_head": {"en": "MetaNutri ecosystem", "zh": "MetaNutri 生态"},
    "side_here_tag": {"en": "You are here", "zh": "当前位置"},
    "side_hub_tag": {"en": "Research hub", "zh": "研究主仓"},
    "side_rag_tag": {"en": "RAG module", "zh": "RAG 模块"},
    "side_this_desc": {
        "en": "This demo — PPGR model, validation and TreeSHAP.",
        "zh": "本演示 — PPGR 模型、验证与 TreeSHAP。",
    },
    "side_hub_desc": {
        "en": "Main repository — paper, technical report and experiment pipeline.",
        "zh": "主仓库 — 论文、技术报告与实验流水线。",
    },
    "side_rag_desc": {
        "en": "Retrieval-augmented health & nutrition Q&A with built-in evaluation.",
        "zh": "带内置评测的健康营养检索问答（RAG）。",
    },
    "side_report": {"en": "technical report ↗", "zh": "技术报告 ↗"},
    "side_preprint": {"en": "preprint ↗", "zh": "预印本 ↗"},
    "side_data_head": {"en": "Data & licence", "zh": "数据与许可"},
    "side_data": {
        "en": "CGMacros (PhysioNet, CC BY-NC-SA 4.0). Code MIT.",
        "zh": "CGMacros（PhysioNet，CC BY-NC-SA 4.0）。代码 MIT。",
    },
    "side_note": {
        "en": "Research demo on a small cohort. Not a medical device.",
        "zh": "基于小型队列的研究演示，非医疗器械。",
    },
    "side_github": {"en": "GitHub ↗", "zh": "GitHub ↗"},
}

METRIC_LABELS = {
    "en": {"auc": "2-h AUC", "iauc": "2-h iAUC", "peak_rise": "Peak glucose rise"},
    "zh": {"auc": "2 小时 AUC", "iauc": "2 小时 iAUC", "peak_rise": "峰值血糖升幅"},
}

MEAL_LABELS = {
    "en": {
        "carbs_g": "Carbohydrates (g)",
        "protein_g": "Protein (g)",
        "fat_g": "Fat (g)",
        "fiber_g": "Fiber (g)",
        "baseline_glucose": "Meal-time glucose (mg/dL)",
    },
    "zh": {
        "carbs_g": "碳水化合物 (g)",
        "protein_g": "蛋白质 (g)",
        "fat_g": "脂肪 (g)",
        "fiber_g": "膳食纤维 (g)",
        "baseline_glucose": "餐时血糖 (mg/dL)",
    },
}

SUBJECT_LABELS = {
    "en": {
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
    },
    "zh": {
        "age": "年龄（岁）",
        "bmi": "BMI（kg/m²）",
        "a1c": "HbA1c（%）",
        "fasting_bg": "空腹血糖 (mg/dL)",
        "insulin": "空腹胰岛素 (µIU/mL)",
        "tg": "甘油三酯 (mg/dL)",
        "cholesterol": "总胆固醇 (mg/dL)",
        "hdl": "HDL (mg/dL)",
        "non_hdl": "非 HDL (mg/dL)",
        "ldl": "LDL (mg/dL)",
        "vldl": "VLDL (mg/dL)",
        "cho_hdl_ratio": "胆固醇 / HDL 比值",
    },
}

SHAP_LABELS = {
    "en": {
        "Age": "Age",
        "BMI": "BMI",
        "Carbohydrates (g)": "Carbohydrates",
        "Fat (g)": "Fat",
        "Fiber (g)": "Fiber",
        "Protein (g)": "Protein",
        "Meal-time glucose (mg/dL)": "Meal-time glucose",
        "Fasting glucose": "Fasting glucose",
        "Fasting insulin (µIU/mL)": "Fasting insulin",
        "HbA1c (%)": "HbA1c",
        "HOMA-IR": "HOMA-IR",
        "HDL": "HDL",
        "LDL": "LDL",
        "Non-HDL": "Non-HDL",
        "Total cholesterol": "Total cholesterol",
        "Triglycerides": "Triglycerides",
        "Chol/HDL ratio": "Chol/HDL ratio",
        "Sex (M=1/F=−1)": "Sex",
    },
    "zh": {
        "Age": "年龄",
        "BMI": "BMI",
        "Carbohydrates (g)": "碳水化合物",
        "Fat (g)": "脂肪",
        "Fiber (g)": "膳食纤维",
        "Protein (g)": "蛋白质",
        "Meal-time glucose (mg/dL)": "餐时血糖",
        "Fasting glucose": "空腹血糖",
        "Fasting insulin (µIU/mL)": "空腹胰岛素",
        "HbA1c (%)": "HbA1c",
        "HOMA-IR": "HOMA-IR",
        "HDL": "HDL",
        "LDL": "LDL",
        "Non-HDL": "非 HDL",
        "Total cholesterol": "总胆固醇",
        "Triglycerides": "甘油三酯",
        "Chol/HDL ratio": "胆固醇/HDL 比值",
        "Sex (M=1/F=−1)": "性别",
    },
}

SEX_OPTIONS = {"en": ["Female", "Male"], "zh": ["女性", "男性"]}

# Rows shown in the external-validation panel: (key, cohort, model).
EXTERNAL_ROWS = [
    ("cg", "CGMacros (LOPO, internal)", "xgboost"),
    ("bi", "BIG IDEAs (LOPO, internal)", "xgboost"),
    ("bx", "BIG IDEAs (external transfer)", "xgboost"),
    ("br", "BIG IDEAs (external transfer)", "carb-only linear"),
]
EXTERNAL_LABELS = {
    "cg": {
        "en": "CGMacros (LOPO, internal)",
        "zh": "CGMacros（LOPO，内部）",
    },
    "bi": {
        "en": "BIG IDEAs (LOPO, internal)",
        "zh": "BIG IDEAs（LOPO，内部）",
    },
    "bx": {
        "en": "BIG IDEAs (external, XGBoost)",
        "zh": "BIG IDEAs（外部，XGBoost）",
    },
    "br": {
        "en": "BIG IDEAs (external, carb-only Ridge)",
        "zh": "BIG IDEAs（外部，仅碳水 Ridge）",
    },
}


def t(key: str) -> str:
    """Look up a UI string in the active language."""
    return STRINGS[key][LANG["code"]]


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
  --sans: "IBM Plex Sans", "Source Sans", "Noto Sans SC", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --serif: "Source Serif 4", Georgia, "Times New Roman", "Noto Serif SC",
    "Songti SC", serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
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
  font-family: var(--sans);
}

.block-container {
  max-width: 1160px;
  padding-top: 2.4rem;
  padding-bottom: 3.5rem;
}

h1, h2, h3 {
  font-family: var(--serif) !important;
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
  font-family: var(--mono);
  font-size: 0.7rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--oxblood);
}
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.55rem 0 0; }
.chip {
  font-family: var(--mono);
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
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--oxblood);
}
.section-head h3 { margin: 0.12rem 0 0; font-size: 1.25rem !important; }
.group-label {
  font-family: var(--mono);
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
  font-family: var(--mono);
  font-size: 0.66rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] p {
  font-family: var(--serif) !important;
  font-size: 1.85rem !important;
  line-height: 1.12 !important;
  font-variant-numeric: tabular-nums;
  color: var(--ink) !important;
  margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono);
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
  font-family: var(--mono) !important;
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
  font-family: var(--mono);
  font-size: 0.64rem !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
}

[data-testid="stNumberInput"] input {
  font-family: var(--mono) !important;
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
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  border-bottom: 1px solid var(--ink);
  padding: 0.45rem 0.6rem;
}
[data-testid="stTable"] tbody td {
  font-family: var(--mono);
  font-size: 0.8rem;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--line);
  padding: 0.5rem 0.6rem;
}
[data-testid="stTable"] tbody td:first-child { font-family: var(--sans); }
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

/* ---- sidebar ---------------------------------------------------------- */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top: 1.6rem;
  padding-bottom: 2rem;
}
.side-brand {
  font-family: var(--serif);
  font-size: 1.32rem;
  line-height: 1.1;
  color: var(--ink);
}
.side-brand span {
  display: block;
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--oxblood);
  margin-top: 0.32rem;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-family: var(--mono);
  font-size: 0.6rem !important;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
  font-family: var(--mono) !important;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--line) !important;
  border-radius: 6px;
  padding: 0.3rem 0.5rem;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.12em;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[data-selected="true"],
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
  color: #FBF8F2 !important;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[data-selected="true"] p,
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-checked="true"] p {
  color: #FBF8F2 !important;
}

.side-head {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--sage);
  border-top: 1px solid var(--line);
  padding-top: 0.9rem;
  margin: 1.15rem 0 0.45rem;
}
.side-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.62rem 0.72rem 0.56rem;
  margin-bottom: 0.5rem;
}
.side-card.current { border-left: 2px solid var(--oxblood); }
.side-card .tag {
  font-family: var(--mono);
  font-size: 0.56rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--sage);
}
.side-card.current .tag { color: var(--oxblood); }
.side-card .name {
  font-family: var(--mono);
  font-size: 0.76rem;
  color: var(--ink);
  margin: 0.14rem 0 0.24rem;
  word-break: break-all;
}
.side-card .desc {
  font-size: 0.74rem;
  line-height: 1.45;
  color: var(--muted);
  margin-bottom: 0.34rem;
}
.side-card .links a {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.08em;
  color: var(--oxblood);
  text-decoration: none;
  margin-right: 0.6rem;
}
.side-card .links a:hover { text-decoration: underline; }
.side-note {
  font-size: 0.72rem;
  line-height: 1.5;
  color: var(--muted);
}
.side-note.dim { font-size: 0.68rem; margin-top: 0.55rem; }
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
    lang = LANG["code"]
    core = df[df["feature_set"] == "core (5 features)"]
    data: dict[str, dict[str, float]] = {}
    for key, cohort, model in EXTERNAL_ROWS:
        sub = core[(core["cohort"] == cohort) & (core["model"] == model)]
        row: dict[str, float] = {}
        for target in ("auc", "iauc", "peak_rise"):
            vals = sub.loc[sub["target"] == target, "pearson_r"]
            row[METRIC_LABELS[lang][target]] = (
                float(vals.iloc[0]) if len(vals) else float("nan")
            )
        data[EXTERNAL_LABELS[key][lang]] = row
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


def _side_card(
    tag: str,
    name: str,
    desc: str,
    links: list[tuple[str, str]],
    *,
    current: bool = False,
) -> None:
    anchors = "".join(f'<a href="{href}">{label}</a>' for label, href in links)
    cls = "side-card current" if current else "side-card"
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="tag">{tag}</div>'
        f'<div class="name">{name}</div>'
        f'<div class="desc">{desc}</div>'
        f'<div class="links">{anchors}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Language switch plus the MetaNutri repository map."""
    with st.sidebar:
        st.markdown(
            f'<div class="side-brand">MetaNutri<span>{t("side_brand_sub")}</span></div>',
            unsafe_allow_html=True,
        )
        st.segmented_control(
            t("side_lang_label"),
            options=list(LANGS),
            default="EN",
            key="lang_choice",
            width="stretch",
        )

        st.markdown(
            f'<div class="side-head">{t("side_repos_head")}</div>',
            unsafe_allow_html=True,
        )
        _side_card(
            t("side_here_tag"),
            "MetaNutri-Ai-ppgr-predictor",
            t("side_this_desc"),
            [("GitHub ↗", REPO_THIS)],
            current=True,
        )
        _side_card(
            t("side_hub_tag"),
            "MetaNutri---AI-",
            t("side_hub_desc"),
            [
                ("GitHub ↗", REPO_HUB),
                (t("side_report"), URL_REPORT),
                (t("side_preprint"), URL_PREPRINT),
            ],
        )
        _side_card(
            t("side_rag_tag"),
            "MetaNutri-Ai-RAG",
            t("side_rag_desc"),
            [("GitHub ↗", REPO_RAG)],
        )

        st.markdown(
            f'<div class="side-head">{t("side_data_head")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="side-note">{t("side_data")}</div>'
            f'<div class="side-note dim">{t("side_note")}</div>',
            unsafe_allow_html=True,
        )


def render_external() -> None:
    """Read-only summary of the cross-cohort (BIG IDEAs) validation."""
    if not os.path.exists(EXTERNAL_CSV):
        st.info(t("ext_missing"))
        return

    st.markdown(t("ext_intro"))
    st.table(_external_table(load_external()), border="horizontal", width="stretch")
    st.caption(t("ext_caption"))
    if os.path.exists(EXTERNAL_FIG):
        st.image(EXTERNAL_FIG, caption=t("ext_fig_caption"))
    st.markdown(t("ext_honesty"))


def _number_input(name: str, label: str, bounds: dict, medians: dict, step: float) -> float:
    lo, hi = bounds.get(name, [0.0, 100.0])
    lo = float(np.floor(lo))
    hi = float(np.ceil(hi))
    if hi <= lo:
        hi = lo + 1.0
    default = float(np.clip(medians.get(name, lo), lo, hi))
    return st.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=default,
        step=step,
        key=f"in_{name}",
    )


def _current_code() -> str:
    """Read the sidebar switch from session state before rendering."""
    return "zh" if st.session_state.get("lang_choice") == "中文" else "en"


def main() -> None:
    LANG["code"] = _current_code()
    lang = LANG["code"]

    model = load_model()
    meta = model.meta
    lopo = meta["lopo_metrics"]
    bounds = meta["ui"]["bounds"]
    medians = meta["medians"]

    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    # ---- masthead --------------------------------------------------------
    st.markdown(
        f'<div class="eyebrow">{t("masthead_eyebrow")}</div>',
        unsafe_allow_html=True,
    )
    st.title(t("title"))
    st.markdown(
        '<div class="chips">'
        f'<span class="chip">{meta["n_subjects"]} {t("chip_subjects")}</span>'
        f'<span class="chip">{meta["n_meals"]:,} {t("chip_meals")}</span>'
        f'<span class="chip">{t("chip_lopo")}</span>'
        f'<span class="chip">{t("chip_model")}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.warning(t("warning"))

    tab_predict, tab_validation, tab_method = st.tabs(
        [t("tab_predict"), t("tab_validation"), t("tab_method")]
    )

    # ---- predict ---------------------------------------------------------
    with tab_predict:
        with st.form("inputs"):
            with st.container(border=True):
                _section("01", t("sec_meal"))
                meal_cols = st.columns(5)
                meal = {}
                for col, name in zip(meal_cols, MEAL_FIELDS, strict=False):
                    with col:
                        meal[name] = _number_input(
                            name, MEAL_LABELS[lang][name], bounds, medians, 1.0
                        )

            with st.container(border=True):
                _section("02", t("sec_subject"))
                s1, s2, s3 = st.columns(3)
                subject = {}
                with s1:
                    _group(t("group_anthro"))
                    subject["age"] = _number_input(
                        "age", SUBJECT_LABELS[lang]["age"], bounds, medians, 1.0
                    )
                    sex = st.selectbox(t("sex"), SEX_OPTIONS[lang], key="sex")
                    subject["gender"] = (
                        -1.0 if SEX_OPTIONS[lang].index(sex) == 0 else 1.0
                    )
                    subject["bmi"] = _number_input(
                        "bmi", SUBJECT_LABELS[lang]["bmi"], bounds, medians, 0.1
                    )
                    subject["a1c"] = _number_input(
                        "a1c", SUBJECT_LABELS[lang]["a1c"], bounds, medians, 0.1
                    )
                with s2:
                    _group(t("group_glyc"))
                    subject["fasting_bg"] = _number_input(
                        "fasting_bg",
                        SUBJECT_LABELS[lang]["fasting_bg"],
                        bounds,
                        medians,
                        1.0,
                    )
                    subject["insulin"] = _number_input(
                        "insulin",
                        SUBJECT_LABELS[lang]["insulin"],
                        bounds,
                        medians,
                        0.1,
                    )
                    subject["tg"] = _number_input(
                        "tg", SUBJECT_LABELS[lang]["tg"], bounds, medians, 1.0
                    )
                    subject["cholesterol"] = _number_input(
                        "cholesterol",
                        SUBJECT_LABELS[lang]["cholesterol"],
                        bounds,
                        medians,
                        1.0,
                    )
                with s3:
                    _group(t("group_lipid"))
                    subject["hdl"] = _number_input(
                        "hdl", SUBJECT_LABELS[lang]["hdl"], bounds, medians, 1.0
                    )
                    subject["non_hdl"] = _number_input(
                        "non_hdl", SUBJECT_LABELS[lang]["non_hdl"], bounds, medians, 1.0
                    )
                    subject["ldl"] = _number_input(
                        "ldl", SUBJECT_LABELS[lang]["ldl"], bounds, medians, 1.0
                    )
                    subject["vldl"] = _number_input(
                        "vldl", SUBJECT_LABELS[lang]["vldl"], bounds, medians, 0.1
                    )
                    subject["cho_hdl_ratio"] = _number_input(
                        "cho_hdl_ratio",
                        SUBJECT_LABELS[lang]["cho_hdl_ratio"],
                        bounds,
                        medians,
                        0.1,
                    )

            submitted = st.form_submit_button(
                t("btn_predict"), type="primary", width="stretch"
            )

        if not submitted:
            st.markdown(f'<div class="hint">{t("hint")}</div>', unsafe_allow_html=True)
        else:
            values = PPGRModel.build_inputs(meal, subject)
            preds = model.predict(values)

            st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
            _section("03", t("sec_predicted"))
            c1, c2, c3 = st.columns(3)
            for col, target in zip((c1, c2, c3), ("auc", "iauc", "peak_rise"), strict=False):
                col.metric(
                    METRIC_LABELS[lang][target],
                    f"{preds[target]:,.0f}",
                    help=t("metric_help_pred").format(r=lopo[target]["pearson_r"]),
                )
            st.caption(t("homa").format(v=values["homa_ir"]))

            tt, glucose, tp_eff = model.curve(
                baseline=float(meal["baseline_glucose"]),
                peak_rise=preds["peak_rise"],
                iauc=preds["iauc"],
            )
            frame = pd.DataFrame(
                {
                    t("chart_pred"): glucose,
                    t("chart_base"): np.full_like(
                        glucose, float(meal["baseline_glucose"])
                    ),
                },
                index=tt.astype(int),
            )
            frame.index.name = t("chart_index")
            _section("04", t("sec_curve"))
            st.line_chart(frame, color=["#8E2231", "#9AA0A6"], height=320)
            st.caption(t("curve_caption").format(tp=tp_eff))

            _section("05", t("sec_shap"))
            target = st.radio(
                t("radio_label"),
                options=list(TARGET_LABELS),
                format_func=lambda key: METRIC_LABELS[lang][key],
                horizontal=True,
                key="shap_target",
            )
            contribs = model.explain(values, target)
            top = contribs[:12]
            shap_frame = pd.DataFrame(
                {"contribution": [v for _, v in top]},
                index=[SHAP_LABELS[lang].get(n, n) for n, _ in top],
            )
            st.bar_chart(shap_frame, horizontal=True, color="#8E2231", height=360)
            st.caption(t("shap_caption"))

    # ---- validation ------------------------------------------------------
    with tab_validation:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        _section("06", t("sec_validation"))
        c1, c2, c3 = st.columns(3)
        for col, target in zip((c1, c2, c3), ("auc", "iauc", "peak_rise"), strict=False):
            m = lopo[target]
            col.metric(
                METRIC_LABELS[lang][target],
                f"r = {m['pearson_r']:.2f}",
                help=t("val_help").format(n=m["n"], r2=m["r2"]),
            )
        st.markdown(t("val_para"))

        _section("07", t("sec_external"))
        render_external()

    # ---- method ----------------------------------------------------------
    with tab_method:
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        _section("08", t("sec_method"))
        st.markdown(t("method_bullets"))
        st.markdown(t("repos_line"))

    st.markdown(t("footer"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
