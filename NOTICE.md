# NOTICE — License layering for this repository

This repository contains three kinds of material with different licenses. The
repository-level [MIT license](LICENSE) applies **only** to the first layer.

## 1. Source code — MIT

`app.py`, `inference.py`, `test_app.py` and everything under `src/` is original
work and is licensed under the repository-level MIT license (`LICENSE`,
© 2026 ElijahZhao). You may reuse, modify and redistribute it under MIT terms.

## 2. Data products derived from CGMacros — CC BY-NC-SA 4.0

CGMacros (PhysioNet, DOI `10.13026/3z8q-x658`) is licensed
**CC BY-NC-SA 4.0**. The ShareAlike and NonCommercial terms propagate to
derivative works, therefore the following artifacts are distributed under
**CC BY-NC-SA 4.0**, not MIT:

- `model/*.json` — the frozen XGBoost boosters trained on CGMacros
- `experiments/*.csv` — result tables computed from CGMacros and BIG IDEAs
- `assets/*` — figures and derived tables bundled for the demo
- `reports/figures/*` — all figures rendered from those data
- `reports/technical_report.md` and its PDF export — derivative work of both
  datasets (as declared in the report's own §10)

Use of these artifacts is restricted to non-commercial purposes (research,
education, portfolio) and must carry attribution to the CGMacros authors
(Das et al., *Scientific Data* 12, 1557, 2025) plus share-alike terms.
Raw CGMacros data is **never** redistributed in this repository;
`src/download_data.py` fetches it from PhysioNet.

## 3. Datasets themselves — governed by their upstream licenses

- **CGMacros**: CC BY-NC-SA 4.0 — see `data/README.md` for download,
  SHA256 verification and attribution details.
- **BIG IDEAs Lab Glycemic Variability and Wearable Device Data**
  (PhysioNet, DOI `10.13026/zthx-5212`): **ODC-By 1.0** — attribution
  required (Cho, Kim, Bent & Dunn). This permissive license applies to
  BIG IDEAs-derived files; where a product mixes both cohorts (e.g.
  `experiments/external_results.csv`), the stricter CGMacros terms above
  apply to the combined artifact.

---

This repository is a curated deployment of the research module in
[`ElijahZhao/MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-).
The same layering is documented there in `research/NOTICE.md`.
Questions about this layering: open an issue on either repository.
