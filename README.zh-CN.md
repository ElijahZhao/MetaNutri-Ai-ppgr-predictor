# PPGR Predictor

<div align="center">

<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

**餐后血糖反应（PPGR）预测 —— XGBoost + 留一被试交叉验证**

[![在线演示](https://img.shields.io/badge/Streamlit-%E5%9C%A8%E7%BA%BF%E6%BC%94%E7%A4%BA-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://metanutri-ai-ppgr-predictor.streamlit.app/)
[![CI](https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ElijahZhao/MetaNutri-Ai-ppgr-predictor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB6E4B?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License: MIT（代码）](https://img.shields.io/badge/License-MIT_%28%E4%BB%A3%E7%A0%81%29-yellow?style=for-the-badge)](LICENSE)
[![License: CC BY-NC-SA 4.0（衍生数据）](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0_%28%E8%A1%8D%E7%94%9F%E6%95%B0%E6%8D%AE%29-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![父仓库](https://img.shields.io/badge/%E7%88%B6%E4%BB%93%E5%BA%93-MetaNutri--AI--10b981?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ElijahZhao/MetaNutri---AI-)

</div>

根据一餐的**宏量营养素构成**加上少量**被试级临床特征**，预测餐后 **2 小时血糖反应（PPGR）**。

模型基于 [CGMacros](https://physionet.org/content/cgmacros/1.0.0/)（45 名受试者，1,557 餐）训练，采用**留一被试交叉验证（LOPO）**——每一次预测的对象都是模型从未见过的人。

**在线演示：** <https://metanutri-ai-ppgr-predictor.streamlit.app/> ·
**父平台（全栈项目）：** <https://meta-nutri-ai.vercel.app/> ·
源码 [`MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-)

**姊妹演示：** [`MetaNutri-Ai-RAG`](https://github.com/ElijahZhao/MetaNutri-Ai-RAG)——独立的 RAG 问答演示，基于《中国食物成分表》与《中国居民膳食指南（2022）》，自带检索管线与 50 条内置评测：<https://metanutri-ai-rag-question-answering.streamlit.app>。

## 这个仓库是什么（以及不是什么）

本仓库是两仓库项目的**演示半区**，同时是三仓库项目中的一员。父仓库
[`MetaNutri---AI-`](https://github.com/ElijahZhao/MetaNutri---AI-) 持有完整项目：全栈平台 *和* 用于训练与评估下列模型的 `research/` 模块。本仓库只存放部署演示所需的内容。姊妹仓库 [`MetaNutri-Ai-RAG`](https://github.com/ElijahZhao/MetaNutri-Ai-RAG) 是**独立**的 RAG 展示：它不从父仓库导出——与父仓库共享的只有 MetaNutri 品牌、相同的基础营养数据源与 MIT 许可家族。

- **它是** Streamlit Community Cloud 的部署目标——自包含（根目录 `app.py` + `requirements.txt`，约 2.3 MB），模型导出为 **JSON**，运行时无需训练环境。
- **它不是**父仓库的 submodule、fork 或镜像，并且**从不独立编辑**——由父仓库重新生成。
- **数据流单向：** `MetaNutri---AI-` → 本仓库。没有回流。
- **运行时互不调用：** 无 API 调用、无共享包、无数据交换。应用加载导出的提升器并在本地预测。
- **为什么不用 submodule：** Streamlit Community Cloud 构建*仓库根目录*并期望根目录有 `app.py`；submodule 会给构建增加一次 checkout 步骤和一个失败模式，且仍无法替代这个精选子集。

**真值来源。** 本 README 中的每一个数字（结果、外部验证、图表）都由父仓库的 `research/experiments/*.csv` 生成，且必须与其[技术报告](reports/technical_report.pdf)一致。两者不一致时，以父仓库为准并重新生成本仓库。

**报告布局。** `reports/technical_report.md` 及其 PDF 从父模块原样复制，两份可逐字节比对。其 §9 将管线称为 `research/`；在本部署中同样的脚本位于仓库根目录（`bash src/run_all.sh`）。

**同步漂移检查。** CI 运行 [`scripts/check_sync.py`](scripts/check_sync.py)，通过 GitHub API 解析父仓库文件树，比对 32 个策展文件的 git blob SHA。*Sync（父模块）* job 变红意味着父仓库前进了，或本仓库某副本被改动——重新复制该文件；若策展集本身意图变更，则更新脚本。

## 结果（LOPO，Pearson r）

| 目标 | 仅早餐（n=423） | 全部餐次（n=1,557） |
|---|---|---|
| **XGBoost** —— 2-h AUC | **0.890** | **0.838** |
| **XGBoost** —— 2-h iAUC | **0.655** | **0.451** |
| **XGBoost** —— 血糖峰值升高 | **0.632** | **0.542** |
| 仅能量线性 —— iAUC | 0.255 | 0.143 |
| 仅碳水线性 —— iAUC | 0.107 | 0.147 |

参考基线被模型大幅甩开正是要点：仅碳水计数几乎解释不了被试特异反应。仅早餐列复现了 CGMacros 论文自身的基线（AUC *r* ≈ 0.89，iAUC *r* ≈ 0.64）；全部餐次列是本项目对午餐、晚餐与加餐的扩展。

![模型对比](reports/figures/fig2_model_comparison.png)

合并相关系数混淆了两个问题。将其拆分为**被试内**（模型能否正确排序一个人自己的各餐？）与**被试间**（模型能否还原*谁*反应更大？）技能，可见 AUC 的高合并 *r* 主要由被试间分量驱动，而 iAUC 的两个分量几乎相等——使 iAUC 成为更真实的餐级信号。中位切分 ROC-AUC 在各目标上为 0.74–0.96。详见[技术报告](reports/technical_report.pdf) §4.5。

![被试内与被试间技能](reports/figures/fig6_within_between.png)

预测还带**保形区间**。其边际覆盖率与名义值一致（0.800 / 0.898），但区间较宽，且对少数个体覆盖不足——在报告 §4.6 中如实说明。

![保形区间](reports/figures/fig7_conformal.png)

混合效应检查（被试随机截距）证实同一图景：AUC 的 ICC 为 0.69，iAUC 仅 0.33；个体内碳水升高反应、蛋白质降低反应。详见报告 §4.7。

## 外部验证（BIG IDEAs）

将冻结的 CGMacros 模型**不做任何重训练**地应用到 [BIG IDEAs](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/)（16 名受试者，656 餐——不同的 CGM 设备 Dexcom G6 与自由生活饮食日志），仅使用两个数据集共有的 5 个特征。

| 协议（5 个共享特征） | 2-h iAUC *r* | 2-h AUC *r* | 峰值 *r* |
|---|---|---|---|
| CGMacros（LOPO，内部） | 0.49 | 0.82 | 0.51 |
| BIG IDEAs（LOPO，内部） | 0.46 | 0.63 | 0.48 |
| **BIG IDEAs（外部迁移）** | **0.23** | **0.57** | **0.22** |
| BIG IDEAs（外部，仅碳水 Ridge） | 0.36 | 0.29 | 0.36 |

AUC 部分迁移，但 iAUC 与峰值升高崩盘——而仅碳水的 Ridge 回归在这两个目标上迁移得*更好*。BIG IDEAs 内部 LOPO 行说明下降是队列间偏移，而非外部数据噪声。演示在"能否泛化到其他队列？"下暴露此只读面板。

![外部验证](reports/figures/fig5_external_validation.png)

## 工作原理

- **特征** —— 餐次：碳水 / 蛋白质 / 脂肪 / 膳食纤维（g）、餐时血糖。被试：年龄、性别、BMI、HbA1c、空腹血糖、空腹胰岛素、HOMA-IR（衍生）、甘油三酯、总胆固醇、HDL、非 HDL、LDL、VLDL、Chol/HDL。
- **目标** —— 2 小时增量 AUC（基线以上的正增量面积）、2 小时总 AUC、血糖峰值升高。
- **模型** —— XGBoost，每目标一个提升器，`max_depth=1`、`n_estimators=80`、`learning_rate=0.2`。
- **协议** —— 留一被试交叉验证。缺失特征用**训练折**中位数插补、以**训练折**统计量标准化——不留出任何被试的信息泄漏。
- **可解释性** —— 精确 TreeSHAP 值（XGBoost `pred_contribs`）。

## 本地运行演示

```bash
pip install -r requirements.txt
streamlit run app.py
```

应用预测三个指标，绘制*示意性* 2 小时曲线，并显示逐特征 TreeSHAP 贡献。

> 模型预测的是**标量，不是时间序列**。绘制的曲线是缩放至预测峰值升高与 iAUC 的群体平均 gamma 形状，UI 将其标注为重建而非模型输出。

## 测试

```bash
python test_app.py
```

用 Streamlit 自带的 `AppTest` 无头启动 `app.py`，点击 **Predict**，断言三个标量预测与 TreeSHAP 面板无异常渲染——无需服务器与网络。

## 仓库结构

```
app.py                Streamlit 演示
test_app.py           演示的无头 AppTest 冒烟测试
inference.py          模型加载、预测、TreeSHAP、曲线重建
model/                训练好的提升器 + 预处理元数据
assets/               外部验证面板所需的打包结果
requirements.txt      演示依赖（锁定版本）
src/                  父仓库 research/src/ 的溯源副本
reports/              父仓库 research/reports/ 的溯源副本
experiments/          父仓库 research/experiments/ 的溯源副本
data/README.md        数据溯源与下载说明
```

`src/`、`reports/`、`experiments/` 三个目录（以及 `data/README.md`）是父仓库 [`research/`](https://github.com/ElijahZhao/MetaNutri---AI-/tree/main/research) 模块的**溯源副本**——保留它们是为了让这里每个数字都能追溯到产生它的代码。重训练**始终在父仓库**的 `research/` 目录进行；以下命令按该布局假设：

```bash
bash src/run_all.sh --skip-download   # 数据 → 结果 → 图表 → PDF，一条命令
```

或逐步执行：

```bash
python src/download_data.py      # 下载 CGMacros（约 627 MB）
python src/build_dataset.py      # 构建餐级数据集
python src/experiment.py         # 运行 LOPO 套件
python src/train_final.py        # 导出可部署模型
```

## 数据与许可

**许可分层——详见 [NOTICE.md](NOTICE.md)。** 代码（`app.py`、`inference.py`、`src/`、`test_app.py`）为 **MIT**。由数据集计算得到的一切（`model/*.json`、`experiments/*.csv`、`assets/*`、`reports/*`）为 **CC BY-NC-SA 4.0**。仓库级 [LICENSE](LICENSE) 文件是 MIT 文本，仅覆盖代码层。

[CGMacros](https://physionet.org/content/cgmacros/1.0.0/) —— PhysioNet，DOI [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658)，许可 **CC BY-NC-SA 4.0**。原始数据**不**在本仓库再分发；`src/download_data.py` 负责获取。由其衍生的模型以相同许可分发，并标注原始作者。

外部验证使用 [BIG IDEAs](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) —— PhysioNet，DOI [10.13026/zthx-5212](https://doi.org/10.13026/zthx-5212)，许可 **ODC-By 1.0**（署名）。

## 局限

- 队列小（45 名受试者）且人口学范围窄；未见被试的留出性能远低于被试内性能。
- iAUC 比 AUC 难预测得多——AUC 由个体整体血糖水平主导，而 iAUC 剥离出餐驱动的波动。
- **外部验证温和：** AUC 跨队列迁移（*r* ≈ 0.57），但 iAUC 与峰值升高不迁移（*r* ≈ 0.22），仅碳水基线在这两个目标上反而迁移更好。
- 示意曲线只是视觉辅助，**不是**对反应形状的已验证预测。
- **非医疗器械。** 仅限研究与作品集用途。

**开发说明。** 本项目及其父仓库实现过程中使用了 AI 编码工具作为辅助——主要用于脚手架搭建、重构与文档撰写。研究设计、实验、评估与最终审查为作者本人完成，AI 生成内容在合入前均经过人工审核与调整，而非原样采用。
