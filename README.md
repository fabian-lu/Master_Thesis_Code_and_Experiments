# Explaining the Explanations

**Evaluation of the Feasibility, Quality, and Usefulness of LLM-Generated Narratives for XAI Outputs**

M.Sc. Applied Statistics thesis, University of Göttingen (2026).
Author: Fabian Lukassen.
First supervisor: Prof. Dr. Thomas Kneib.
Second supervisor: Prof. Dr. Benjamin Säfken (TU Clausthal).

This repository contains the compiled thesis and all code and data required to reproduce the two empirical studies it reports.

---

## The thesis in one paragraph

Explainable AI (XAI) methods such as SHAP and LIME produce numerical feature attributions that non-experts cannot read directly. Large language models (LLMs) promise a fix: translate those attributions into plain-text narratives (NLEs). This thesis evaluates that pipeline along three dimensions — feasibility, quality, and usefulness. Part 1 runs a blocked factorial experiment (4 ML models × 3 XAI conditions × 3 LLMs × 8 prompting strategies → 660 NLEs) and finds that the pipeline reliably produces high-quality NLEs, with LLM choice driving most of the variance and XAI method mattering little. Part 2 fixes the best pipeline configuration and runs five downstream experiments (forward simulatability, counterfactual reasoning, mental-model transfer, selective reliance, and a placebic control; 2,636 judgments). High-quality NLEs do not improve accuracy on any task; they inflate confidence regardless of correctness; and in the out-of-distribution setting they halve users' ability to detect unreliable predictions. This *Quality–Usefulness Gap* is the central finding.

---

## Repository layout

```
.
├── Thesis_latex/main.pdf               Compiled thesis (the primary artefact)
├── README.md                           This file
│
├── data/                               All datasets and intermediate artefacts
│   ├── raw/                              UCI household power consumption (source data)
│   ├── processed/                        Weekly-aggregated train/test splits
│   ├── serialized_models/                Trained XGBoost, Random Forest, MLP (.joblib)
│   ├── model_predictions/                Per-model predictions on the test set + metrics
│   ├── xai/                              LIME contributions (SHAP recomputed in-notebook)
│   └── nle/                              Generated NLEs (zero-shot + LLM-specific caches)
│
└── code_and_experiments/               Code organised by thesis chapter
    ├── 2_theoretical_background/         Figures for Chapter 2 (SHAP/LIME waterfalls, temperature plot)
    ├── 4_data_preprocessing/             EDA, model training, SHAP-vs-LIME comparison
    ├── 5_part1_feasibility_quality/      Part 1 — feasibility & quality (Chapter 5)
    │   ├── generating_explanations.ipynb     Generate NLEs across the factorial grid
    │   ├── evaluating_nle_quality.ipynb      Score NLEs on the G-Eval quality dimensions
    │   └── statistical_analysis/             Factorial-design analysis (ANOVA, η²p, post-hoc)
    └── 6_part2_usefulness/               Part 2 — usefulness (Chapter 6)
        ├── experiments/                      Five downstream-task notebooks (E1–E5) + utils
        ├── results/                          Raw judgment output from E1–E5
        └── statistical_analysis/             R scripts (GLMM/CLMM, Bayesian, sensitivity)
```

---

## Reproducibility

**Environment.** Python 3.11 (notebooks) and R (statistical analysis for Part 2).

**Data.** `data/raw/household_power_consumption.txt` is the UCI dataset
([Hébrail & Bérard, 2012](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)).
Everything downstream is derived from it by the notebooks in `code_and_experiments/4_data_preprocessing/`.

**LLM access.** The generation and evaluation notebooks call GPT-4o (via Azure OpenAI), DeepSeek-R1 (via OpenRouter), and Llama-3-8B (locally via Ollama). API keys are read from the notebook environment and are not committed to this repository.

**Suggested order.**
1. `code_and_experiments/4_data_preprocessing/` — preprocessing, model training, SHAP/LIME
2. `code_and_experiments/5_part1_feasibility_quality/generating_explanations.ipynb` — produce the 660-NLE corpus
3. `code_and_experiments/5_part1_feasibility_quality/evaluating_nle_quality.ipynb` — G-Eval scoring
4. `code_and_experiments/5_part1_feasibility_quality/statistical_analysis/analysis.ipynb` — Part 1 stats
5. `code_and_experiments/6_part2_usefulness/experiments/` — run E1–E5 to collect judgments
6. `code_and_experiments/6_part2_usefulness/statistical_analysis/*.R` — Part 2 statistical analysis
