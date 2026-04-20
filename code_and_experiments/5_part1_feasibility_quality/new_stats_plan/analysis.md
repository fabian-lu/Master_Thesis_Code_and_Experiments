# Statistical Analysis Plan — Part 1: Feasibility & Quality

This is the refined analysis plan for the thesis version of Study 1. It addresses the independence violation in the original paper (by including instance as a fixed blocking factor, the systematic between-instance variance is absorbed into the model, so the residuals satisfy the independence assumption that raw observations violated), aligns with the thesis's 3-dimension evaluation framework, and grounds all methodological decisions in Montgomery (2012), *Design and Analysis of Experiments*, 8th edition (Wiley).

---

## Table of Contents

1. [Design Summary](#1-design-summary)
2. [Data Sources](#2-data-sources)
3. [Thesis vs Paper 1 Differences](#3-thesis-vs-paper-1-differences)
4. [Part A: Feasibility — Structural Compliance (descriptive)](#4-part-a-feasibility--structural-compliance-descriptive)
5. [Part B: Feasibility (Accuracy) + Quality (Lay Rel., Helpfulness)](#5-part-b-feasibility-accuracy--quality-lay-rel-helpfulness)
6. [Methodological Decisions and Rationales](#6-methodological-decisions-and-rationales)
7. [Analysis Protocol Per RQ](#7-analysis-protocol-per-rq)
8. [Formulas and Effect Sizes](#8-formulas-and-effect-sizes)
9. [Multiple Comparisons](#9-multiple-comparisons)
10. [Robustness Checks](#10-robustness-checks)
11. [Implementation Checklist](#11-implementation-checklist)

---

## 1. Design Summary

| Factor | Levels | N per level |
|--------|--------|-------------|
| ML Model | XGBoost, Random Forest, MLP, SARIMAX | 198 / 198 / 198 / 66 |
| XAI Method | SHAP, LIME, None | 198 / 198 / 264 |
| LLM (generator) | GPT-4o, Llama-3-8B, DeepSeek-R1 | 240 / 240 / 180 |
| Prompting Strategy | 8 strategies | ~82 each |
| Test Instance (block) | 3 (early, middle, late) | 220 each |
| Judge | GPT-4o, DeepSeek-R1 | 660 each |

**Total corpus:** 660 NLEs, each scored by 2 judges on 3 dimensions = 3,960 individual scores.

**Statistical unit:** The NLE (N=660). Scores are **averaged across the two judges** before analysis (see §6.2 for rationale).

**Design type:** Incomplete factorial with fixed blocking. Not all factor combinations exist (see §6.4).

---

## 2. Data Sources

### Generation data (NLE text, metadata)
- `~/Desktop/First_Paper_XAI_LLM/Data/narratives/*.csv` — one file per strategy
- Key columns: `LLM`, `Model`, `XAI`, `Strategy`, `WeekEndDate`, `Explanation`, `TokensAnswer`

### Evaluation data (G-Eval scores)
- `~/Desktop/First_Paper_XAI_LLM/Data/forsetzung_results/20250813_135743/geval_gpt4.csv`
- `~/Desktop/First_Paper_XAI_LLM/Data/forsetzung_results/20250813_135743/geval_deepseek.csv`
- Score columns used (thesis uses 3 of Paper 1's 5):
  - `eval_accuracy_g_eval_score` → **Accuracy** (maps to Feasibility construct)
  - `eval_lay_user_relevancy_g_eval_score` → **Lay Relevancy** (maps to Quality construct)
  - `eval_usefulness_explanation_helpfulness_g_eval_score` → **Helpfulness** (maps to Quality construct)
- Dropped from thesis: `eval_expert_relevancy_g_eval_score`, `eval_usefulness_prediction_closeness_g_eval_score`

### Instance identifier
- `WeekEndDate` column: 3 unique values (2009-11-09, 2010-04-26, 2010-07-26)

---

## 3. Thesis vs Paper 1 Differences

| Aspect | Paper 1 (published) | Thesis (this analysis) |
|--------|---------------------|------------------------|
| Dimensions | 5 | **3** (Accuracy, Lay Rel., Helpfulness) |
| Instance handling | Not accounted for (independence violation) | **Fixed blocking factor** in all models |
| Judge handling | Averaged, no control | **Averaged** (primary), judge-as-fixed (robustness) |
| Primary analysis | One-way ANOVA per factor | **RQ-specific factorial model** with all relevant factors |
| Effect sizes | ω² from one-way ANOVA | **Partial η²** from blocked factorial model |
| Pairwise comparisons | Independent t-tests with Cohen's d | **Contrasts from fitted model** with blocked Cohen's d |

---

## 4. Part A: Feasibility — Structural Compliance (descriptive)

Structural compliance is purely descriptive. No hypothesis tests — just counts and distributions.

### 4.1 Bullet Point Compliance

The constraint is **≤ 6 bullet points** per NLE.

- **Extraction:** Parse `Explanation` text, count bullet points (lines starting with `- `, `• `, `* `, or numbered `1. `, etc.)
- **Compliance rate:** proportion of NLEs with ≤ 6 bullet points
- **Violations:** list any NLEs exceeding 6 bullet points, broken down by all factors (strategy, LLM, model, XAI)
- **Distribution:** report mean, median, SD, min, max bullet count
- **Breakdown by factor:** report per strategy, LLM, model, XAI
- **Visualisation:** histogram or boxplot of bullet point counts by strategy

### 4.2 Word Count Compliance

The constraint is **≤ 200 words** per NLE.

- **Extraction:** word-tokenise `Explanation` text (split on whitespace), count words
- **Compliance rate:** proportion of NLEs with ≤ 200 words
- **Violations:** list any NLEs exceeding 200 words, broken down by all factors (strategy, LLM, model, XAI)
- **Distribution:** report mean, median, SD, min, max word count
- **Mean word count by factor:** break down by strategy, LLM, model, XAI
- **Visualisation:** histogram or boxplot of word counts by strategy

### 4.3 Structural Compliance Summary

Report:
- Bullet point compliance: X/660 (≤ 6 bullets), mean = X, SD = X
- Word count compliance: X/660 (≤ 200 words), mean = X, SD = X

---

## 5. Part B: Feasibility (Accuracy) + Quality (Lay Rel., Helpfulness)

### Dimensions

| Dimension | Construct | Definition |
|-----------|-----------|------------|
| Accuracy | Feasibility | Factual correctness of values, features, metrics |
| Lay Relevancy | Quality | Usefulness for a non-technical user |
| Helpfulness | Quality | Cues for judging prediction accuracy |

All three dimensions receive the **same statistical treatment**. Accuracy results are reported under Feasibility; Lay Relevancy and Helpfulness under Quality.

### Research Questions

| RQ | Question | Factor | Subset | N |
|----|----------|--------|--------|---|
| RQ-F1 | Does factual accuracy vary by factor? | all 4 | full data | 660 |
| RQ-Q1 | Does XAI method affect quality? | XAI | SARIMAX excluded | 594 |
| RQ-Q2 | Does LLM choice affect quality? | LLM | full data | 660 |
| RQ-Q3 | Does ML model quality affect quality? | Model | SARIMAX excluded | 594 |
| RQ-Q4 | Does inherent interpretability help? | Model | XAI=None only | 264 |
| RQ-Q5 | Which prompting strategy is best? | Strategy | full data | 660 |

---

## 6. Methodological Decisions and Rationales

This section documents every key statistical decision and its justification. Each can be cited directly in the thesis.

### 6.1 Instance as a Fixed Blocking Factor

**Decision:** Include instance as a **fixed effect** (blocking factor) in all models.

**The problem it solves:** The same 3 test instances appear across all conditions. Every group being compared contains NLEs from all 3 instances. If some instances are systematically easier or harder to explain, this creates within-instance correlation that violates the independence assumption of standard ANOVA. Failing to block inflates the error term — Montgomery (2012, §4.1, p. 146) demonstrates this directly: in his Example 4.1, the mean square error more than doubles (7.33 → 15.11) when blocking is omitted, potentially hiding real treatment effects.

**Why fixed, not random:** Montgomery (2012, §3.9, p. 116) defines a random factor as one where "the experimenter randomly selects *a* of these levels from the population of factor levels." Our 3 instances were **deliberately selected** using a tercile-based criterion (requiring temporal diversity and a specific model error ranking), not randomly sampled from the 60-instance test set. Since the levels were purposively chosen, the factor is fixed by definition. Inference is conditional on these specific instances, not generalised to a population of instances.

**Why not random despite serving as blocks:** Montgomery (2012, §5.6, p. 221) often presents blocks as random in industrial experiments (e.g., operators, batches) because those blocks are typically sampled from a larger population. But "blocks are random" is a modeling assumption for that context, not a definitional requirement. When blocks are deliberately selected, they are fixed — the blocking mechanism (partitioning variance) works the same way regardless.

**The small number (3) reinforces the decision:** With only 3 levels, variance component estimation for a random effect would be extremely unreliable (effectively estimating a variance from 2 degrees of freedom). However, the number of levels is not the primary reason — the selection mechanism is.

**Inference scope limitation:** Because instances are fixed, conclusions are conditional on the selected instances. This must be stated explicitly:

> "Test instances were included as a fixed blocking factor because they were purposively rather than randomly selected. Consequently, inference is conditional on these selected instances rather than generalised to a random population of future instances."

**Statistical model (single-factor RCBD):** Following Montgomery (2012, §4.1, Eq. 4.1, p. 141):

```
y_ij = μ + τ_i + β_j + ε_ij
```

where:
- `μ` = overall mean
- `τ_i` = effect of the i-th treatment level (fixed)
- `β_j` = effect of the j-th block / instance (fixed)
- `ε_ij` ~ NID(0, σ²) = random error

The F-test for the treatment effect uses F₀ = MS_Treatments / MS_E, which is valid when both treatments and blocks are fixed (Montgomery, 2012, §4.1.1, p. 143).

### 6.2 Judge Handling: Averaging (Primary)

**Decision:** Average scores across the two judges (GPT-4o, DeepSeek-R1) before analysis, yielding N=660 judge-averaged scores per dimension.

**Rationale — experimental vs observational units:** The **experimental unit** is the NLE — it is the entity to which treatments (LLM, XAI, Model, Strategy) are applied. The two judges are **observational replicates** (subsamples) that each measure the same NLE. They are not independent experimental units.

If judge is included as a fixed effect without accounting for the repeated-measures structure (i.e., treating all 1,320 rows as independent), the model's residual error term reflects only **within-NLE judge disagreement**, not the actual **between-NLE variance** due to treatment effects. This is **pseudoreplication** — the effective sample size is artificially doubled, F-statistics are inflated, and Type I error rates increase.

**Montgomery's principle:** Montgomery (2012, §13.1) distinguishes experimental units from measurement units. When multiple measurements are taken on the same experimental unit, they should be averaged (or modeled with a proper nested/repeated-measures structure) to avoid inflating error degrees of freedom.

**Why not include judge as a fixed effect (primary)?** It would require either:
- A repeated-measures model with NLE as a random intercept (but NLE has 660 levels — and it IS the unit, not a grouping factor), or
- Explicit nesting of judge within NLE

Both add complexity without clear benefit for our RQs, which are about generation factors, not judge behaviour.

**Robustness check:** Run all primary models separately for each judge. If both judges show the same direction and significance, the finding is robust. Also report inter-judge agreement (Pearson/Spearman correlation, mean absolute difference per dimension).

### 6.3 Factorial Model as Primary Analysis (Not One-Way ANOVA)

**Decision:** The primary analysis for each RQ uses an **RQ-specific factorial model** that includes the focal factor, all other relevant design factors, and the instance block — not a one-way ANOVA on just the focal factor.

**Rationale:** Montgomery (2012, Ch. 5, §5.1-5.3) and Collins et al. (2014) both argue that the core advantage of factorial experiments is estimating main effects **while averaging over (collapsing across) all other factors simultaneously**. A one-way ANOVA per factor ignores this structure and risks confounding the focal factor's effect with imbalances in other factors.

Concretely: in a one-way ANOVA for LLM effect (`score ~ C(LLM) + C(instance)`), the LLM groups may differ in their composition of strategies or models due to the incomplete crossing. The apparent LLM effect could partly reflect strategy composition. The factorial model conditions on all other factors, isolating the **unique contribution** of each.

Collins et al. (2014, p. 499) state: "The main effect is the difference between the mean response at one level of a particular factor and the mean response at the other level, **collapsing over the levels of all remaining factors**."

**RQ-specific model template:**

```
score ~ C(focal_factor) + C(other_factor_1) + C(other_factor_2) + ... + C(instance)
```

using Type II sums of squares (see §6.5).

The original one-way ANOVAs from Paper 1 (without blocking or factorial adjustment) may be included in the appendix as a comparison to show how conclusions are affected by the methodological correction.

### 6.4 Handling the Incomplete Factorial

**The problem:** The design is not fully crossed:
- SARIMAX only has XAI=None (no SHAP, no LIME)
- DeepSeek-R1 has no explicit CoT strategies (CoT zero-shot, CoT few-shot excluded)

This means some factor combinations have **structural zeros** (empty cells). A naïve full factorial ANOVA assumes all cells are populated.

**Solution:** Use **RQ-specific subsets** that yield clean, estimable comparisons:

| RQ | Subset | Rationale |
|----|--------|-----------|
| RQ-Q1 (XAI) | Exclude SARIMAX (N=594) | SARIMAX only has XAI=None; including it confounds Model and XAI |
| RQ-Q2 (LLM) | Full data (N=660) | All LLMs present, but note DeepSeek missing CoT — interpret cautiously |
| RQ-Q3 (Model) | Exclude SARIMAX (N=594) | Compares 3 ML models (XGB, RF, MLP) across all XAI/LLM/Strategy combinations |
| RQ-Q4 (Interpretability) | XAI=None only (N=264) | All 4 models including SARIMAX at same XAI condition — fair comparison |
| RQ-Q5 (Strategy) | Full data (N=660) | Note DeepSeek missing 2 strategies — interpret cautiously |

Within each subset, the model includes all factors that **vary in that subset**. This follows the general linear model approach for incomplete designs: fit the model on the observed design matrix and base inference on estimable contrasts.

**Thesis language:**

> "Because the design contained structural exclusions (SARIMAX observed only at XAI=None; DeepSeek-R1 not crossed with explicit CoT strategies), analyses were conducted using linear models on the observed design matrix rather than balanced factorial ANOVA. Hypothesis tests and pairwise comparisons were based on estimable contrasts from the fitted models."

### 6.5 Type II Sums of Squares

**Decision:** Use **Type II SS** for all factorial models.

**Rationale:** In unbalanced/incomplete designs, the three types of SS (I, II, III) give different results:
- **Type I (sequential):** Order-dependent; inappropriate unless a specific ordering has scientific meaning.
- **Type II:** Tests each factor's main effect **assuming no interactions** involving that factor. Appropriate when the primary interest is main effects and interactions are either absent or tested separately.
- **Type III:** Tests hypotheses that may involve averaging over empty cells — can produce uninterpretable results in incomplete designs.

For our design, Type II SS is the safest choice: we are primarily interested in main effects, and we test interactions separately (§7, Step 4). Type III SS would require assumptions about missing cells (SARIMAX × SHAP, etc.) that don't exist.

**Reference:** Montgomery (2012, §5.3, p. 200-202) discusses SS types for unbalanced data. Langsrud (2003, "ANOVA for unbalanced data: Use Type II instead of Type III sums of squares", *Statistics and Computing*) argues Type II is generally preferable for testing main effects.

### 6.6 Scores as Approximately Continuous

**Decision:** Treat the 1-5 G-Eval scores as approximately continuous and analyse with linear models (ANOVA).

**Rationale:** The scores are not strictly continuous integers (1-5), but:
- GPT-4o produces **probability-weighted continuous scores** via G-Eval (e.g., 3.7, 4.2) because the API exposes token log-probabilities
- DeepSeek-R1 produces integer scores (1-5) at τ=0.0
- After averaging across the two judges, scores take fractional values (e.g., 3.5, 4.0, 4.5), creating a quasi-continuous scale
- ANOVA is robust to moderate violations of normality, especially with large N (Montgomery, 2012, §3.4, p. 80)

**Robustness check:** Kruskal-Wallis + Dunn's post-hoc as a nonparametric sensitivity check for dimensions with strong ceiling effects.

---

## 7. Analysis Protocol Per RQ

The same protocol applies to every RQ. Run it for all 3 DVs (Accuracy, Lay Relevancy, Helpfulness) separately.

### Step 1: Descriptive statistics

- Mean ± SD per factor level, per dimension (using judge-averaged scores)
- Tables use judge-averaged scores for readability
- Check for ceiling/floor effects (proportion of scores at 5.0 or 1.0)

### Step 2: Interaction screening (FIRST)

**Test interactions before interpreting main effects.** If a significant interaction involving the focal factor exists, the main effect is scientifically misleading as a global summary (Montgomery, 2012, §5.3.6, p. 202).

For each RQ, fit the model WITH relevant two-way interactions between the focal factor and other design factors:

```python
# Example: RQ-Q2 (LLM), testing LLM interactions
model_int = ols('score ~ C(llm)*C(strategy) + C(llm)*C(model) + C(llm)*C(xai) + C(instance)',
                data=df).fit()
anova_int = sm.stats.anova_lm(model_int, typ=2)
# Check interaction terms for significance
```

**Decision rule:**
- If **no** interactions involving the focal factor are significant → proceed to Step 3 (main effects)
- If **any** interaction is significant → skip the omnibus main effect, proceed directly to **simple effects analysis** (effect of focal factor at each level of the interacting factor)

### Step 3: Main effects model

If interactions are non-significant, fit the additive factorial model:

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('score ~ C(llm) + C(model) + C(xai) + C(strategy) + C(instance)',
            data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # Type II SS
```

- Report F-statistic and p-value for the **focal factor only** (other factors are controls)
- Compute partial η²_p for the focal factor (see §8)

### Step 4: Pairwise comparisons (if omnibus significant)

Extract pairwise contrasts **from the same fitted model** using **estimated marginal means (EMMs)**, not raw means. EMMs adjust for imbalances in the incomplete design.

- Apply BH FDR correction across all pairwise comparisons within each dimension
- Compute d_RMSE using √(MS_E) and EMMs from the factorial model (see §8)
- Also report unstandardised EMM differences with 95% CIs

### Step 5: Model diagnostics

For every fitted model, verify assumptions (Montgomery, 2012, §3.4-3.5):

- **Residuals vs fitted plot:** check for homoscedasticity (constant variance)
- **Normal Q-Q plot:** check for approximate normality of residuals
- **Leverage/influence diagnostics:** check for influential observations (Cook's distance)
- If heteroscedasticity is detected: report HC3 robust standard errors as sensitivity check

### Step 6: Two-way interaction tests (all 6 pairs)

**Why two-way interactions only?**

Montgomery (2012, §6.5) describes the **sparsity of effects principle** (also called the hierarchical ordering principle): in most factorial experiments, the system is dominated by main effects and low-order (two-way) interactions. Three-way and higher interactions are rarely significant, difficult to interpret, and in our incomplete design often **inestimable** (e.g., a Model × XAI × LLM interaction cannot be computed when the SARIMAX × SHAP cell is empty).

**All six two-way interactions are tested:**

| # | Interaction | Estimable? | Notes |
|---|------------|------------|-------|
| 1 | Model × XAI | Partially | SARIMAX only has XAI=None → confounded. Test on SARIMAX-excluded subset (N=594). |
| 2 | Model × LLM | Yes | All model–LLM combinations exist. |
| 3 | Model × Strategy | Yes | All combinations exist. |
| 4 | XAI × LLM | Partially | SARIMAX exclusion applies; test on N=594 subset. |
| 5 | XAI × Strategy | Partially | Same SARIMAX constraint. Test on N=594 subset. |
| 6 | LLM × Strategy | Partially | DeepSeek-R1 missing 2 CoT strategies → some cells empty. Test on full data but note incompleteness. |

Three-way and higher interactions are not tested: per the sparsity of effects principle (Montgomery, 2012, §6.5), they are rarely significant, difficult to interpret, and in our incomplete design often inestimable due to empty cells.

**Model for each interaction test:**

```python
# Example: testing XAI × LLM interaction (on SARIMAX-excluded subset)
model = ols('score ~ C(xai) * C(llm) + C(model) + C(strategy) + C(instance)',
            data=df_no_sarimax).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
# Report the interaction term's F and p from the Type II ANOVA table
```

If a significant interaction is found, the corresponding main effect must be interpreted **conditionally** -- e.g., if XAI × LLM is significant, the effect of XAI differs depending on the LLM, so reporting a single XAI main effect is an oversimplification (Montgomery, 2012, §5.3.6, p. 202). Follow up with **simple effects analysis**: test the effect of factor1 at each level of factor2 separately.

---

## 8. Formulas and Effect Sizes

### 8.1 The Blocked Factorial ANOVA Model

For a single focal factor A with `a` levels, `p` other design factors, and instance as a fixed block with `b=3` levels, the general linear model is:

```
y = μ + α_i + (other factor effects) + β_j + ε
```

The ANOVA partitions total variance:

```
SS_Total = SS_A + SS_other_factors + SS_Instance + SS_Error
```

The F-test for factor A:

```
F_A = MS_A / MS_E
```

with df_A = a - 1 and df_E = N - (total model df) - 1.

### 8.2 Effect Size: Partial Eta-Squared (η²_p)

**Use partial η² as the omnibus effect size.** It is the standard effect size for factorial ANOVA, universally understood, and straightforward to compute from the ANOVA table. It estimates the proportion of variance accounted for by factor A after removing variance due to other factors.

**Formula** (Cohen, 1973; Richardson, 2011):

```
η²_p = SS_A / (SS_A + SS_E)
```

where:
- `SS_A` = sum of squares for the focal factor
- `SS_E` = residual (error) sum of squares from the full blocked model

**Benchmarks** (Cohen, 1988):
- < .01 = negligible
- .01 -- .06 = small
- .06 -- .14 = medium
- > .14 = large

**Note:** η²_p is a biased estimator (it slightly overestimates the population effect). For transparency, partial omega-squared (ω²_p) can be reported alongside as a less-biased alternative. However, the council of statistical reviewers flagged that ω² formulas vary across sources and designs, making η²_p the safer primary choice for thesis defensibility.

### 8.3 Model-Based Standardised Mean Difference (d_RMSE)

**Use the root mean square error (√MS_E) from the blocked factorial model as the pooled standard deviation, with estimated marginal means (EMMs) as the numerator.** Because MS_E has already had instance and other factor variance partialled out, this effect size controls for blocking and other design factors.

**This is not classical raw-score Cohen's d.** It is a model-based standardised contrast (sometimes called partial d or d_RMSE). Values will be larger than raw Cohen's d because the denominator excludes systematic factor variance. This is appropriate for multifactor designs where the raw SD conflates treatment effects with residual noise (Olejnik & Algina, 2000).

**Formula:**

```
d_RMSE = (EMM_1 - EMM_2) / √(MS_E)
```

where:
- `EMM_1`, `EMM_2` = estimated marginal means for the two levels being compared (adjusted for other factors and blocking)
- `MS_E` = residual mean square from the blocked factorial model

**Also report the unstandardised EMM difference and its 95% CI** — for a bounded 1-5 scale, unstandardised differences are often more interpretable than standardised ones.

**Benchmarks** (Cohen, 1988, applied cautiously to partial d):
- < 0.2 = negligible
- 0.2 -- 0.5 = small
- 0.5 -- 0.8 = medium
- > 0.8 = large

### 8.4 The Standard ANOVA F-Test

For completeness, the F-statistic for testing the null hypothesis H₀: all treatment means are equal:

```
F₀ = MS_Treatments / MS_E
```

which follows an F-distribution with (a-1) and (N - total_model_df - 1) degrees of freedom under H₀ (Montgomery, 2012, §4.1.1, p. 143).

---

## 9. Multiple Comparisons

### Multiplicity families (precisely defined)

Three distinct families per RQ:

1. **Omnibus tests across dimensions:** BH FDR applied across the 3 dimension-level F-tests for the focal factor within each RQ (3 tests per family).

2. **Pairwise comparisons within each dimension:** BH FDR applied across all pairwise contrasts for the focal factor within each dimension separately. E.g., for LLM (3 levels) on Accuracy: 3 pairwise comparisons → 1 family.

3. **Interaction tests:** BH FDR applied across the 6 two-way interaction tests within each RQ (if all 6 are run on the same data subset).

### Across RQs

No correction. Each RQ tests a distinct, pre-specified hypothesis. This is standard in multi-experiment/multi-RQ studies.

### Reporting

Report both raw p-values and FDR-corrected p-values (p_FDR). Significance is judged at α = .05 on the corrected values.

**Thesis language:**

> "False discovery rate control (Benjamini & Hochberg, 1995) was applied within each research question family of tests. Three families were defined per research question: (a) omnibus tests across the three quality dimensions, (b) pairwise comparisons within each dimension, and (c) interaction tests. No adjustment was made across distinct pre-specified research questions."

---

## 10. Robustness Checks

### 10.1 Judge-Specific Analyses

Run every primary model separately for GPT-4o judge scores and DeepSeek-R1 judge scores (instead of averaging). If both judges show the same direction and significance, the finding is robust.

Also report inter-judge agreement: Pearson/Spearman correlation and mean absolute difference per dimension.

### 10.2 Judge-Level Mixed Model

As a secondary analysis, fit a mixed model on the unaggregated judge-level data (N=1,320) that properly accounts for the nested structure (two judge scores per NLE):

```python
# Using statsmodels or R's lme4:
# score ~ C(focal) + C(other_factors) + C(instance) + C(judge) + (1|NLE)
```

This includes judge as a fixed effect while using a random intercept for NLE to avoid pseudoreplication. It also allows testing judge × focal factor interactions to check whether judges differ in their assessment of the treatment effect.

### 10.3 Without-Instance-Blocking Comparison

Run each analysis BOTH with and without the instance blocking factor. Report whether conclusions change. This directly validates (or invalidates) the original Paper 1 results and demonstrates the impact of the methodological correction.

### 10.4 Heteroskedasticity-Robust Standard Errors

For the primary linear models, compute HC3 robust standard errors as a sensitivity check against variance heterogeneity:

```python
model.get_robustcov_results(cov_type='HC3')
```

### 10.5 Non-Parametric Alternatives

For dimensions with strong ceiling effects (many scores at 5.0): Kruskal-Wallis test + Dunn's post-hoc. These are rough sensitivity checks (they don't account for blocking), not direct analogues of the primary analysis.

### 10.6 Welch's ANOVA + Games-Howell

For one-way comparisons where Levene's test indicates variance heterogeneity. These don't support blocking natively, so they serve as supplementary checks only.

---

## 11. Implementation Checklist

### Part A: Structural Compliance (descriptive)

- [ ] Load all narrative CSVs, concatenate into one dataframe
- [ ] Extract `instance` from `WeekEndDate`
- [ ] **Bullet point count:** parse `Explanation`, count bullet markers, report compliance rate and distribution
- [ ] **Word count:** tokenise `Explanation`, count words, report compliance rate and distribution
- [ ] **Distributions by factor:** boxplots/histograms for word count and bullet count, broken down by all factors
- [ ] Write structural compliance summary

### Part B: Accuracy + Quality — Per RQ

For each RQ, for each DV (Accuracy, Lay Rel., Helpfulness):

- [ ] **Average scores** across the two judges
- [ ] **Descriptive statistics** (mean ± SD per level, judge-averaged)
- [ ] **Step 2: Interaction screening** — test focal-factor interactions FIRST
- [ ] If interaction significant → simple effects analysis, skip main effect
- [ ] If no interaction → **Step 3: Main effects model**: `score ~ C(focal) + C(other_factors) + C(instance)`, Type II SS
- [ ] **Partial η²_p** from the factorial model
- [ ] **Step 4: Pairwise comparisons** via EMMs from the fitted model
- [ ] **d_RMSE** using √(MS_E) and EMMs; also report unstandardised EMM differences + 95% CI
- [ ] **BH FDR correction** within each multiplicity family (omnibus, pairwise, interactions)
- [ ] **Step 5: Model diagnostics** — residual plots, QQ, leverage
- [ ] **Step 6: All 6 two-way interactions** (on appropriate subsets per estimability table)

### Robustness Checks

- [ ] Judge-specific analyses (per judge, not averaged)
- [ ] Judge-level mixed model with (1|NLE) random intercept (secondary)
- [ ] With vs without instance blocking — did conclusions change?
- [ ] HC3 robust standard errors
- [ ] Kruskal-Wallis + Dunn's nonparametric check
- [ ] Inter-judge agreement: correlation, ICC, mean absolute difference per dimension

### Cross-Cutting

- [ ] Comparison table: original Paper 1 (unblocked one-way) vs corrected (blocked factorial) results
- [ ] Ceiling effect check: proportion of scores at 5.0 per dimension
- [ ] Generate all figures and tables for thesis Chapter 5

---

## References

- Montgomery, D. C. (2012). *Design and Analysis of Experiments* (8th ed.). Wiley.
- Collins, L. M., Dziak, J. J., Kugler, K. C., & Trail, J. B. (2014). Factorial experiments: Efficient tools for evaluation of intervention components. *American Journal of Preventive Medicine*, 47(4), 498-504.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Keppel, G., & Wickens, T. D. (2004). *Design and Analysis: A Researcher's Handbook* (4th ed.). Pearson.
- Olejnik, S., & Algina, J. (2003). Generalized eta and omega squared statistics. *Psychological Methods*, 8(4), 434-447.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Langsrud, Ø. (2003). ANOVA for unbalanced data: Use Type II instead of Type III sums of squares. *Statistics and Computing*, 13(2), 163-167.

---

*Last updated: April 2025*
