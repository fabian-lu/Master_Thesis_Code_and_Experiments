# Statistical Analysis Plan — Part 2: Usefulness

This is the refined analysis plan for the thesis version of Study 2, incorporating two rounds of LLM council review. Key revisions from the original Paper 2 analysis: (1) collider bias in overconfidence analyses replaced with valid ordinal calibration methods, (2) judge × condition interactions always fitted (no step-down model selection), (3) generator estimand clarity, (4) ROPE for positive null evidence, (5) random slopes attempted and confirmed singular. All methodological decisions are grounded in the mixed-effects modeling literature and the existing analysis infrastructure (R, lme4, ordinal, brms).

---

## Table of Contents

1. [Design Summary](#1-design-summary)
2. [Data Sources](#2-data-sources)
3. [Thesis vs Paper 2 Differences](#3-thesis-vs-paper-2-differences)
4. [Methodological Decisions and Rationales](#4-methodological-decisions-and-rationales)
5. [General Analysis Protocol](#5-general-analysis-protocol)
6. [Experiment-Specific Plans](#6-experiment-specific-plans)
7. [Effect Sizes](#7-effect-sizes)
8. [Multiple Comparisons](#8-multiple-comparisons)
9. [Robustness and Sensitivity Analyses](#9-robustness-and-sensitivity-analyses)
10. [Implementation Checklist](#10-implementation-checklist)

---

## 1. Design Summary

**Fixed pipeline (motivated by Part 1):** XGBoost + SHAP TreeExplainer + zero-shot prompting + GPT-4o / DeepSeek-R1 in a 2×2 generator-judge design.

**Test instances:** 60 (full chronological test set; 55 for E4 due to sliding window).

**Total judgments:** 2,636 across 5 experiments.

| Experiment | Task | Conditions | N judgments | Design |
|-----------|------|------------|-------------|--------|
| E1 | Forward simulatability (error bucket classification) | 5 (B, X, T, X+T, E+X+T) | 720 | Within-instance, ablation |
| E2 | Selective reliance (reliable/unreliable classification) | 2×2 (Baseline/OOD × No NLE/NLE) | 720 | Within-instance, 2×2 factorial |
| E3 | Counterfactual simulatability (direction prediction) | 2 (X, E+X) | 360 | Within-instance, paired |
| E4 | Mental model transfer (error bucket, no NLE at test) | 2 (Baseline training, NLE training) | 330 | Within-instance, sliding window |
| E5 | Placebic control (error bucket classification) | 3 (Baseline, Real NLE, Placebo NLE) | 600 | Within-instance, placebo-controlled |

**Outcome variables (all experiments):**
- **Accuracy:** Binary (0/1) — was the judgment correct?
- **Confidence:** Ordinal (1–5 Likert scale)

**Key structural feature:** The same instances appear across all conditions within each experiment. Observations within an instance are NOT independent — if an instance is inherently easy or hard, all observations from it tend to cluster in the same direction. This requires mixed-effects models with instance as a random intercept.

**Rows per instance per experiment (due to the 2×2 generator-judge design):**

| Experiment | No-NLE conditions | NLE conditions | Rows/instance | Total rows |
|-----------|-------------------|----------------|---------------|------------|
| E1 | 4 conds × 2 judges = 8 | 1 cond × 2 gen × 2 judges = 4 | 12 | 720 |
| E2 | 2 conds × 2 judges = 4 | 2 conds × 2 gen × 2 judges = 8 | 12 | 720 |
| E3 | 1 cond × 2 judges = 2 | 1 cond × 2 gen × 2 judges = 4 | 6 | 360 |
| E4 | 1 cond × 2 judges = 2 | 1 cond × 2 gen × 2 judges = 4 | 6 | 330 (55 inst) |
| E5 | 1 cond × 2 judges = 2 | 2 conds × 2 gen × 2 judges = 8 | 10 | 600 |

Even the sparsest case (E3: 6 rows per instance) provides enough within-instance observations for variance component estimation. The critical requirement for a reliable random intercept is the number of *groups* (instances), not the number of observations per group — the rule of thumb is 20+ groups, and all experiments have 55–60.

---

## 2. Data Sources

### Experiment result CSVs
- `results/E1/full_results.csv` through `results/E5/full_results.csv`
- Key columns: `instance_idx`, `condition`, `correct`, `confidence`, `judge`, `generator`, `true_bucket`/`true_direction`/`true_reliability`, `predicted_bucket`/`predicted_direction`/`reliability`

### NLE cache
- `results/nle/nle_cache_gpt.csv`, `results/nle/nle_cache_deepseek.csv`

### Existing analysis code
- `statistical_analysis/00_setup.R` through `07_descriptive.R`
- `statistical_analysis/output/` — plots and summary CSVs
- `statistical_analysis/takeaway.md` — results from existing (pre-revision) analysis

---

## 3. Thesis vs Paper 2 Differences

| Aspect | Paper 2 (current analysis) | Thesis (this revised plan) |
|--------|---------------------------|---------------------------|
| Overconfidence | correctness × condition interaction in CLMM | **Calibration-based** (calibration curves, Somers' D, calibration model); old interaction reported descriptively only |
| Judge handling | Judge as additive fixed effect | **Judge × condition interaction** always fitted; marginal means via emmeans |
| Generator handling | Averaged implicitly; generator effect in sensitivity | **Estimand clarified**: primary averages over generators; generator-equalized sensitivity |
| Null evidence | Bayesian CIs spanning 1 | **ROPE** for positive null evidence (BF dropped — too complex for thesis) |
| Nonparametric | Friedman / Wilcoxon on averaged data | **Dropped** (loses within-instance structure; inappropriate for clustered design) |
| OR benchmarks | Small=1.5, Medium=2.5, Large=4.0 | **Dropped** (arbitrary); report predicted probability differences instead |
| Random slopes | Not tested | **Tested**: all singular across all experiments, confirming intercept-only |
| E4 covariates | Trial order only | **Instance difficulty** covariate added; sliding-window autocorrelation acknowledged |

---

## 4. Methodological Decisions and Rationales

### 4.1 Instance as Random Intercept

**Decision:** Include `instance_idx` as a random intercept in all models.

**Rationale:** The same 60 instances are evaluated under all conditions. Observations from the same instance are correlated (some instances are inherently easier/harder). The random intercept absorbs this between-instance variance, producing valid standard errors and avoiding pseudoreplication.

**Why random, not fixed (unlike Part 1):** In Part 1, 3 instances were deliberately selected (fixed by Montgomery's selection mechanism criterion). In Part 2, all 60 test instances are used — they constitute the full available test set from a chronological split, and inference should generalise to future instances from the same data-generating process. With 60 levels, variance component estimation is reliable.

### 4.2 Judge as Fixed Effect with Condition Interaction

**Decision:** Include `judge` as a fixed effect AND test `judge × condition` interactions in all primary models.

**Rationale (additive component):** Only 2 judge levels (GPT-4o, DeepSeek-R1) — too few for random effect estimation. Judges show systematic accuracy differences (e.g., E1: DeepSeek substantially more accurate). Including judge as a fixed effect controls for this.

**Rationale (interaction):** The council flagged that condition effects may differ by judge (effect modification). If GPT-4o responds differently to NLEs than DeepSeek-R1, the pooled condition effect averages over heterogeneous effects and could be misleading. The judge × condition interaction tests this directly.

**Protocol:**
1. Always fit the full interaction model: `correct ~ condition * judge + (1 | instance_idx)`
2. Extract the **marginal condition effect** averaged across judges via `emmeans(m, ~ condition)` — this is the primary estimand
3. Also report **judge-specific condition effects** via `emmeans(m, ~ condition | judge)`
4. Report the interaction estimate with CI for transparency (does the condition effect differ by judge?)
5. Do NOT drop the interaction based on a significance test — this avoids pretest model selection bias

**Why not step-down model selection:** Using an LRT p-value to decide whether to keep the interaction biases subsequent inference (the "test then select" problem). With only 2 judges, the interaction test is underpowered anyway, so "non-significant" does not mean "absent." Always fitting the interaction model and extracting marginal means is cleaner and avoids this issue.

**What was done before:** Only `condition + judge` (additive). The thesis revision always includes the interaction.

### 4.3 Generator Handling and Estimand Clarity

**The structural problem:** Generator (GPT-4o vs DeepSeek-R1 as NLE author) only exists in NLE conditions. No-NLE conditions have no generator. This means generator cannot be included as a crossed factor in the full model.

**Primary estimand:** The average treatment effect of NLE presence, marginalised over generators. This is the scientifically relevant question: "Does having an NLE help, regardless of which model wrote it?" The primary models include `condition + judge + (1 | instance_idx)` where condition absorbs the NLE presence manipulation. NLE conditions contribute 4 rows per instance (2 generators × 2 judges) while no-NLE conditions contribute 2 rows (2 judges only). The GLMM handles this imbalance correctly — the random intercept links observations from the same instance.

**Generator-equalized sensitivity:** To verify that the imbalance in rows per condition doesn't drive results, fit a sensitivity model on a generator-equalized dataset: for NLE conditions, randomly select one generator per instance (halving NLE rows to 2 per instance, matching no-NLE conditions). Run the primary model on this balanced dataset and confirm that conclusions are unchanged.

**Generator effect test (NLE conditions only):** Subset to NLE conditions and test `correct ~ condition + generator + judge + (1 | instance_idx)` to verify that NLE source doesn't matter. Also test `generator × judge` interaction (same-family bias).

### 4.4 Overconfidence Analysis: Avoiding Collider Bias

**The problem with the original approach:** The previous analysis tested `confidence ~ correctness × condition + judge + (1 | instance_idx)`. This conditions on `correctness`, which is a **post-treatment variable** — it is determined after the experimental manipulation (condition). Conditioning on a post-treatment outcome creates collider bias: the apparent interaction between correctness and condition may reflect selection effects rather than causal mechanisms. For example, if NLEs change *which* instances are answered correctly (even without changing overall accuracy), the subgroup "incorrect under NLE" differs from "incorrect without NLE" in ways that distort the confidence comparison.

**Revised approach — valid ordinal calibration methods:**

1. **Confidence main effect model (primary):** `confidence ~ condition * judge + (1 | instance_idx)` tests whether NLEs shift the overall confidence distribution, without conditioning on correctness. This remains the primary confidence analysis. Marginal condition effects extracted via emmeans.

2. **Mixed-effects calibration model:** Test whether the confidence → accuracy mapping differs by condition:
   ```r
   glmer(correct ~ confidence_numeric * condition + judge + (1 | instance_idx),
         data = df, family = binomial)
   ```
   - Main effect of `confidence_numeric`: does higher stated confidence correspond to higher accuracy?
   - `confidence × condition` interaction: does the calibration slope differ by condition (i.e., are NLE-condition judges better or worse calibrated)?
   This avoids the collider problem because correctness is the *outcome*, not a predictor of confidence.

3. **Empirical calibration curves:** For each condition, plot the 5 Likert levels on the x-axis and the observed accuracy rate (proportion correct) on the y-axis. Perfect calibration would show a monotonic increase. Deviations reveal miscalibration patterns (e.g., overconfidence at high Likert levels).

4. **Somers' D (ordinal association):** Compute Somers' D between ordinal confidence and binary accuracy for each condition. This measures whether higher confidence monotonically tracks higher accuracy without assuming specific probability values for the Likert scale. Higher Somers' D = better calibrated.

5. **Conditional confidence tables (descriptive ONLY):** Report mean confidence among correct and incorrect judgments per condition. Same information as the old interaction, but explicitly labelled as descriptive, not causal — the groups being compared (correct vs incorrect within a condition) may differ in composition across conditions.

**Why not Brier score / ECE:** The council flagged that linearly rescaling a 1–5 Likert scale to [0, 1] and computing Brier scores is invalid. Brier and ECE require calibrated probabilities; a judge selecting "4" does not mean they are exactly 75% confident. The ordinal methods above (calibration model, Somers' D, empirical calibration curves) respect the ordinal nature of the confidence scale.

### 4.5 Evidence for Null Results: ROPE and Bayes Factors

**The problem:** Four of five experiments show null accuracy effects. Saying "the CI spans 1" does not distinguish between "the effect is truly absent" and "we lack power to detect it." A statistics professor will want positive evidence for the null.

**Revised approach — ROPE analysis:**

1. **Region of Practical Equivalence (ROPE):** Define a ROPE around OR = 1 on the log-odds scale. A ROPE of [−0.18, +0.18] on log-odds corresponds to OR ∈ [0.84, 1.20], which represents effects too small to matter practically (< 5 percentage points on accuracy). Report the percentage of the posterior distribution falling inside the ROPE. If > 95% of the posterior is inside the ROPE, conclude practical equivalence.

   **Justification for ROPE width:** In this domain (error bucket classification with ~40% baseline accuracy), a 5-percentage-point change would be the minimum practically meaningful shift. This translates to approximately OR = 1.20 at 40% baseline, giving the symmetric ROPE [0.84, 1.20] or [−0.18, +0.18] on log-odds. Note: the OR-to-probability mapping is baseline-dependent; this ROPE is chosen as a uniform conservative benchmark across experiments.

2. **Reporting template for null results:**
   > "The posterior OR was X.XX (95% CrI [X.XX, X.XX]), with XX% of the posterior falling inside the ROPE [0.84, 1.20], providing [moderate/strong] evidence that NLEs do not meaningfully affect accuracy."

**Why not Bayes Factors:** The council flagged that Savage-Dickey BF₀₁ requires careful specification of the exact parameter, prior, and contrast coding for each comparison. In complex multilevel logistic models, this is error-prone. ROPE with posterior summaries is simpler, more transparent, and sufficient for a master's thesis.

### 4.6 Random Slopes: Attempted, Confirmed Singular

**Decision:** The primary model uses a random intercept only: `(1 | instance_idx)`. Random slopes for condition by instance were tested as the initial specification but are not retained.

**Rationale:** In principle, the "keep it maximal" approach (Barr et al., 2013) would suggest fitting `(1 + condition | instance_idx)` to allow the condition effect to vary across instances. This was attempted for all five experiments. In every case, the random slope variance collapsed to zero (singular fit), meaning the data do not support estimating heterogeneity of condition effects across instances. This is consistent with the finding that the condition effect — or lack thereof — is uniform across instances.

**Protocol:** For each experiment, fit the random-slopes model first. If singular (expected), fall back to random-intercept-only with explicit documentation:

```r
# Attempt maximal model
m_slopes <- glmer(correct ~ condition * judge + (1 + condition | instance_idx),
                  data = df, family = binomial)
# Check singularity
isSingular(m_slopes)  # TRUE in all experiments
# Fall back to random-intercept-only
m_primary <- glmer(correct ~ condition * judge + (1 | instance_idx),
                   data = df, family = binomial)
```

**Thesis language:**
> "Random slopes for condition by instance were tested as the initial specification following Barr et al. (2013). In all five experiments, the random slope variance was estimated at zero (singular fit), indicating that the condition effect does not vary meaningfully across instances. The random-intercept-only model was therefore retained."

### 4.7 Scores as Binary / Ordinal

**Accuracy:** Strictly binary (correct/incorrect). Modelled with GLMM, binomial family, logit link.

**Confidence:** Ordinal 1–5 Likert scale. Modelled with CLMM (cumulative link mixed model) with logit link, which respects the ordinal nature (Christensen, 2019). The proportional odds assumption is checked by comparing with a partial proportional odds model.

---

## 5. General Analysis Protocol

Applied to every experiment. Run for both accuracy and confidence.

### Step 1: Descriptive Statistics

- Accuracy rate per condition (with 95% Wilson CIs)
- Mean and median confidence per condition
- Contingency tables: condition × judge × accuracy
- Check for floor/ceiling effects on confidence

### Step 2: Primary GLMM for Accuracy

```r
# Always fit the interaction model (no step-down selection)
m_primary <- glmer(correct ~ condition * judge + (1 | instance_idx),
                   data = df, family = binomial)

# Reduced model (no condition) for omnibus test
m_red <- glmer(correct ~ judge + (1 | instance_idx),
               data = df, family = binomial)

# Omnibus condition effect via marginal means
emm <- emmeans(m_primary, ~ condition)  # averaged across judges
pairs(emm, adjust = "holm", type = "response")

# Judge-specific condition effects
emm_judge <- emmeans(m_primary, ~ condition | judge, type = "response")

# Report interaction estimate for transparency
# (extracted from model summary, not used for model selection)
```

**No step-down:** The interaction model is always retained. Marginal condition effects (averaged across judges) are the primary estimand. Judge-specific effects are reported alongside. The interaction estimate is reported with CI for transparency but is not used as a gatekeeper.

### Step 3: Planned Contrasts and Pairwise Comparisons

```r
emm <- emmeans(m_primary, ~ condition, type = "response")
pairs(emm, adjust = "holm")

# Experiment-specific planned contrasts (see §6)
```

Report: OR with 95% CI, predicted probabilities, Holm-corrected p-values.

### Step 4: Primary CLMM for Confidence

```r
c_full <- clmm(confidence ~ condition * judge + (1 | instance_idx), data = df)
c_add  <- clmm(confidence ~ condition + judge + (1 | instance_idx), data = df)
c_red  <- clmm(confidence ~ judge + (1 | instance_idx), data = df)

# Same interaction → additive → omnibus testing sequence
anova(c_add, c_full)
anova(c_red, c_add)
```

### Step 5: Calibration Analysis (replaces overconfidence interaction)

```r
# 1. Mixed-effects calibration model
m_cal <- glmer(correct ~ as.numeric(confidence) * condition + judge + (1 | instance_idx),
               data = df, family = binomial)
# Check: does confidence × condition interaction indicate different calibration by condition?

# 2. Empirical calibration curves
df %>%
  group_by(condition, confidence) %>%
  summarise(accuracy = mean(correct), n = n()) %>%
  ggplot(aes(x = confidence, y = accuracy, colour = condition)) +
  geom_line() + geom_point(aes(size = n))
# Perfect calibration: monotonic increase

# 3. Somers' D per condition
library(Hmisc)
df %>%
  group_by(condition) %>%
  summarise(somers_d = somers2(as.numeric(confidence), correct)["Dxy"])

# 4. Conditional confidence table (descriptive only)
df %>%
  group_by(condition, correct) %>%
  summarise(mean_conf = mean(as.numeric(confidence)),
            median_conf = median(as.numeric(confidence)),
            n = n())
```

### Step 6: Model Diagnostics

```r
library(DHARMa)
sim_res <- simulateResiduals(m_primary)
plot(sim_res)
testDispersion(sim_res)
check_convergence(m_primary)
check_singularity(m_primary)
```

### Step 7: Bayesian Robustness with ROPE

```r
library(brms)

bm <- brm(
  correct ~ condition + judge + (1 | instance_idx),
  data = df,
  family = bernoulli(link = "logit"),
  prior = c(
    prior(normal(0, 1.5), class = "b"),
    prior(student_t(3, 0, 2.5), class = "sd")
  ),
  chains = 4, iter = 4000, cores = 4
)

# Posterior OR + 95% CrI
posterior_summary(bm, pars = "b_conditionEXplusT")
exp(fixef(bm))  # ORs

# ROPE analysis
rope_range <- c(-0.18, 0.18)  # log-odds, corresponds to OR in [0.84, 1.20]
hypothesis(bm, "conditionEXplusT = 0", rope = rope_range)
# Report: % of posterior inside ROPE
```

---

## 6. Experiment-Specific Plans

### 6.1 Experiment 1: Forward Simulatability (E1)

**N = 720.** 5 conditions × 60 instances × varying rows per condition.

**Primary accuracy model:**
```r
m1 <- glmer(correct ~ condition * judge + (1 | instance_idx),
            data = e1, family = binomial)
```

**Planned contrasts:**
1. **E+X+T vs X+T** — the critical NLE effect: does adding NLE to full structured info help?
2. **X+T vs Baseline** — does non-NLE information help?
3. All pairwise with Holm correction (exploratory)

**Confidence model:** CLMM with same structure.

**Calibration:** Mixed-effects calibration model, empirical calibration curves, Somers' D per condition. Conditional confidence table (descriptive).

**Robustness:** Add `true_bucket` as covariate to control for class imbalance.

### 6.2 Experiment 2: Selective Reliance (E2)

**N = 720.** 2×2 factorial: poisoning (Baseline/OOD) × NLE (No/Yes).

**Primary accuracy model:**
```r
m2 <- glmer(correct ~ poisoning * has_nle * judge + (1 | instance_idx),
            data = e2, family = binomial)

# Simplify: test the three-way first, then drop if n.s.
m2_2way <- glmer(correct ~ poisoning * has_nle + poisoning * judge + has_nle * judge +
                  (1 | instance_idx), data = e2, family = binomial)
```

**The critical test:** poisoning × has_nle interaction. If significant → simple effects: NLE effect within Baseline, NLE effect within OOD.

**Judge handling for E2:** Because of the known judge asymmetry (GPT-4o rarely flags OOD), test the three-way poisoning × has_nle × judge interaction. If significant, report simple effects per judge.

**Confidence model:** CLMM with same factorial structure.

**Calibration:** Separate calibration analysis (calibration model, Somers' D) for Baseline and OOD data × NLE presence.

### 6.3 Experiment 3: Counterfactual Simulatability (E3)

**N = 360.** 2 conditions: X vs E+X.

**Primary accuracy model:**
```r
m3 <- glmer(correct ~ condition * judge + (1 | instance_idx),
            data = e3, family = binomial)
```

Only 2 condition levels, so the condition coefficient IS the test. The primary question is whether E+X outperforms X, not whether either exceeds some chance level.

**No chance baseline test:** The council flagged that testing against 50% is invalid for a 3-category task (higher/lower/similar). Even a naive "always guess higher" strategy yields 46.6% accuracy (28/60). The task's difficulty is characterised descriptively (per-class accuracy, confusion matrix) rather than via a formal chance-level test.

**True direction covariate:** Add `true_direction` as a fixed effect to check if NLEs help differentially for "higher" vs "lower" directions.

**Per-class performance (descriptive):** Report confusion matrices and per-class recall for both conditions. This reveals whether NLEs improve accuracy on one direction class at the expense of another, which overall accuracy would mask.

### 6.4 Experiment 4: Mental Model Transfer (E4)

**N = 330.** 2 conditions: Baseline training vs NLE training. 55 test positions (sliding window).

**Primary accuracy model:**
```r
m4 <- glmer(correct ~ condition * judge + (1 | instance_idx),
            data = e4, family = binomial)
```

**Instance difficulty covariate:** E4 has extreme random intercept variance (27.04), suggesting huge variation in instance difficulty. Add a continuous difficulty proxy:

```r
# Difficulty = absolute prediction error (higher = harder to classify)
e4$abs_error <- abs(e4$predicted_value - e4$true_value)  # or from metadata
m4_diff <- glmer(correct ~ condition + judge + scale(abs_error) + (1 | instance_idx),
                 data = e4, family = binomial)
```

This absorbs some of the instance variance via a fixed covariate, potentially improving power.

**Trial order and sliding-window autocorrelation:** The sliding window creates 80% overlap between adjacent training sets (positions t and t+1 share 4 of 5 training instances). This induces temporal autocorrelation beyond what the instance random intercept captures. Include `trial_order` (test position) as a linear fixed-effect covariate. Also test `condition × trial_order` interaction (do NLE-trained judges improve more over time?). Acknowledge in the thesis that E4's overlapping-window structure means inferential claims should be more cautious than in the other experiments.

**Robustness:** `true_bucket` covariate. Note convergence issues with "very_large" bucket (quasi-complete separation) — use Bayesian model if frequentist fails.

### 6.5 Experiment 5: Placebic Control (E5)

**N = 600.** 3 conditions: Baseline, Real NLE, Placebo NLE.

**Primary accuracy model:**
```r
m5 <- glmer(correct ~ condition * judge + (1 | instance_idx),
            data = e5, family = binomial)
```

**Planned contrasts (critical — these are the scientific core of E5):**
```r
emm5 <- emmeans(m5, ~ condition, type = "response")
contrast(emm5, method = list(
  "Real vs Baseline"    = c(-1, 1, 0),
  "Placebo vs Baseline" = c(-1, 0, 1),
  "Real vs Placebo"     = c(0, 1, -1)   # THE critical content-vs-presence test
), adjust = "holm")
```

**Interpretation logic:**
- Real ≈ Placebo on accuracy → effect (if any) is presence-driven, not content-driven
- Real ≈ Placebo on confidence → confidence inflation is placebic
- Real > Placebo on either → content matters

**ROPE analysis is especially important here** for the Real vs Placebo contrast: we need to show these are practically equivalent, not just "not significantly different."

**Derangement sensitivity limitation:** The placebo NLEs were generated using a single random derangement (a permutation with no fixed points). Results may depend on the specific placebo assignment. Since the experiment is already run and NLEs are fixed, this cannot be re-run with alternative derangements. Acknowledge this as a limitation in the thesis. The finding is strengthened by the fact that Real vs Placebo is non-significant across *both* accuracy and confidence, making it unlikely that a different derangement would produce a qualitatively different result.

---

## 7. Effect Sizes

### 7.1 Odds Ratio (Primary)

OR with 95% CI from GLMM fixed effects. Reported for all condition contrasts.

**Do not use benchmark thresholds** (Small=1.5, Medium=2.5, Large=4.0). These are arbitrary and context-dependent. Instead, contextualise ORs by converting to predicted probability differences at the observed baseline rate.

### 7.2 Predicted Probability Differences

More interpretable than ORs for a non-statistical audience. Report EMM-derived predicted probabilities per condition and their differences.

```r
emm <- emmeans(model, ~ condition, type = "response")
pairs(emm, type = "response")  # gives probability differences
```

### 7.3 Cumulative Odds Ratios (Confidence)

From CLMMs. Report for condition contrasts on confidence. Also report mean/median confidence per condition for readability.

---

## 8. Multiple Comparisons

### Across experiments

**No correction.** Each experiment tests a distinct, pre-specified hypothesis. Standard in multi-experiment studies.

### Within experiments

- **Planned contrasts** (1–2 per experiment): No correction, or Holm if >2
- **Exploratory pairwise** (all pairs in E1, E5): Holm correction
- **E5's three contrasts:** Holm-corrected (3 tests)

### Interaction tests

If testing judge × condition alongside the omnibus condition effect, these are part of the same model and do not require separate multiplicity correction.

---

## 9. Robustness and Sensitivity Analyses

### 9.1 Bayesian GLMMs with ROPE

**All experiments.** Fit Bayesian GLMMs via `brms` with weakly informative priors:
- Fixed effects: Normal(0, 1.5) on log-odds — covers ORs from ~0.05 to ~20
- Random effect SD: Student-t(3, 0, 2.5)

Report: posterior median OR, 95% CrI, % posterior in ROPE.

**ROPE:** [−0.18, +0.18] on log-odds ≈ OR ∈ [0.84, 1.20].

**ROPE interpretation:**
- < 50% inside ROPE: inconclusive
- 50–89% inside ROPE: suggestive of practical equivalence
- 90–95% inside ROPE: moderate evidence for practical equivalence
- > 95% inside ROPE: strong evidence for practical equivalence

### 9.2 Judge-Specific Analyses

Run every primary model separately for GPT-4o and DeepSeek-R1. If effects replicate directionally across both judges, the finding is robust. Report per-judge ORs and CIs.

Note: per-judge models have half the data and may show convergence issues (especially E2 GPT-4o with quasi-separation). Use Bayesian fallback if needed.

### 9.3 Generator Effects (NLE Conditions Only)

For experiments with NLE conditions (E1 E+X+T, E2 NLE, E3 E+X, E4 NLE training, E5 Real/Placebo):

```r
# Subset to NLE conditions
df_nle <- df %>% filter(has_nle == TRUE)

# Generator main effect
m_gen <- glmer(correct ~ condition + generator + judge + (1 | instance_idx),
               data = df_nle, family = binomial)

# Same-family bias
df_nle$same_family <- (df_nle$generator == df_nle$judge)
m_fam <- glmer(correct ~ condition + same_family + judge + (1 | instance_idx),
               data = df_nle, family = binomial)
```

### 9.4 Generator-Equalized Sensitivity

To address the row imbalance (NLE conditions: 4 rows/instance; no-NLE: 2 rows/instance):

```r
set.seed(42)
# For each NLE instance, randomly keep one generator
df_eq <- df %>%
  group_by(instance_idx, condition, judge) %>%
  slice_sample(n = 1) %>%
  ungroup()

# Refit primary model on balanced data
m_eq <- glmer(correct ~ condition + judge + (1 | instance_idx),
              data = df_eq, family = binomial)
```

Repeat with 100 random seeds and report the distribution of ORs and p-values.

### 9.5 Class Imbalance Control (E1, E4, E5)

Add `true_bucket` as covariate, optionally test `condition × true_bucket` interaction:

```r
m_bucket <- glmer(correct ~ condition + true_bucket + judge + (1 | instance_idx),
                  data = df, family = binomial)
```

### 9.6 HC3 / Sandwich Standard Errors

As a check against potential misspecification of the random effects structure, compute cluster-robust standard errors:

```r
library(clubSandwich)
coef_test(m_primary, vcov = "CR2", cluster = df$instance_idx)
```

### 9.7 Random Slopes

See §4.6. Random slopes for condition by instance were tested for all experiments and were singular in every case, confirming that the random-intercept-only specification is appropriate. No further action needed.

---

## 10. Implementation Checklist

### Per Experiment (E1–E5)

- [ ] **Load and prepare data** (factor levels, check for NAs)
- [ ] **Descriptive statistics** (accuracy and confidence per condition, per judge)
- [ ] **Random slopes test**: fit `(1 + condition | instance_idx)`, confirm singular, document
- [ ] **Step 2: Primary GLMM for accuracy**
  - [ ] Fit interaction model (`condition * judge`) — always retained
  - [ ] Extract marginal condition effects via emmeans (averaged across judges)
  - [ ] Extract judge-specific condition effects
  - [ ] Report interaction estimate with CI
  - [ ] Check convergence, singularity (DHARMa)
  - [ ] Planned contrasts via emmeans
  - [ ] Exploratory pairwise with Holm
- [ ] **Step 4: Primary CLMM for confidence**
  - [ ] Same interaction structure as accuracy
  - [ ] Marginal and judge-specific effects
- [ ] **Step 5: Calibration analysis**
  - [ ] Mixed-effects calibration model (`correct ~ confidence * condition`)
  - [ ] Empirical calibration curves (Likert level vs accuracy per condition)
  - [ ] Somers' D per condition
  - [ ] Conditional confidence table (descriptive: mean confidence | correct/incorrect × condition)
- [ ] **Step 6: Diagnostics** (DHARMa, convergence, singularity)
- [ ] **Step 7: Bayesian GLMM**
  - [ ] Fit brms model
  - [ ] Posterior OR + 95% CrI
  - [ ] ROPE analysis (% posterior in [−0.18, +0.18])

### Experiment-Specific Additions

- [ ] **E1:** `true_bucket` covariate robustness
- [ ] **E2:** poisoning × NLE interaction; three-way with judge; simple effects per data type
- [ ] **E3:** `true_direction` covariate; per-class confusion matrices
- [ ] **E4:** Instance difficulty covariate; trial order + condition × trial_order
- [ ] **E5:** Real vs Placebo ROPE analysis (critical); planned content-vs-presence contrasts

### Cross-Experiment

- [ ] **Sensitivity analyses** per §9 (judge-specific, generator, same-family, generator-equalized, random slopes)
- [ ] **Cross-experiment summary table** (OR, p, Bayesian OR, CrI, ROPE%)
- [ ] **Calibration comparison** across all 5 experiments (Somers' D, calibration curves)

---

## References

- Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language*, 68(3), 255–278.
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. *Journal of Statistical Software*, 67(1).
- Bürkner, P.-C. (2017). brms: An R package for Bayesian multilevel models using Stan. *Journal of Statistical Software*, 80(1).
- Christensen, R. H. B. (2019). *ordinal: Regression Models for Ordinal Data.* R package.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. *Advances in Methods and Practices in Psychological Science*, 1(2), 270–280.
- Lenth, R. V. (2024). *emmeans: Estimated Marginal Means.* R package.
- Maxwell, S. E., Delaney, H. D., & Kelley, K. (2017). *Designing Experiments and Analyzing Data* (3rd ed.). Routledge.

---

*Last updated: April 2025 (post-council revision)*
