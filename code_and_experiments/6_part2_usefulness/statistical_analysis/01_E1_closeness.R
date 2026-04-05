# ==============================================================================
# 01_E1_closeness.R — Experiment 1: Closeness Task
# ==============================================================================
# Research questions:
#   1. Any difference in accuracy across the 5 conditions?
#   2. Does E+X+T vs X+T specifically change accuracy? (NLE effect)
#   3. Does condition affect confidence?
#   4. Overconfidence effect? (higher confidence when wrong, especially with NLEs)
#
# Design: 5 conditions (Baseline, X, T, X+T, E+X+T), 60 instances, 720 rows
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 1: CLOSENESS TASK\n")
cat("================================================================\n")

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================

cat("\n--- Descriptive Statistics ---\n")
desc_cond <- descriptive_by_condition(e1)
print(desc_cond)

cat("\n--- By Condition × Judge ---\n")
desc_cond_judge <- descriptive_by_condition_judge(e1)
print(desc_cond_judge)

cat("\n--- Overconfidence Descriptive ---\n")
overconf <- overconfidence_descriptive(e1)
print(overconf)

cat("\n--- True Bucket Distribution ---\n")
print(table(e1$true_bucket))

# ==============================================================================
# 2. Primary Analysis: Accuracy (GLMM)
# ==============================================================================

cat("\n--- Primary GLMM: Accuracy ---\n")

m1_full <- glmer(correct ~ condition + judge + (1 | instance_idx),
                 data = e1, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

m1_reduced <- glmer(correct ~ judge + (1 | instance_idx),
                    data = e1, family = binomial,
                    control = glmerControl(optimizer = "bobyqa"))

# Omnibus LRT for condition effect
cat("\nOmnibus LRT (condition effect on accuracy):\n")
lrt_acc <- anova(m1_reduced, m1_full, test = "Chisq")
print(lrt_acc)

# Model summary
cat("\nModel summary:\n")
print(summary(m1_full))

# Odds ratios
cat("\nOdds Ratios:\n")
ors <- extract_ors(m1_full)
print(ors)

# ==============================================================================
# 3. Planned Contrasts
# ==============================================================================

cat("\n--- Planned Contrasts ---\n")

emm1 <- emmeans(m1_full, ~ condition, type = "response")
cat("\nEstimated Marginal Means (probability scale):\n")
print(summary(emm1))

# Critical contrast: E+X+T vs X+T (NLE effect)
cat("\nPlanned contrast: E+X+T vs X+T (NLE effect):\n")
nle_contrast <- contrast(emm1, method = list(
  "E+X+T vs X+T" = c(0, 0, 0, -1, 1)
), adjust = "none")
print(summary(nle_contrast))

# Additional planned: X+T vs Baseline (information effect)
cat("\nPlanned contrast: X+T vs Baseline (information effect):\n")
info_contrast <- contrast(emm1, method = list(
  "X+T vs Baseline" = c(-1, 0, 0, 1, 0)
), adjust = "none")
print(summary(info_contrast))

# All pairwise with Holm
cat("\nAll pairwise comparisons (Holm-corrected):\n")
pairwise <- pairs(emm1, adjust = "holm")
print(summary(pairwise))

# ==============================================================================
# 4. Confidence Analysis (CLMM)
# ==============================================================================

cat("\n--- Confidence Analysis (CLMM) ---\n")

c1_full <- clmm(confidence ~ condition + judge + (1 | instance_idx),
                data = e1)

c1_reduced <- clmm(confidence ~ judge + (1 | instance_idx),
                   data = e1)

cat("\nOmnibus LRT (condition effect on confidence):\n")
lrt_conf <- anova(c1_reduced, c1_full)
print(lrt_conf)

cat("\nCLMM summary:\n")
print(summary(c1_full))

# Confidence contrasts
emm1_conf <- emmeans(c1_full, ~ condition)
cat("\nConfidence: E+X+T vs X+T:\n")
print(summary(contrast(emm1_conf, method = list(
  "E+X+T vs X+T" = c(0, 0, 0, -1, 1)
), adjust = "none")))

cat("\nConfidence: all pairwise (Holm):\n")
print(summary(pairs(emm1_conf, adjust = "holm")))

# ==============================================================================
# 5. Overconfidence Analysis
# ==============================================================================

cat("\n--- Overconfidence Analysis ---\n")

e1$correctness <- factor(ifelse(e1$correct == 1, "correct", "incorrect"),
                         levels = c("correct", "incorrect"))

c1_overconf <- clmm(confidence ~ correctness * condition + judge + (1 | instance_idx),
                    data = e1)

cat("\nOverconfidence model summary:\n")
print(summary(c1_overconf))

# Test: among INCORRECT responses, does confidence differ by condition?
cat("\nConfidence among INCORRECT responses only:\n")
e1_wrong <- e1 %>% filter(correct == 0)
if (nrow(e1_wrong) > 20) {
  c1_wrong <- clmm(confidence ~ condition + judge + (1 | instance_idx),
                   data = e1_wrong)
  print(summary(c1_wrong))
} else {
  cat("Too few incorrect responses to model.\n")
}

# ==============================================================================
# 6. Robustness: Control for true_bucket
# ==============================================================================

cat("\n--- Robustness: Controlling for true_bucket ---\n")

e1$true_bucket <- factor(e1$true_bucket,
                         levels = c("small", "medium", "large", "very_large"))

m1_bucket <- glmer(correct ~ condition + true_bucket + judge + (1 | instance_idx),
                   data = e1, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

cat("\nModel with true_bucket covariate:\n")
print(summary(m1_bucket))

cat("\nCompare: with vs without true_bucket:\n")
print(anova(m1_full, m1_bucket, test = "Chisq"))

# ==============================================================================
# 7. Diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")
run_glmm_diagnostics(m1_full, "E1 accuracy GLMM")

cat("\n================================================================\n")
cat("  E1 ANALYSIS COMPLETE\n")
cat("================================================================\n")
