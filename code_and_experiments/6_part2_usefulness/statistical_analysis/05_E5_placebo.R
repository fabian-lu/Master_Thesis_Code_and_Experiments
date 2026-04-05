# ==============================================================================
# 05_E5_placebo.R — Experiment 5: Placebic NLE Control
# ==============================================================================
# Research questions:
#   1. Any accuracy difference across the 3 conditions?
#   2. Is Placebo_NLE different from Real_NLE? (content vs presence)
#   3. Does confidence differ? Do placebic NLEs boost confidence like real ones?
#
# Design: 3 conditions (Baseline, Real_NLE, Placebo_NLE), 60 instances, 600 rows
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 5: PLACEBIC NLE CONTROL\n")
cat("================================================================\n")

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================

cat("\n--- Descriptive Statistics ---\n")
desc_e5 <- descriptive_by_condition(e5)
print(desc_e5)

cat("\n--- By Condition × Judge ---\n")
print(descriptive_by_condition_judge(e5))

cat("\n--- Overconfidence Descriptive ---\n")
print(overconfidence_descriptive(e5))

cat("\n--- True Bucket Distribution ---\n")
print(table(e5$true_bucket))

# ==============================================================================
# 2. Primary Analysis: Accuracy (GLMM)
# ==============================================================================

cat("\n--- Primary GLMM: Accuracy ---\n")

m5_full <- glmer(correct ~ condition + judge + (1 | instance_idx),
                 data = e5, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

m5_reduced <- glmer(correct ~ judge + (1 | instance_idx),
                    data = e5, family = binomial,
                    control = glmerControl(optimizer = "bobyqa"))

# Omnibus LRT
cat("\nOmnibus LRT (condition effect on accuracy):\n")
lrt_acc <- anova(m5_reduced, m5_full, test = "Chisq")
print(lrt_acc)

# Model summary
cat("\nModel summary:\n")
print(summary(m5_full))

# Odds ratios
cat("\nOdds Ratios:\n")
print(extract_ors(m5_full))

# ==============================================================================
# 3. Planned Contrasts (Critical)
# ==============================================================================

cat("\n--- Planned Contrasts ---\n")

emm5 <- emmeans(m5_full, ~ condition, type = "response")
cat("\nEstimated Marginal Means (probability scale):\n")
print(summary(emm5))

# Three planned contrasts with Holm correction
cat("\nPlanned contrasts (Holm-corrected):\n")
planned <- contrast(emm5, method = list(
  "Real_NLE vs Baseline"   = c(-1, 1, 0),
  "Placebo_NLE vs Baseline" = c(-1, 0, 1),
  "Placebo_NLE vs Real_NLE" = c(0, -1, 1)  # CRITICAL TEST
), adjust = "holm")
print(summary(planned))

# ==============================================================================
# 4. Confidence Analysis (CLMM)
# ==============================================================================

cat("\n--- Confidence Analysis (CLMM) ---\n")

c5_full <- clmm(confidence ~ condition + judge + (1 | instance_idx),
                data = e5)

c5_reduced <- clmm(confidence ~ judge + (1 | instance_idx),
                   data = e5)

cat("\nOmnibus LRT (condition effect on confidence):\n")
lrt_conf <- anova(c5_reduced, c5_full)
print(lrt_conf)

cat("\nCLMM summary:\n")
print(summary(c5_full))

# Confidence contrasts
emm5c <- emmeans(c5_full, ~ condition)
cat("\nConfidence planned contrasts (Holm-corrected):\n")
planned_conf <- contrast(emm5c, method = list(
  "Real_NLE vs Baseline"   = c(-1, 1, 0),
  "Placebo_NLE vs Baseline" = c(-1, 0, 1),
  "Placebo_NLE vs Real_NLE" = c(0, -1, 1)
), adjust = "holm")
print(summary(planned_conf))

# ==============================================================================
# 5. Overconfidence Analysis
# ==============================================================================

cat("\n--- Overconfidence Analysis ---\n")

e5$correctness <- factor(ifelse(e5$correct == 1, "correct", "incorrect"),
                         levels = c("correct", "incorrect"))

c5_overconf <- clmm(confidence ~ correctness * condition + judge + (1 | instance_idx),
                    data = e5)

cat("\nOverconfidence interaction model:\n")
print(summary(c5_overconf))

# Among incorrect only: does confidence differ by condition?
cat("\nConfidence among INCORRECT responses only:\n")
e5_wrong <- e5 %>% filter(correct == 0)
if (nrow(e5_wrong) > 20) {
  c5_wrong <- clmm(confidence ~ condition + judge + (1 | instance_idx),
                   data = e5_wrong)
  print(summary(c5_wrong))
} else {
  cat("Too few incorrect responses to model.\n")
}

# ==============================================================================
# 6. Robustness: Control for true_bucket
# ==============================================================================

cat("\n--- Robustness: Controlling for true_bucket ---\n")

e5$true_bucket <- factor(e5$true_bucket,
                         levels = c("small", "medium", "large", "very_large"))

m5_bucket <- glmer(correct ~ condition + true_bucket + judge + (1 | instance_idx),
                   data = e5, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

cat("\nModel with true_bucket covariate:\n")
print(summary(m5_bucket))

# ==============================================================================
# 7. Diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")
run_glmm_diagnostics(m5_full, "E5 accuracy GLMM")

cat("\n================================================================\n")
cat("  E5 ANALYSIS COMPLETE\n")
cat("================================================================\n")
