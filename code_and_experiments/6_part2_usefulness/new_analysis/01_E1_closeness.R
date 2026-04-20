# ==============================================================================
# 01_E1_closeness.R — Experiment 1: Forward Simulatability (Closeness Task)
# ==============================================================================
# RQ-U1: Do NLEs improve ability to classify prediction error magnitude?
# Design: 5 conditions (B, X, T, X+T, E+X+T), 60 instances, N=720
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/new_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 1: FORWARD SIMULATABILITY\n")
cat("================================================================\n")

e1$condition <- factor(e1$condition, levels = c("Baseline", "X", "T", "X+T", "E+X+T"))

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
print(descriptives(e1))
print(descriptives_by_judge(e1))

cat("\nTrue bucket distribution:\n")
print(table(e1$true_bucket))

# ==============================================================================
# 2. Primary GLMM for Accuracy (condition * judge)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

acc <- fit_accuracy_model(e1)

cat("\nOmnibus LRT (condition):\n")
print(acc$lrt)

cat("\nModel summary:\n")
print(summary(acc$model))

cat("\nMarginal condition EMMs (averaged over judges):\n")
print(summary(acc$emm_cond))

cat("\nJudge-specific EMMs:\n")
print(summary(acc$emm_judge))

# Planned contrasts
cat("\nPlanned: E+X+T vs X+T (NLE effect):\n")
print(contrast(acc$emm_cond, method = list("E+X+T vs X+T" = c(0, 0, 0, -1, 1))))

cat("\nPlanned: X+T vs Baseline (information effect):\n")
print(contrast(acc$emm_cond, method = list("X+T vs Baseline" = c(-1, 0, 0, 1, 0))))

cat("\nAll pairwise (Holm):\n")
print(pairs(acc$emm_cond, adjust = "holm"))

# ==============================================================================
# 3. Primary CLMM for Confidence (condition * judge)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

conf <- fit_confidence_model(e1)

cat("\nOmnibus LRT (condition on confidence):\n")
print(conf$lrt)

cat("\nCLMM summary:\n")
print(summary(conf$model))

cat("\nConfidence EMMs:\n")
print(summary(conf$emm_cond))

cat("\nConfidence: E+X+T vs X+T:\n")
print(contrast(conf$emm_cond, method = list("E+X+T vs X+T" = c(0, 0, 0, -1, 1))))

cat("\nConfidence pairwise (Holm):\n")
print(pairs(conf$emm_cond, adjust = "holm"))

# ==============================================================================
# 4. Calibration Analysis (replaces overconfidence interaction)
# ==============================================================================
cat("\n--- 4. Calibration Analysis ---\n")
cal <- run_calibration(e1)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(acc$model, "E1 accuracy GLMM")

# ==============================================================================
# 6. Bayesian GLMM with ROPE
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm1 <- run_bayesian_rope(e1)

# ==============================================================================
# 7. Sensitivity Analyses
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

# Judge-specific
run_judge_specific(e1, "correct ~ condition + (1 | instance_idx)")

# Generator effects (E+X+T condition only)
run_generator_test(e1, rlang::expr(condition == "E+X+T"))

# Random slopes
run_random_slopes(e1, "correct ~ condition * judge + (condition | instance_idx)")

# ==============================================================================
# 8. E1-Specific: true_bucket covariate
# ==============================================================================
cat("\n--- 8. Robustness: true_bucket covariate ---\n")
e1$true_bucket <- factor(e1$true_bucket, levels = c("small", "medium", "large", "very_large"))
tryCatch({
  m_bucket <- glmer(correct ~ condition * judge + true_bucket + (1 | instance_idx),
                    data = e1, family = binomial,
                    control = glmerControl(optimizer = "bobyqa"))
  cat("Model with true_bucket:\n")
  print(summary(m_bucket))
}, error = function(e) cat(sprintf("true_bucket model error: %s\n", e$message)))

cat("\n================================================================\n")
cat("  E1 COMPLETE\n")
cat("================================================================\n")
