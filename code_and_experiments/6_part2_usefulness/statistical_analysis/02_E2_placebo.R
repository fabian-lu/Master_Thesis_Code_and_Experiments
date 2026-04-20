# ==============================================================================
# 02_E2_placebo.R — Experiment 2: Placebic NLE Control
# ==============================================================================
# RQ-U2: Is confidence inflation driven by NLE content or mere presence?
# Design: 3 conditions (Baseline, Real NLE, Placebo NLE), 60 instances, N=600
# Critical test: Real vs Placebo equivalence (ROPE)
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/statistical_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 2: PLACEBIC NLE CONTROL\n")
cat("================================================================\n")

e2$condition <- factor(e2$condition, levels = c("Baseline", "Real_NLE", "Placebo_NLE"))

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
print(descriptives(e2))
print(descriptives_by_judge(e2))

# ==============================================================================
# 2. Primary GLMM: Accuracy (condition * judge)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

acc2 <- fit_accuracy_model(e2)

cat("\nOmnibus LRT:\n")
print(acc2$lrt)

cat("\nModel summary:\n")
print(summary(acc2$model))

cat("\nMarginal EMMs:\n")
print(summary(acc2$emm_cond))

cat("\nJudge-specific EMMs:\n")
print(summary(acc2$emm_judge))

# Critical planned contrasts
cat("\nPlanned contrasts:\n")
emm2 <- acc2$emm_cond
cat("\n  Real vs Baseline:\n")
print(contrast(emm2, method = list("Real vs Baseline" = c(-1, 1, 0))))
cat("\n  Placebo vs Baseline:\n")
print(contrast(emm2, method = list("Placebo vs Baseline" = c(-1, 0, 1))))
cat("\n  Real vs Placebo (CRITICAL — content vs presence):\n")
print(contrast(emm2, method = list("Real vs Placebo" = c(0, 1, -1))))

cat("\nAll pairwise (Holm):\n")
print(pairs(emm2, adjust = "holm"))

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

conf2 <- fit_confidence_model(e2)

cat("\nConfidence LRT:\n")
print(conf2$lrt)

cat("\nCLMM summary:\n")
print(summary(conf2$model))

cat("\nConfidence contrasts:\n")
emm2_conf <- conf2$emm_cond
if (!is.null(emm2_conf)) {
  cat("\n  Real vs Baseline:\n")
  print(contrast(emm2_conf, method = list("Real vs Baseline" = c(-1, 1, 0))))
  cat("\n  Placebo vs Baseline:\n")
  print(contrast(emm2_conf, method = list("Placebo vs Baseline" = c(-1, 0, 1))))
  cat("\n  Real vs Placebo:\n")
  print(contrast(emm2_conf, method = list("Real vs Placebo" = c(0, 1, -1))))
} else {
  cat("  CLMM emmeans failed (likely NaN SEs from near-zero random intercept variance).\n")
  cat("  Falling back to additive model emmeans:\n")
  emm2_conf_add <- tryCatch(emmeans(conf2$model_add, ~ condition), error = function(e) NULL)
  if (!is.null(emm2_conf_add)) {
    print(pairs(emm2_conf_add, adjust = "holm"))
  } else {
    cat("  Additive emmeans also failed.\n")
  }
}

# ==============================================================================
# 4. Calibration
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
cal2 <- run_calibration(e2)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(acc2$model, "E2 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE (especially Real vs Placebo)
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm2 <- run_bayesian_rope(e2)

# Extra: ROPE specifically for Real vs Placebo contrast
if (!is.null(bm2)) {
  cat("\n  ROPE for Real vs Placebo (the critical equivalence test):\n")
  post <- as.data.frame(bm2)
  # Real_NLE coefficient - Placebo_NLE coefficient
  real_col <- grep("Real", names(post), value = TRUE)
  plac_col <- grep("Placebo", names(post), value = TRUE)
  if (length(real_col) > 0 & length(plac_col) > 0) {
    diff_post <- post[[real_col[1]]] - post[[plac_col[1]]]
    rope <- c(-0.18, 0.18)
    in_rope <- mean(diff_post >= rope[1] & diff_post <= rope[2]) * 100
    cat(sprintf("  Real-Placebo posterior: median OR = %.3f\n", exp(median(diff_post))))
    cat(sprintf("  95%% CrI: [%.3f, %.3f]\n", exp(quantile(diff_post, 0.025)),
                exp(quantile(diff_post, 0.975))))
    cat(sprintf("  ROPE [%.2f, %.2f]: %.1f%% inside\n", rope[1], rope[2], in_rope))
  }
}

# ==============================================================================
# 7. Sensitivity
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

run_judge_specific(e2, "correct ~ condition + (1 | instance_idx)")
run_generator_test(e2, rlang::expr(condition != "Baseline"))
run_random_slopes(e2, "correct ~ condition * judge + (condition | instance_idx)")

# E2-specific: true_bucket covariate
if ("true_bucket" %in% names(e2)) {
  cat("\ntrue_bucket robustness:\n")
  e2$true_bucket <- factor(e2$true_bucket)
  tryCatch({
    m2_bucket <- glmer(correct ~ condition * judge + true_bucket + (1 | instance_idx),
                       data = e2, family = binomial,
                       control = glmerControl(optimizer = "bobyqa"))
    print(summary(m2_bucket))
  }, error = function(e) cat(sprintf("Error: %s\n", e$message)))
}

cat("\nNote: Placebo NLEs were generated using a single random derangement.\n")
cat("Results may depend on the specific assignment. This is acknowledged as a limitation.\n")

cat("\n================================================================\n")
cat("  E2 COMPLETE\n")
cat("================================================================\n")
