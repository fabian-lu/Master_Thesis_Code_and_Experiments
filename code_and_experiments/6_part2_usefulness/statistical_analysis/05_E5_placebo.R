# ==============================================================================
# 05_E5_placebo.R — Experiment 5: Placebic NLE Control
# ==============================================================================
# RQ-U5: Is confidence inflation driven by NLE content or mere presence?
# Design: 3 conditions (Baseline, Real NLE, Placebo NLE), 60 instances, N=600
# Critical test: Real vs Placebo equivalence (ROPE)
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/new_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 5: PLACEBIC NLE CONTROL\n")
cat("================================================================\n")

e5$condition <- factor(e5$condition, levels = c("Baseline", "Real_NLE", "Placebo_NLE"))

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
print(descriptives(e5))
print(descriptives_by_judge(e5))

# ==============================================================================
# 2. Primary GLMM: Accuracy (condition * judge)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

acc5 <- fit_accuracy_model(e5)

cat("\nOmnibus LRT:\n")
print(acc5$lrt)

cat("\nModel summary:\n")
print(summary(acc5$model))

cat("\nMarginal EMMs:\n")
print(summary(acc5$emm_cond))

cat("\nJudge-specific EMMs:\n")
print(summary(acc5$emm_judge))

# Critical planned contrasts
cat("\nPlanned contrasts:\n")
emm5 <- acc5$emm_cond
cat("\n  Real vs Baseline:\n")
print(contrast(emm5, method = list("Real vs Baseline" = c(-1, 1, 0))))
cat("\n  Placebo vs Baseline:\n")
print(contrast(emm5, method = list("Placebo vs Baseline" = c(-1, 0, 1))))
cat("\n  Real vs Placebo (CRITICAL — content vs presence):\n")
print(contrast(emm5, method = list("Real vs Placebo" = c(0, 1, -1))))

cat("\nAll pairwise (Holm):\n")
print(pairs(emm5, adjust = "holm"))

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

conf5 <- fit_confidence_model(e5)

cat("\nConfidence LRT:\n")
print(conf5$lrt)

cat("\nCLMM summary:\n")
print(summary(conf5$model))

cat("\nConfidence contrasts:\n")
emm5_conf <- conf5$emm_cond
if (!is.null(emm5_conf)) {
  cat("\n  Real vs Baseline:\n")
  print(contrast(emm5_conf, method = list("Real vs Baseline" = c(-1, 1, 0))))
  cat("\n  Placebo vs Baseline:\n")
  print(contrast(emm5_conf, method = list("Placebo vs Baseline" = c(-1, 0, 1))))
  cat("\n  Real vs Placebo:\n")
  print(contrast(emm5_conf, method = list("Real vs Placebo" = c(0, 1, -1))))
} else {
  cat("  CLMM emmeans failed (likely NaN SEs from near-zero random intercept variance).\n")
  cat("  Falling back to additive model emmeans:\n")
  emm5_conf_add <- tryCatch(emmeans(conf5$model_add, ~ condition), error = function(e) NULL)
  if (!is.null(emm5_conf_add)) {
    print(pairs(emm5_conf_add, adjust = "holm"))
  } else {
    cat("  Additive emmeans also failed.\n")
  }
}

# ==============================================================================
# 4. Calibration
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
cal5 <- run_calibration(e5)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(acc5$model, "E5 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE (especially Real vs Placebo)
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm5 <- run_bayesian_rope(e5)

# Extra: ROPE specifically for Real vs Placebo contrast
if (!is.null(bm5)) {
  cat("\n  ROPE for Real vs Placebo (the critical equivalence test):\n")
  post <- as.data.frame(bm5)
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

run_judge_specific(e5, "correct ~ condition + (1 | instance_idx)")
run_generator_test(e5, rlang::expr(condition != "Baseline"))
run_random_slopes(e5, "correct ~ condition * judge + (condition | instance_idx)")

# E5-specific: true_bucket covariate
if ("true_bucket" %in% names(e5)) {
  cat("\ntrue_bucket robustness:\n")
  e5$true_bucket <- factor(e5$true_bucket)
  tryCatch({
    m5_bucket <- glmer(correct ~ condition * judge + true_bucket + (1 | instance_idx),
                       data = e5, family = binomial,
                       control = glmerControl(optimizer = "bobyqa"))
    print(summary(m5_bucket))
  }, error = function(e) cat(sprintf("Error: %s\n", e$message)))
}

cat("\nNote: Placebo NLEs were generated using a single random derangement.\n")
cat("Results may depend on the specific assignment. This is acknowledged as a limitation.\n")

cat("\n================================================================\n")
cat("  E5 COMPLETE\n")
cat("================================================================\n")
