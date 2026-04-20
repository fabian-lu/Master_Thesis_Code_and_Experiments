# ==============================================================================
# 04_E4_transfer.R — Experiment 4: Mental Model Transfer
# ==============================================================================
# RQ-U4: Does NLE exposure during training build transferable mental models?
# Design: 2 conditions (Baseline, NLE training), 55 instances, N=330
# Note: Sliding window creates overlapping training sets (autocorrelation caveat)
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/new_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 4: MENTAL MODEL TRANSFER\n")
cat("================================================================\n")

e4$condition <- factor(e4$condition)

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
print(descriptives(e4))
print(descriptives_by_judge(e4))

if ("true_bucket" %in% names(e4)) {
  cat("\nTrue bucket distribution:\n")
  print(table(e4$true_bucket))
}

# ==============================================================================
# 2. Primary GLMM: Accuracy (condition * judge)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

acc4 <- fit_accuracy_model(e4)

cat("\nModel summary:\n")
print(summary(acc4$model))

cat("\nMarginal EMMs:\n")
print(summary(acc4$emm_cond))

cat("\nJudge-specific EMMs:\n")
print(summary(acc4$emm_judge))

cat("\nOmnibus LRT:\n")
print(acc4$lrt)

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

conf4 <- fit_confidence_model(e4)
cat("\nConfidence LRT:\n")
print(conf4$lrt)
cat("\nCLMM summary:\n")
print(summary(conf4$model))

# ==============================================================================
# 4. Calibration
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
cal4 <- run_calibration(e4)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(acc4$model, "E4 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm4 <- run_bayesian_rope(e4)

# ==============================================================================
# 7. Sensitivity
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

run_judge_specific(e4, "correct ~ condition + (1 | instance_idx)")
run_generator_test(e4, rlang::expr(condition != "Baseline"))
run_random_slopes(e4, "correct ~ condition * judge + (condition | instance_idx)")

# ==============================================================================
# 8. E4-Specific: Instance difficulty + Trial order
# ==============================================================================
cat("\n--- 8. E4-Specific Analyses ---\n")

# Instance difficulty covariate (absolute prediction error)
if ("abs_error" %in% names(e4) | all(c("predicted_value", "true_value") %in% names(e4))) {
  if (!"abs_error" %in% names(e4)) {
    e4$abs_error <- abs(as.numeric(e4$predicted_value) - as.numeric(e4$true_value))
  }
  cat("\nModel with difficulty covariate (abs_error):\n")
  tryCatch({
    m4_diff <- glmer(correct ~ condition * judge + scale(abs_error) + (1 | instance_idx),
                     data = e4, family = binomial,
                     control = glmerControl(optimizer = "bobyqa"))
    print(summary(m4_diff))
  }, error = function(e) cat(sprintf("Error: %s\n", e$message)))
}

# Trial order (test position)
cat("\nTrial order analysis:\n")
e4$trial_order <- as.numeric(as.character(e4$instance_idx))
tryCatch({
  m4_order <- glmer(correct ~ condition * judge + scale(trial_order) + (1 | instance_idx),
                    data = e4, family = binomial,
                    control = glmerControl(optimizer = "bobyqa"))
  cat("Trial order coefficient:\n")
  print(round(summary(m4_order)$coefficients, 4))
}, error = function(e) cat(sprintf("Error: %s\n", e$message)))

# Condition x trial_order interaction
cat("\nCondition x trial_order interaction:\n")
tryCatch({
  m4_order_int <- glmer(correct ~ condition * scale(trial_order) + judge + (1 | instance_idx),
                        data = e4, family = binomial,
                        control = glmerControl(optimizer = "bobyqa"))
  print(round(summary(m4_order_int)$coefficients, 4))
}, error = function(e) cat(sprintf("Error: %s\n", e$message)))

# true_bucket robustness
if ("true_bucket" %in% names(e4)) {
  cat("\ntrue_bucket robustness:\n")
  e4$true_bucket <- factor(e4$true_bucket)
  tryCatch({
    m4_bucket <- glmer(correct ~ condition * judge + true_bucket + (1 | instance_idx),
                       data = e4, family = binomial,
                       control = glmerControl(optimizer = "bobyqa"))
    print(summary(m4_bucket))
  }, error = function(e) cat(sprintf("Error (likely separation): %s\n", e$message)))
}

cat("\nNote: E4 uses a sliding window with 80% overlap between adjacent training sets.\n")
cat("This induces temporal autocorrelation beyond the instance random intercept.\n")
cat("Inferential claims should be interpreted with this caveat.\n")

cat("\n================================================================\n")
cat("  E4 COMPLETE\n")
cat("================================================================\n")
