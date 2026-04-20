# ==============================================================================
# 03_E3_counterfactual.R — Experiment 3: Counterfactual Simulatability
# ==============================================================================
# RQ-U3: Do NLEs help predict direction of model output change after perturbation?
# Design: 2 conditions (X, E+X), 60 instances, N=360
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/new_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 3: COUNTERFACTUAL SIMULATABILITY\n")
cat("================================================================\n")

e3$condition <- factor(e3$condition)

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
print(descriptives(e3))
print(descriptives_by_judge(e3))

cat("\nTrue direction distribution:\n")
print(table(e3$true_direction))

# Per-class confusion matrices
cat("\nConfusion matrix (SHAP only, X condition):\n")
print(table(Predicted = e3$predicted_direction[e3$condition == "X"],
            True = e3$true_direction[e3$condition == "X"]))

cat("\nConfusion matrix (NLE + SHAP, E+X condition):\n")
print(table(Predicted = e3$predicted_direction[e3$condition == "E+X"],
            True = e3$true_direction[e3$condition == "E+X"]))

# ==============================================================================
# 2. Primary GLMM: Accuracy (condition * judge)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

acc3 <- fit_accuracy_model(e3)

cat("\nModel summary:\n")
print(summary(acc3$model))

cat("\nMarginal EMMs:\n")
print(summary(acc3$emm_cond))

cat("\nJudge-specific EMMs:\n")
print(summary(acc3$emm_judge))

# Only 2 levels — condition coefficient IS the test
cat("\nCondition OR and CI:\n")
fe <- fixef(acc3$model)
se <- sqrt(diag(vcov(acc3$model)))
cond_idx <- grep("condition", names(fe))
if (length(cond_idx) > 0) {
  or <- exp(fe[cond_idx])
  ci_lo <- exp(fe[cond_idx] - 1.96 * se[cond_idx])
  ci_hi <- exp(fe[cond_idx] + 1.96 * se[cond_idx])
  cat(sprintf("  OR = %.3f [%.3f, %.3f]\n", or, ci_lo, ci_hi))
}

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

conf3 <- fit_confidence_model(e3)
cat("\nConfidence LRT:\n")
print(conf3$lrt)
cat("\nCLMM summary:\n")
print(summary(conf3$model))

# ==============================================================================
# 4. Calibration
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
cal3 <- run_calibration(e3)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(acc3$model, "E3 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm3 <- run_bayesian_rope(e3)

# ==============================================================================
# 7. Sensitivity
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

run_judge_specific(e3, "correct ~ condition + (1 | instance_idx)")
run_generator_test(e3, rlang::expr(condition == "E+X"))
run_random_slopes(e3, "correct ~ condition * judge + (condition | instance_idx)")

# ==============================================================================
# 8. E3-Specific: true_direction covariate
# ==============================================================================
cat("\n--- 8. true_direction covariate ---\n")
tryCatch({
  e3$true_direction <- factor(e3$true_direction)
  m3_dir <- glmer(correct ~ condition * judge + true_direction + (1 | instance_idx),
                  data = e3, family = binomial,
                  control = glmerControl(optimizer = "bobyqa"))
  cat("Model with true_direction:\n")
  print(summary(m3_dir))
}, error = function(e) cat(sprintf("Error: %s\n", e$message)))

# Per-class recall
cat("\nPer-class recall by condition:\n")
class_recall <- e3 %>%
  group_by(condition, true_direction) %>%
  summarise(recall = mean(correct), n = n(), .groups = "drop")
print(as.data.frame(class_recall))

cat("\n================================================================\n")
cat("  E3 COMPLETE\n")
cat("================================================================\n")
