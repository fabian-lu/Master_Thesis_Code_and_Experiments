# ==============================================================================
# 04_E4_transfer.R — Experiment 4: Mental Model Transfer
# ==============================================================================
# Research questions:
#   1. Does training with NLEs improve transfer accuracy (tested WITHOUT NLE)?
#   2. Confidence effect?
#
# Design: 2 conditions (Baseline, E), 55 instances (idx 5-59), 330 rows
#         Sliding window: first 5 instances used as training context
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 4: MENTAL MODEL TRANSFER\n")
cat("================================================================\n")

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================

cat("\n--- Descriptive Statistics ---\n")
desc_e4 <- descriptive_by_condition(e4)
print(desc_e4)

cat("\n--- By Condition × Judge ---\n")
print(descriptive_by_condition_judge(e4))

cat("\n--- Overconfidence Descriptive ---\n")
print(overconfidence_descriptive(e4))

cat("\n--- True Bucket Distribution ---\n")
print(table(e4$true_bucket))

cat("\n--- Trial Order Range ---\n")
cat(sprintf("trial_order range: %d to %d\n", min(e4$trial_order), max(e4$trial_order)))

# ==============================================================================
# 2. Primary Analysis: Accuracy (GLMM)
# ==============================================================================

cat("\n--- Primary GLMM: Accuracy ---\n")

m4 <- glmer(correct ~ condition + judge + (1 | instance_idx),
            data = e4, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))

cat("\nModel summary (condition coefficient = E vs Baseline test):\n")
print(summary(m4))

# Odds ratios
cat("\nOdds Ratios:\n")
print(extract_ors(m4))

# Marginal means
emm4 <- emmeans(m4, ~ condition, type = "response")
cat("\nEstimated marginal means (probability scale):\n")
print(summary(emm4))

cat("\nContrast E vs Baseline:\n")
print(summary(pairs(emm4)))

# ==============================================================================
# 3. Confidence Analysis (CLMM)
# ==============================================================================

cat("\n--- Confidence Analysis (CLMM) ---\n")

c4 <- clmm(confidence ~ condition + judge + (1 | instance_idx),
           data = e4)

cat("\nCLMM summary:\n")
print(summary(c4))

emm4c <- emmeans(c4, ~ condition)
cat("\nConfidence contrast E vs Baseline:\n")
print(summary(pairs(emm4c)))

# ==============================================================================
# 4. Sensitivity: Trial Order (Learning Over Time)
# ==============================================================================

cat("\n--- Sensitivity: Trial Order ---\n")

m4_order <- glmer(correct ~ condition + scale(trial_order) + judge + (1 | instance_idx),
                  data = e4, family = binomial,
                  control = glmerControl(optimizer = "bobyqa"))

cat("\nModel with trial_order covariate:\n")
print(summary(m4_order))

cat("\nCompare: with vs without trial_order:\n")
print(anova(m4, m4_order, test = "Chisq"))

# Check if trial_order interacts with condition (learning rate differs?)
m4_order_int <- glmer(correct ~ condition * scale(trial_order) + judge + (1 | instance_idx),
                      data = e4, family = binomial,
                      control = glmerControl(optimizer = "bobyqa"))

cat("\nCondition × trial_order interaction:\n")
print(anova(m4_order, m4_order_int, test = "Chisq"))

# ==============================================================================
# 5. Robustness: Control for true_bucket
# ==============================================================================

cat("\n--- Robustness: Controlling for true_bucket ---\n")

e4$true_bucket <- factor(e4$true_bucket,
                         levels = c("small", "medium", "large", "very_large"))

m4_bucket <- glmer(correct ~ condition + true_bucket + judge + (1 | instance_idx),
                   data = e4, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

cat("\nModel with true_bucket covariate:\n")
print(summary(m4_bucket))

# ==============================================================================
# 6. Diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")
run_glmm_diagnostics(m4, "E4 accuracy GLMM")

cat("\n================================================================\n")
cat("  E4 ANALYSIS COMPLETE\n")
cat("================================================================\n")
