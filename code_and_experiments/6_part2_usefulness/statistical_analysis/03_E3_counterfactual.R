# ==============================================================================
# 03_E3_counterfactual.R — Experiment 3: Counterfactual Simulatability
# ==============================================================================
# Research questions:
#   1. Does adding NLEs to XAI values improve directional prediction accuracy?
#   2. Is there a confidence effect?
#
# Design: 2 conditions (X, E+X), 60 instances, 360 rows
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 3: COUNTERFACTUAL SIMULATABILITY\n")
cat("================================================================\n")

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================

cat("\n--- Descriptive Statistics ---\n")
desc_e3 <- descriptive_by_condition(e3)
print(desc_e3)

cat("\n--- By Condition × Judge ---\n")
print(descriptive_by_condition_judge(e3))

cat("\n--- Overconfidence Descriptive ---\n")
print(overconfidence_descriptive(e3))

cat("\n--- True Direction Distribution ---\n")
print(table(e3$true_direction))

# ==============================================================================
# 2. Primary Analysis: Accuracy (GLMM)
# ==============================================================================

cat("\n--- Primary GLMM: Accuracy ---\n")

m3 <- glmer(correct ~ condition + judge + (1 | instance_idx),
            data = e3, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))

cat("\nModel summary (condition coefficient = E+X vs X test):\n")
print(summary(m3))

# Odds ratios
cat("\nOdds Ratios:\n")
print(extract_ors(m3))

# Marginal means
emm3 <- emmeans(m3, ~ condition, type = "response")
cat("\nEstimated marginal means (probability scale):\n")
print(summary(emm3))

cat("\nContrast E+X vs X:\n")
print(summary(pairs(emm3)))

# ==============================================================================
# 3. Confidence Analysis (CLMM)
# ==============================================================================

cat("\n--- Confidence Analysis (CLMM) ---\n")

c3 <- clmm(confidence ~ condition + judge + (1 | instance_idx),
           data = e3)

cat("\nCLMM summary:\n")
print(summary(c3))

emm3c <- emmeans(c3, ~ condition)
cat("\nConfidence contrast E+X vs X:\n")
print(summary(pairs(emm3c)))

# ==============================================================================
# 4. Overconfidence Analysis
# ==============================================================================

cat("\n--- Overconfidence Analysis ---\n")

e3$correctness <- factor(ifelse(e3$correct == 1, "correct", "incorrect"),
                         levels = c("correct", "incorrect"))

c3_overconf <- clmm(confidence ~ correctness * condition + judge + (1 | instance_idx),
                    data = e3)
cat("\nOverconfidence interaction model:\n")
print(summary(c3_overconf))

# ==============================================================================
# 5. Diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")
run_glmm_diagnostics(m3, "E3 accuracy GLMM")

cat("\n================================================================\n")
cat("  E3 ANALYSIS COMPLETE\n")
cat("================================================================\n")
