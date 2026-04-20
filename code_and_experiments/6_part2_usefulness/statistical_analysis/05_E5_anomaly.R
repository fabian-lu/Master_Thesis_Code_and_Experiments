# ==============================================================================
# 05_E5_anomaly.R — Experiment 5: Selective Reliance (Anomaly Detection)
# ==============================================================================
# RQ-U5: Do NLEs help detect unreliable predictions from OOD inputs?
# Design: 2x2 factorial (poisoning x NLE), 60 instances, N=720
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/statistical_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 5: SELECTIVE RELIANCE\n")
cat("================================================================\n")

e5$poisoning <- factor(e5$poisoning, levels = c("baseline", "ood"))
e5$has_nle   <- factor(e5$has_nle)

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
desc <- e5 %>%
  group_by(poisoning, has_nle) %>%
  summarise(n = n(), accuracy = mean(correct), mean_conf = mean(confidence_num), .groups = "drop")
print(as.data.frame(desc))
print(descriptives_by_judge(e5 %>% mutate(condition = paste(poisoning, has_nle, sep = "_"))))

# ==============================================================================
# 2. Primary GLMM: poisoning * has_nle * judge (three-way)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

# Three-way model (to check judge asymmetry)
m5_3way <- glmer(correct ~ poisoning * has_nle * judge + (1 | instance_idx),
                 data = e5, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

# Two-way model (primary)
m5_full <- glmer(correct ~ poisoning * has_nle + judge + (1 | instance_idx),
                 data = e5, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

m5_no_int <- glmer(correct ~ poisoning + has_nle + judge + (1 | instance_idx),
                   data = e5, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

m5_red <- glmer(correct ~ judge + (1 | instance_idx),
                data = e5, family = binomial,
                control = glmerControl(optimizer = "bobyqa"))

# Three-way interaction test
cat("\nThree-way interaction (poisoning * has_nle * judge):\n")
print(anova(m5_full, m5_3way, test = "Chisq"))

# Critical: poisoning x NLE interaction
cat("\nInteraction LRT (poisoning x has_nle):\n")
print(anova(m5_no_int, m5_full, test = "Chisq"))

cat("\nFull model summary:\n")
print(summary(m5_full))

# Simple effects: NLE effect within each poisoning level
cat("\nSimple effects — NLE effect within Baseline:\n")
emm5 <- emmeans(m5_full, ~ has_nle | poisoning, type = "response")
print(pairs(emm5, adjust = "holm"))

cat("\nSimple effects — Poisoning effect within each NLE level:\n")
emm5b <- emmeans(m5_full, ~ poisoning | has_nle, type = "response")
print(pairs(emm5b, adjust = "holm"))

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

c2_full <- clmm(confidence ~ poisoning * has_nle + judge + (1 | instance_idx), data = e5)
c2_no_int <- clmm(confidence ~ poisoning + has_nle + judge + (1 | instance_idx), data = e5)

cat("\nConfidence interaction LRT:\n")
print(anova(c2_no_int, c2_full))
cat("\nCLMM summary:\n")
print(summary(c2_full))

# ==============================================================================
# 4. Calibration Analysis
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
e5$condition <- paste(e5$poisoning, e5$has_nle, sep = "_")
cal5 <- run_calibration(e5)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(m5_full, "E5 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm5 <- run_bayesian_rope(e5,
  formula_str = "correct ~ poisoning * has_nle + judge + (1 | instance_idx)")

# ==============================================================================
# 7. Sensitivity
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

# Judge-specific
run_judge_specific(e5, "correct ~ poisoning * has_nle + (1 | instance_idx)")

# Generator effects (NLE conditions)
run_generator_test(e5, rlang::expr(has_nle == TRUE))

# Random slopes
run_random_slopes(e5, "correct ~ poisoning * has_nle + judge + (has_nle | instance_idx)")

cat("\n================================================================\n")
cat("  E5 COMPLETE\n")
cat("================================================================\n")
