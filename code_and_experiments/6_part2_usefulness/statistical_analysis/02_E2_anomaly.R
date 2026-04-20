# ==============================================================================
# 02_E2_anomaly.R — Experiment 2: Selective Reliance (Anomaly Detection)
# ==============================================================================
# RQ-U2: Do NLEs help detect unreliable predictions from OOD inputs?
# Design: 2x2 factorial (poisoning x NLE), 60 instances, N=720
# ==============================================================================

source("/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/new_analysis/00_setup.R")

cat("\n================================================================\n")
cat("  EXPERIMENT 2: SELECTIVE RELIANCE\n")
cat("================================================================\n")

e2$poisoning <- factor(e2$poisoning, levels = c("baseline", "ood"))
e2$has_nle   <- factor(e2$has_nle)

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================
cat("\n--- 1. Descriptive Statistics ---\n")
desc <- e2 %>%
  group_by(poisoning, has_nle) %>%
  summarise(n = n(), accuracy = mean(correct), mean_conf = mean(confidence_num), .groups = "drop")
print(as.data.frame(desc))
print(descriptives_by_judge(e2 %>% mutate(condition = paste(poisoning, has_nle, sep = "_"))))

# ==============================================================================
# 2. Primary GLMM: poisoning * has_nle * judge (three-way)
# ==============================================================================
cat("\n--- 2. Primary GLMM: Accuracy ---\n")

# Three-way model (to check judge asymmetry)
m2_3way <- glmer(correct ~ poisoning * has_nle * judge + (1 | instance_idx),
                 data = e2, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

# Two-way model (primary)
m2_full <- glmer(correct ~ poisoning * has_nle + judge + (1 | instance_idx),
                 data = e2, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

m2_no_int <- glmer(correct ~ poisoning + has_nle + judge + (1 | instance_idx),
                   data = e2, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

m2_red <- glmer(correct ~ judge + (1 | instance_idx),
                data = e2, family = binomial,
                control = glmerControl(optimizer = "bobyqa"))

# Three-way interaction test
cat("\nThree-way interaction (poisoning * has_nle * judge):\n")
print(anova(m2_full, m2_3way, test = "Chisq"))

# Critical: poisoning x NLE interaction
cat("\nInteraction LRT (poisoning x has_nle):\n")
print(anova(m2_no_int, m2_full, test = "Chisq"))

cat("\nFull model summary:\n")
print(summary(m2_full))

# Simple effects: NLE effect within each poisoning level
cat("\nSimple effects — NLE effect within Baseline:\n")
emm2 <- emmeans(m2_full, ~ has_nle | poisoning, type = "response")
print(pairs(emm2, adjust = "holm"))

cat("\nSimple effects — Poisoning effect within each NLE level:\n")
emm2b <- emmeans(m2_full, ~ poisoning | has_nle, type = "response")
print(pairs(emm2b, adjust = "holm"))

# ==============================================================================
# 3. Confidence (CLMM)
# ==============================================================================
cat("\n--- 3. Confidence (CLMM) ---\n")

c2_full <- clmm(confidence ~ poisoning * has_nle + judge + (1 | instance_idx), data = e2)
c2_no_int <- clmm(confidence ~ poisoning + has_nle + judge + (1 | instance_idx), data = e2)

cat("\nConfidence interaction LRT:\n")
print(anova(c2_no_int, c2_full))
cat("\nCLMM summary:\n")
print(summary(c2_full))

# ==============================================================================
# 4. Calibration Analysis
# ==============================================================================
cat("\n--- 4. Calibration ---\n")
e2$condition <- paste(e2$poisoning, e2$has_nle, sep = "_")
cal2 <- run_calibration(e2)

# ==============================================================================
# 5. Diagnostics
# ==============================================================================
cat("\n--- 5. Diagnostics ---\n")
run_diagnostics(m2_full, "E2 accuracy GLMM")

# ==============================================================================
# 6. Bayesian + ROPE
# ==============================================================================
cat("\n--- 6. Bayesian + ROPE ---\n")
bm2 <- run_bayesian_rope(e2,
  formula_str = "correct ~ poisoning * has_nle + judge + (1 | instance_idx)")

# ==============================================================================
# 7. Sensitivity
# ==============================================================================
cat("\n--- 7. Sensitivity ---\n")

# Judge-specific
run_judge_specific(e2, "correct ~ poisoning * has_nle + (1 | instance_idx)")

# Generator effects (NLE conditions)
run_generator_test(e2, rlang::expr(has_nle == TRUE))

# Random slopes
run_random_slopes(e2, "correct ~ poisoning * has_nle + judge + (has_nle | instance_idx)")

cat("\n================================================================\n")
cat("  E2 COMPLETE\n")
cat("================================================================\n")
