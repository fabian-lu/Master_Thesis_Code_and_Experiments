# ==============================================================================
# 02_E2_anomaly.R — Experiment 2: Anomaly Detection (Selective Reliance)
# ==============================================================================
# Research questions:
#   1. Is there a poisoning x NLE interaction on accuracy?
#      (Does NLE help for baseline but hurt for OOD?)
#   2. Does NLE presence affect confidence differently for poisoned vs clean?
#
# Design: 2x2 factorial (poisoning: baseline/ood × has_nle: True/False)
#         60 instances, 720 rows
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  EXPERIMENT 2: ANOMALY DETECTION\n")
cat("================================================================\n")

# ==============================================================================
# 1. Descriptive Statistics
# ==============================================================================

cat("\n--- Descriptive Statistics ---\n")

desc_e2 <- e2 %>%
  group_by(poisoning, has_nle) %>%
  summarise(
    n         = n(),
    accuracy  = mean(correct, na.rm = TRUE),
    mean_conf = mean(confidence_num, na.rm = TRUE),
    med_conf  = median(confidence_num, na.rm = TRUE),
    .groups   = "drop"
  )
print(desc_e2)

cat("\n--- By Poisoning × NLE × Judge ---\n")
desc_e2_judge <- e2 %>%
  group_by(poisoning, has_nle, judge) %>%
  summarise(
    n        = n(),
    accuracy = mean(correct, na.rm = TRUE),
    mean_conf = mean(confidence_num, na.rm = TRUE),
    .groups  = "drop"
  )
print(desc_e2_judge)

cat("\n--- Overconfidence Descriptive ---\n")
overconf_e2 <- e2 %>%
  mutate(correctness = ifelse(correct == 1, "correct", "incorrect")) %>%
  group_by(poisoning, has_nle, correctness) %>%
  summarise(
    n         = n(),
    mean_conf = mean(confidence_num, na.rm = TRUE),
    .groups   = "drop"
  )
print(overconf_e2)

# ==============================================================================
# 2. Primary Analysis: Accuracy (GLMM with interaction)
# ==============================================================================

cat("\n--- Primary GLMM: Accuracy (poisoning × has_nle interaction) ---\n")

m2_full <- glmer(correct ~ poisoning * has_nle + judge + (1 | instance_idx),
                 data = e2, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))

m2_no_int <- glmer(correct ~ poisoning + has_nle + judge + (1 | instance_idx),
                   data = e2, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

m2_reduced <- glmer(correct ~ judge + (1 | instance_idx),
                    data = e2, family = binomial,
                    control = glmerControl(optimizer = "bobyqa"))

# Test interaction
cat("\nLRT: Interaction (poisoning × has_nle):\n")
lrt_interaction <- anova(m2_no_int, m2_full, test = "Chisq")
print(lrt_interaction)

# Test main effects
cat("\nLRT: Main effects (poisoning + has_nle vs null):\n")
lrt_main <- anova(m2_reduced, m2_no_int, test = "Chisq")
print(lrt_main)

# Full model summary
cat("\nFull interaction model summary:\n")
print(summary(m2_full))

# Odds ratios
cat("\nOdds Ratios:\n")
print(extract_ors(m2_full))

# ==============================================================================
# 3. Simple Effects (NLE effect within each poisoning level)
# ==============================================================================

cat("\n--- Simple Effects ---\n")

# NLE effect within each poisoning level
cat("\nNLE effect within each poisoning level:\n")
emm2_nle <- emmeans(m2_full, ~ has_nle | poisoning, type = "response")
print(summary(pairs(emm2_nle, adjust = "holm")))

# Poisoning effect within each NLE level
cat("\nPoisoning effect within each NLE level:\n")
emm2_pois <- emmeans(m2_full, ~ poisoning | has_nle, type = "response")
print(summary(pairs(emm2_pois, adjust = "holm")))

# Marginal means
cat("\nEstimated marginal means:\n")
emm2_all <- emmeans(m2_full, ~ poisoning * has_nle, type = "response")
print(summary(emm2_all))

# ==============================================================================
# 4. Confidence Analysis (CLMM with interaction)
# ==============================================================================

cat("\n--- Confidence Analysis (CLMM) ---\n")

c2_full <- clmm(confidence ~ poisoning * has_nle + judge + (1 | instance_idx),
                data = e2)

c2_no_int <- clmm(confidence ~ poisoning + has_nle + judge + (1 | instance_idx),
                  data = e2)

cat("\nLRT: Interaction on confidence:\n")
lrt_conf_int <- anova(c2_no_int, c2_full)
print(lrt_conf_int)

cat("\nCLMM summary:\n")
print(summary(c2_full))

# Confidence simple effects
cat("\nConfidence: NLE effect within each poisoning level:\n")
emm2c <- emmeans(c2_full, ~ has_nle | poisoning)
print(summary(pairs(emm2c, adjust = "holm")))

# ==============================================================================
# 5. Diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")
run_glmm_diagnostics(m2_full, "E2 accuracy GLMM")

cat("\n================================================================\n")
cat("  E2 ANALYSIS COMPLETE\n")
cat("================================================================\n")
