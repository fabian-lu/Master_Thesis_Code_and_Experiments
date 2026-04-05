# ==============================================================================
# 07_descriptive.R — Descriptive statistics and plots for all experiments
# ==============================================================================
# Generates:
#   1. Summary tables (accuracy, confidence per condition)
#   2. Accuracy bar plots per experiment
#   3. Confidence distribution plots
#   4. Overconfidence (calibration) plots
#   5. Cross-experiment summary table
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

library(patchwork)
library(scales)

cat("\n")
cat("================================================================\n")
cat("  DESCRIPTIVE STATISTICS & PLOTS\n")
cat("================================================================\n")

PLOT_DIR <- file.path(OUT_DIR, "plots")
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# 1. Summary Tables
# ==============================================================================

cat("\n=== SUMMARY TABLES ===\n")

# E1
cat("\n--- E1: Closeness Task ---\n")
e1_summary <- descriptive_by_condition(e1)
print(e1_summary)

# E2
cat("\n--- E2: Anomaly Detection ---\n")
e2_summary <- e2 %>%
  group_by(poisoning, has_nle) %>%
  summarise(
    n = n(), accuracy = mean(correct), mean_conf = mean(confidence_num),
    .groups = "drop"
  )
print(e2_summary)

# E3
cat("\n--- E3: Counterfactual ---\n")
e3_summary <- descriptive_by_condition(e3)
print(e3_summary)

# E4
cat("\n--- E4: Mental Model Transfer ---\n")
e4_summary <- descriptive_by_condition(e4)
print(e4_summary)

# E5
cat("\n--- E5: Placebic NLE ---\n")
e5_summary <- descriptive_by_condition(e5)
print(e5_summary)

# Cross-experiment
cat("\n--- Cross-Experiment Summary ---\n")
cross_summary <- bind_rows(
  e1_summary %>% mutate(experiment = "E1"),
  e3_summary %>% mutate(experiment = "E3"),
  e4_summary %>% mutate(experiment = "E4"),
  e5_summary %>% mutate(experiment = "E5")
)
print(cross_summary %>% select(experiment, condition, n, accuracy, mean_conf))

# E2 separate (different structure)
cat("\nE2 (2x2):\n")
print(e2_summary)

# ==============================================================================
# 2. Accuracy Bar Plots
# ==============================================================================

cat("\n=== GENERATING PLOTS ===\n")

theme_paper <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

# E1 accuracy
p_e1_acc <- ggplot(e1_summary, aes(x = condition, y = accuracy, fill = condition)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "E1: Closeness Task — Accuracy by Condition",
       x = NULL, y = "Accuracy") +
  theme_paper +
  theme(legend.position = "none")

# E2 accuracy (grouped bar)
p_e2_acc <- ggplot(e2_summary, aes(x = poisoning, y = accuracy, fill = has_nle)) +
  geom_col(position = position_dodge(0.7), width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)),
            position = position_dodge(0.7), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_fill_manual(values = c("False" = "#999999", "True" = "#4DAF4A"),
                    labels = c("No NLE", "With NLE")) +
  labs(title = "E2: Anomaly Detection — Accuracy by Poisoning × NLE",
       x = "Poisoning Level", y = "Accuracy", fill = NULL) +
  theme_paper

# E3 accuracy
p_e3_acc <- ggplot(e3_summary, aes(x = condition, y = accuracy, fill = condition)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "E3: Counterfactual — Accuracy by Condition",
       x = NULL, y = "Accuracy") +
  theme_paper +
  theme(legend.position = "none")

# E4 accuracy
p_e4_acc <- ggplot(e4_summary, aes(x = condition, y = accuracy, fill = condition)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "E4: Transfer — Accuracy by Condition",
       x = NULL, y = "Accuracy") +
  theme_paper +
  theme(legend.position = "none")

# E5 accuracy
p_e5_acc <- ggplot(e5_summary, aes(x = condition, y = accuracy, fill = condition)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.1f%%", accuracy * 100)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(title = "E5: Placebo — Accuracy by Condition",
       x = NULL, y = "Accuracy") +
  theme_paper +
  theme(legend.position = "none")

# Save accuracy plots
ggsave(file.path(PLOT_DIR, "e1_accuracy.pdf"), p_e1_acc, width = 7, height = 4)
ggsave(file.path(PLOT_DIR, "e2_accuracy.pdf"), p_e2_acc, width = 6, height = 4)
ggsave(file.path(PLOT_DIR, "e3_accuracy.pdf"), p_e3_acc, width = 5, height = 4)
ggsave(file.path(PLOT_DIR, "e4_accuracy.pdf"), p_e4_acc, width = 5, height = 4)
ggsave(file.path(PLOT_DIR, "e5_accuracy.pdf"), p_e5_acc, width = 6, height = 4)

cat("Accuracy plots saved.\n")

# ==============================================================================
# 3. Confidence Distribution Plots
# ==============================================================================

# E1 confidence violin
p_e1_conf <- ggplot(e1, aes(x = condition, y = confidence_num, fill = condition)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.15, outlier.shape = NA) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
  labs(title = "E1: Confidence Distribution by Condition",
       x = NULL, y = "Confidence (1-5)") +
  theme_paper +
  theme(legend.position = "none")

# E5 confidence violin
p_e5_conf <- ggplot(e5, aes(x = condition, y = confidence_num, fill = condition)) +
  geom_violin(alpha = 0.6) +
  geom_boxplot(width = 0.15, outlier.shape = NA) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
  labs(title = "E5: Confidence Distribution by Condition",
       x = NULL, y = "Confidence (1-5)") +
  theme_paper +
  theme(legend.position = "none")

ggsave(file.path(PLOT_DIR, "e1_confidence.pdf"), p_e1_conf, width = 7, height = 4)
ggsave(file.path(PLOT_DIR, "e5_confidence.pdf"), p_e5_conf, width = 6, height = 4)

cat("Confidence plots saved.\n")

# ==============================================================================
# 4. Overconfidence (Calibration) Plots
# ==============================================================================

# Function: confidence by correctness across conditions
plot_calibration <- function(df, title_str, conditions_var = "condition") {
  df %>%
    mutate(correctness = ifelse(correct == 1, "Correct", "Incorrect")) %>%
    group_by(across(all_of(conditions_var)), correctness) %>%
    summarise(mean_conf = mean(confidence_num), .groups = "drop") %>%
    ggplot(aes(x = .data[[conditions_var]], y = mean_conf,
               fill = correctness, group = correctness)) +
    geom_col(position = position_dodge(0.7), width = 0.6) +
    scale_fill_manual(values = c("Correct" = "#4DAF4A", "Incorrect" = "#E41A1C")) +
    labs(title = title_str,
         x = NULL, y = "Mean Confidence", fill = NULL) +
    theme_paper
}

p_e1_calib <- plot_calibration(e1, "E1: Confidence When Correct vs Incorrect")
p_e5_calib <- plot_calibration(e5, "E5: Confidence When Correct vs Incorrect")

ggsave(file.path(PLOT_DIR, "e1_calibration.pdf"), p_e1_calib, width = 7, height = 4)
ggsave(file.path(PLOT_DIR, "e5_calibration.pdf"), p_e5_calib, width = 6, height = 4)

cat("Calibration plots saved.\n")

# ==============================================================================
# 5. Combined Overview Plot
# ==============================================================================

# Accuracy across all experiments (one panel each)
all_acc <- bind_rows(
  e1 %>% group_by(condition) %>%
    summarise(accuracy = mean(correct), .groups = "drop") %>%
    mutate(experiment = "E1: Closeness"),
  e2 %>% mutate(condition = paste0(poisoning, "/", has_nle)) %>%
    group_by(condition) %>%
    summarise(accuracy = mean(correct), .groups = "drop") %>%
    mutate(experiment = "E2: Anomaly"),
  e3 %>% group_by(condition) %>%
    summarise(accuracy = mean(correct), .groups = "drop") %>%
    mutate(experiment = "E3: Counterfactual"),
  e4 %>% group_by(condition) %>%
    summarise(accuracy = mean(correct), .groups = "drop") %>%
    mutate(experiment = "E4: Transfer"),
  e5 %>% group_by(condition) %>%
    summarise(accuracy = mean(correct), .groups = "drop") %>%
    mutate(experiment = "E5: Placebo")
)

p_overview <- ggplot(all_acc, aes(x = condition, y = accuracy)) +
  geom_col(fill = "#377EB8", width = 0.6) +
  geom_text(aes(label = sprintf("%.0f%%", accuracy * 100)), vjust = -0.3, size = 2.8) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  facet_wrap(~ experiment, scales = "free_x", nrow = 1) +
  labs(title = "Accuracy Across All Experiments",
       x = NULL, y = "Accuracy") +
  theme_paper +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        strip.text = element_text(face = "bold"))

ggsave(file.path(PLOT_DIR, "all_experiments_accuracy.pdf"), p_overview,
       width = 14, height = 4)

cat("Overview plot saved.\n")

# ==============================================================================
# 6. Export Summary CSV
# ==============================================================================

# Write cross-experiment summary to CSV
summary_path <- file.path(OUT_DIR, "cross_experiment_summary.csv")
write.csv(all_acc, summary_path, row.names = FALSE)
cat(sprintf("\nSummary CSV saved to %s\n", summary_path))

cat("\n================================================================\n")
cat("  DESCRIPTIVE STATISTICS & PLOTS COMPLETE\n")
cat(sprintf("  All plots saved to: %s\n", PLOT_DIR))
cat("================================================================\n")
