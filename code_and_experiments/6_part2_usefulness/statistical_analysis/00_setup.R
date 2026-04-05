# ==============================================================================
# 00_setup.R — Shared setup, data loading, and helper functions
# ==============================================================================
# Run this file first. It loads all packages, reads all 5 experiment CSVs,
# prepares factor variables, and defines reusable helper functions.
# ==============================================================================

# --- Install packages (uncomment if needed) ---
# install.packages(c(
#   "lme4", "ordinal", "emmeans", "lmerTest",
#   "performance", "DHARMa", "broom.mixed", "ggeffects",
#   "tidyverse", "brms", "ggplot2", "patchwork"
# ))

# --- Load packages ---
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(ordinal)
library(emmeans)
library(performance)
library(DHARMa)
library(broom.mixed)

# --- Paths (absolute) ---
SCRIPT_DIR <- "/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis"
BASE_DIR   <- "/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/results"
OUT_DIR    <- file.path(SCRIPT_DIR, "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Data Loading
# ==============================================================================

load_experiment <- function(exp_name, base_dir = BASE_DIR) {
  path <- file.path(base_dir, exp_name, "full_results.csv")
  df <- read.csv(path, stringsAsFactors = FALSE)

  # Common factor conversions
  df$instance_idx <- factor(df$instance_idx)
  df$judge        <- factor(df$judge)
  df$generator    <- factor(df$generator)
  df$correct      <- as.integer(df$correct)
  df$confidence   <- ordered(df$confidence, levels = 1:5)
  df$confidence_num <- as.numeric(as.character(df$confidence))

  return(df)
}

cat("Loading experiment data...\n")

e1 <- load_experiment("E1") %>%
  mutate(condition = factor(condition,
                            levels = c("Baseline", "X", "T", "X+T", "E+X+T")))

e2 <- load_experiment("E2") %>%
  mutate(
    poisoning = factor(poisoning, levels = c("baseline", "ood")),
    has_nle   = factor(has_nle, levels = c("False", "True"))
  )

e3 <- load_experiment("E3") %>%
  mutate(condition = factor(condition, levels = c("X", "E+X")))

e4 <- load_experiment("E4") %>%
  mutate(
    condition   = factor(condition, levels = c("Baseline", "E")),
    trial_order = as.numeric(as.character(instance_idx)) - 5
  )

e5 <- load_experiment("E5") %>%
  mutate(condition = factor(condition,
                            levels = c("Baseline", "Real_NLE", "Placebo_NLE")))

cat("Data loaded successfully.\n")
cat(sprintf("  E1: %d rows, %d instances, %d conditions\n",
            nrow(e1), nlevels(e1$instance_idx), nlevels(e1$condition)))
cat(sprintf("  E2: %d rows, %d instances\n",
            nrow(e2), nlevels(e2$instance_idx)))
cat(sprintf("  E3: %d rows, %d instances, %d conditions\n",
            nrow(e3), nlevels(e3$instance_idx), nlevels(e3$condition)))
cat(sprintf("  E4: %d rows, %d instances, %d conditions\n",
            nrow(e4), nlevels(e4$instance_idx), nlevels(e4$condition)))
cat(sprintf("  E5: %d rows, %d instances, %d conditions\n",
            nrow(e5), nlevels(e5$instance_idx), nlevels(e5$condition)))

# ==============================================================================
# Helper Functions
# ==============================================================================

#' Run GLMM diagnostics and print summary
run_glmm_diagnostics <- function(model, name = "model") {
  cat(sprintf("\n=== Diagnostics for %s ===\n", name))

  # Convergence
  conv <- check_convergence(model)
  cat(sprintf("Convergence: %s\n", ifelse(conv, "OK", "WARNING")))

  # Singularity
  sing <- check_singularity(model)
  cat(sprintf("Singularity: %s\n", ifelse(sing, "WARNING - singular fit", "OK")))

  # Collinearity (only if >1 fixed effect)
  tryCatch({
    coll <- check_collinearity(model)
    print(coll)
  }, error = function(e) {
    cat("Collinearity check skipped (single predictor or other issue)\n")
  })

  # DHARMa residuals
  tryCatch({
    sim_res <- simulateResiduals(model, n = 1000)
    cat("\nDHARMa uniformity test:\n")
    print(testUniformity(sim_res))
    cat("DHARMa dispersion test:\n")
    print(testDispersion(sim_res))
  }, error = function(e) {
    cat(sprintf("DHARMa diagnostics failed: %s\n", e$message))
  })
}

#' Extract odds ratios with CIs from a GLMM
extract_ors <- function(model) {
  cc <- confint(model, parm = "beta_", method = "Wald")
  fe <- fixef(model)
  data.frame(
    term = names(fe),
    OR   = exp(fe),
    CI_lower = exp(cc[, 1]),
    CI_upper = exp(cc[, 2]),
    row.names = NULL
  )
}

#' Run omnibus LRT: compare full model vs model without the focal term
run_lrt <- function(full_model, reduced_formula, data) {
  reduced_model <- update(full_model, formula = reduced_formula, data = data)
  test <- anova(reduced_model, full_model, test = "Chisq")
  return(test)
}

#' Quick descriptive summary by condition
descriptive_by_condition <- function(df, group_var = "condition") {
  df %>%
    group_by(across(all_of(group_var))) %>%
    summarise(
      n         = n(),
      accuracy  = mean(correct, na.rm = TRUE),
      mean_conf = mean(confidence_num, na.rm = TRUE),
      med_conf  = median(confidence_num, na.rm = TRUE),
      .groups   = "drop"
    )
}

#' Quick descriptive summary by condition AND judge
descriptive_by_condition_judge <- function(df, group_var = "condition") {
  df %>%
    group_by(across(all_of(c(group_var, "judge")))) %>%
    summarise(
      n         = n(),
      accuracy  = mean(correct, na.rm = TRUE),
      mean_conf = mean(confidence_num, na.rm = TRUE),
      .groups   = "drop"
    )
}

#' Overconfidence descriptive: confidence when correct vs incorrect
overconfidence_descriptive <- function(df, group_var = "condition") {
  df %>%
    mutate(correctness = ifelse(correct == 1, "correct", "incorrect")) %>%
    group_by(across(all_of(c(group_var, "correctness")))) %>%
    summarise(
      n         = n(),
      mean_conf = mean(confidence_num, na.rm = TRUE),
      med_conf  = median(confidence_num, na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    pivot_wider(
      names_from  = correctness,
      values_from = c(n, mean_conf, med_conf),
      names_sep   = "_"
    )
}

#' Save results to a text file
save_results <- function(text, filename) {
  path <- file.path(OUT_DIR, filename)
  writeLines(text, path)
  cat(sprintf("Results saved to %s\n", path))
}

cat("\nSetup complete. Run individual experiment scripts next.\n")
