# ==============================================================================
# 00_setup.R — Shared setup, data loading, helper functions
# ==============================================================================
# Source this file at the top of each experiment script.
# Implements the revised plan from analysis.md (post-council).
# ==============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(ordinal)
library(emmeans)
library(performance)
library(DHARMa)
library(broom.mixed)
library(brms)
library(Hmisc)  # for Somers' D

# --- Paths ---
BASE_DIR <- "/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/results"
OUT_DIR  <- "/home/fabian/Desktop/Master_thesis/code_and_experiments/6_part2_usefulness/statistical_analysis/output"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUT_DIR, "plots"), showWarnings = FALSE)

# ==============================================================================
# Data Loading
# ==============================================================================

load_experiment <- function(exp_name) {
  path <- file.path(BASE_DIR, exp_name, "full_results.csv")
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$instance_idx   <- factor(df$instance_idx)
  df$judge          <- factor(df$judge)
  df$generator      <- factor(df$generator)
  df$correct        <- as.integer(df$correct)
  df$confidence     <- ordered(df$confidence, levels = 1:5)
  df$confidence_num <- as.numeric(as.character(df$confidence))
  return(df)
}

cat("Loading experiments...\n")
e1 <- load_experiment("E1")
e2 <- load_experiment("E2")
e3 <- load_experiment("E3")
e4 <- load_experiment("E4")
e5 <- load_experiment("E5")
cat(sprintf("  E1: %d rows\n  E2: %d rows\n  E3: %d rows\n  E4: %d rows\n  E5: %d rows\n",
            nrow(e1), nrow(e2), nrow(e3), nrow(e4), nrow(e5)))

# ==============================================================================
# Helper Functions
# ==============================================================================

# --- Descriptive stats by condition ---
descriptives <- function(df) {
  df %>%
    group_by(condition) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_conf = mean(confidence_num, na.rm = TRUE),
      median_conf = median(confidence_num, na.rm = TRUE),
      conf_na = sum(is.na(confidence_num)),
      .groups = "drop"
    )
}

# --- Descriptive stats by condition x judge ---
descriptives_by_judge <- function(df) {
  df %>%
    group_by(condition, judge) %>%
    summarise(
      n = n(),
      accuracy = mean(correct, na.rm = TRUE),
      mean_conf = mean(confidence_num, na.rm = TRUE),
      .groups = "drop"
    )
}

# --- Primary GLMM: condition * judge + (1 | instance_idx) ---
# Always fits the interaction model (no step-down).
# Returns: model, omnibus LRT, marginal EMMs, judge-specific EMMs
fit_accuracy_model <- function(df, formula_full = NULL, formula_red = NULL) {
  if (is.null(formula_full)) {
    formula_full <- correct ~ condition * judge + (1 | instance_idx)
  }
  if (is.null(formula_red)) {
    formula_red <- correct ~ judge + (1 | instance_idx)
  }

  m_full <- glmer(formula_full, data = df, family = binomial,
                  control = glmerControl(optimizer = "bobyqa"))
  m_red  <- glmer(formula_red, data = df, family = binomial,
                  control = glmerControl(optimizer = "bobyqa"))

  lrt <- anova(m_red, m_full, test = "Chisq")

  # Marginal condition effects (averaged over judges)
  emm_cond <- emmeans(m_full, ~ condition, type = "response")

  # Judge-specific condition effects
  emm_judge <- emmeans(m_full, ~ condition | judge, type = "response")

  # Interaction estimate
  emm_int <- emmeans(m_full, ~ condition * judge, type = "response")

  list(model = m_full, model_red = m_red, lrt = lrt,
       emm_cond = emm_cond, emm_judge = emm_judge, emm_int = emm_int)
}

# --- Primary CLMM: condition * judge + (1 | instance_idx) ---
fit_confidence_model <- function(df, formula_full = NULL, formula_add = NULL, formula_red = NULL) {
  if (is.null(formula_full)) {
    formula_full <- confidence ~ condition * judge + (1 | instance_idx)
  }
  if (is.null(formula_add)) {
    formula_add <- confidence ~ condition + judge + (1 | instance_idx)
  }
  if (is.null(formula_red)) {
    formula_red <- confidence ~ judge + (1 | instance_idx)
  }

  m_full <- tryCatch(clmm(formula_full, data = df),
                     error = function(e) { cat(sprintf("  CLMM interaction model failed: %s\n  Falling back to additive.\n", e$message)); NULL })

  m_add <- clmm(formula_add, data = df)
  m_red <- clmm(formula_red, data = df)

  # Use the interaction model if it worked, otherwise additive
  m_primary <- if (!is.null(m_full)) m_full else m_add

  # LRT: additive vs reduced (this is the omnibus condition test)
  lrt <- tryCatch(anova(m_red, m_add),
                  error = function(e) { cat(sprintf("  CLMM LRT error: %s\n", e$message)); NULL })

  emm_cond <- tryCatch(emmeans(m_primary, ~ condition),
                       error = function(e) { cat(sprintf("  CLMM emmeans error: %s\n", e$message)); NULL })

  list(model = m_primary, model_add = m_add, model_red = m_red, lrt = lrt, emm_cond = emm_cond)
}

# --- Calibration analysis (replaces overconfidence interaction) ---
run_calibration <- function(df) {
  cat("\n  Calibration model: correct ~ confidence_num * condition + judge + (1|instance_idx)\n")
  m_cal <- tryCatch({
    glmer(correct ~ confidence_num * condition + judge + (1 | instance_idx),
          data = df, family = binomial,
          control = glmerControl(optimizer = "bobyqa"))
  }, error = function(e) { cat(sprintf("  Calibration model error: %s\n", e$message)); NULL })

  if (!is.null(m_cal)) {
    cat("  Calibration model summary:\n")
    print(summary(m_cal))
  }

  # Somers' D per condition
  cat("\n  Somers' D (confidence vs accuracy) per condition:\n")
  somers <- df %>%
    group_by(condition) %>%
    summarise(
      somers_d = somers2(confidence_num, correct)["Dxy"],
      n = n(),
      .groups = "drop"
    )
  print(as.data.frame(somers))

  # Empirical calibration curve data
  cat("\n  Empirical calibration (accuracy at each confidence level, per condition):\n")
  cal_curve <- df %>%
    group_by(condition, confidence_num) %>%
    summarise(accuracy = mean(correct), n = n(), .groups = "drop")
  print(as.data.frame(cal_curve))

  # Conditional confidence table (descriptive only)
  cat("\n  Conditional confidence (descriptive — NOT causal):\n")
  cond_conf <- df %>%
    group_by(condition, correct) %>%
    summarise(mean_conf = mean(confidence_num, na.rm = TRUE), n = n(), .groups = "drop")
  print(as.data.frame(cond_conf))

  list(model = m_cal, somers = somers, cal_curve = cal_curve, cond_conf = cond_conf)
}

# --- DHARMa diagnostics ---
run_diagnostics <- function(model, label = "") {
  cat(sprintf("\n  Diagnostics (%s):\n", label))
  cat(sprintf("  Convergence: %s\n", ifelse(check_convergence(model), "OK", "WARNING")))
  cat(sprintf("  Singularity: %s\n", ifelse(check_singularity(model), "SINGULAR", "OK")))
  tryCatch({
    sim <- simulateResiduals(model)
    cat(sprintf("  DHARMa uniformity p: %.3f\n", testUniformity(sim, plot = FALSE)$p.value))
    cat(sprintf("  DHARMa dispersion p: %.3f\n", testDispersion(sim, plot = FALSE)$p.value))
  }, error = function(e) cat(sprintf("  DHARMa error: %s\n", e$message)))
}

# --- Bayesian GLMM with ROPE ---
run_bayesian_rope <- function(df, formula_str = "correct ~ condition + judge + (1 | instance_idx)",
                              rope_range = c(-0.18, 0.18)) {
  cat("\n  Bayesian GLMM (brms)...\n")
  bm <- tryCatch({
    brm(as.formula(formula_str), data = df,
        family = bernoulli(link = "logit"),
        prior = c(prior(normal(0, 1.5), class = "b"),
                  prior(student_t(3, 0, 2.5), class = "sd")),
        chains = 4, iter = 4000, cores = 4, silent = 2, refresh = 0)
  }, error = function(e) { cat(sprintf("  brms error: %s\n", e$message)); NULL })

  if (!is.null(bm)) {
    cat("  Posterior ORs:\n")
    fe <- fixef(bm)
    print(round(exp(fe), 3))

    # ROPE analysis for each condition parameter
    cond_params <- rownames(fe)[grepl("condition", rownames(fe), ignore.case = TRUE)]
    for (param in cond_params) {
      post <- as.data.frame(bm)[[paste0("b_", param)]]
      if (!is.null(post)) {
        in_rope <- mean(post >= rope_range[1] & post <= rope_range[2]) * 100
        cat(sprintf("  ROPE [%.2f, %.2f] for %s: %.1f%% inside\n",
                    rope_range[1], rope_range[2], param, in_rope))
      }
    }
  }

  return(bm)
}

# --- Sensitivity: judge-specific models ---
run_judge_specific <- function(df, formula_str) {
  cat("\n  Judge-specific models:\n")
  for (j in levels(df$judge)) {
    df_j <- df %>% filter(judge == j)
    tryCatch({
      m <- glmer(as.formula(formula_str), data = df_j, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))
      cat(sprintf("  Judge %s: ", j))
      ors <- round(exp(fixef(m)), 3)
      print(ors)
    }, error = function(e) cat(sprintf("  Judge %s: ERROR — %s\n", j, e$message)))
  }
}

# --- Sensitivity: generator effects ---
run_generator_test <- function(df, nle_condition_filter) {
  df_nle <- df %>% filter(!!nle_condition_filter)
  if (nrow(df_nle) < 20) { cat("  Too few NLE rows\n"); return(NULL) }

  # If only one condition level in the NLE subset, drop condition from formula
  n_conds <- length(unique(df_nle$condition))
  gen_formula <- if (n_conds > 1) {
    correct ~ condition + generator + judge + (1 | instance_idx)
  } else {
    correct ~ generator + judge + (1 | instance_idx)
  }

  cat(sprintf("\n  Generator effect (NLE conditions only, N=%d, %d conditions):\n", nrow(df_nle), n_conds))
  tryCatch({
    m_gen <- glmer(gen_formula, data = df_nle, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))
    cat("  Generator coefficient:\n")
    print(round(summary(m_gen)$coefficients, 4))
  }, error = function(e) cat(sprintf("  Generator model error: %s\n", e$message)))

  cat("\n  Same-family bias:\n")
  tryCatch({
    df_nle$same_family <- as.integer(as.character(df_nle$generator) == as.character(df_nle$judge))
    fam_formula <- if (n_conds > 1) {
      correct ~ condition + same_family + judge + (1 | instance_idx)
    } else {
      correct ~ same_family + judge + (1 | instance_idx)
    }
    m_fam <- glmer(fam_formula, data = df_nle, family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))
    cat(sprintf("  Same-family OR: %.3f, p: %.4f\n",
                exp(fixef(m_fam)["same_family"]),
                summary(m_fam)$coefficients["same_family", "Pr(>|z|)"]))
  }, error = function(e) cat(sprintf("  Same-family error: %s\n", e$message)))
}

# --- Sensitivity: random slopes ---
run_random_slopes <- function(df, formula_str) {
  cat("\n  Random slopes test:\n")
  tryCatch({
    m <- glmer(as.formula(formula_str), data = df, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
    cat(sprintf("  Singular: %s\n", ifelse(isSingular(m), "YES (confirms intercept-only)", "NO")))
  }, error = function(e) cat(sprintf("  Random slopes failed: %s\n", e$message)))
}

cat("Setup complete.\n")
