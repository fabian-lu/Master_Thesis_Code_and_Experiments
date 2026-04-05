# ==============================================================================
# 06_sensitivity.R — Cross-experiment sensitivity analyses
# ==============================================================================
# Pre-specified robustness checks:
#   1. Judge-specific analyses (run per judge)
#   2. Generator effects (NLE conditions only)
#   3. Same-family bias check (generator == judge)
#   4. Bayesian robustness (brms)
#   5. Averaged-data analysis (GEE / nonparametric)
#   6. Random slopes attempt
# ==============================================================================

source("/home/fabian/Desktop/Second_XAI_Paper/Code/new_experiments/statistical_analysis/00_setup.R")

cat("\n")
cat("================================================================\n")
cat("  SENSITIVITY ANALYSES\n")
cat("================================================================\n")

# ==============================================================================
# 1. Judge-Specific Analyses
# ==============================================================================

cat("\n=== 1. JUDGE-SPECIFIC ANALYSES ===\n")
cat("Running primary models separately for each judge.\n")

run_judge_specific <- function(df, formula_str, exp_name) {
  cat(sprintf("\n--- %s ---\n", exp_name))
  for (j in c("gpt", "deepseek")) {
    cat(sprintf("\nJudge: %s\n", j))
    df_j <- df %>% filter(judge == j)
    tryCatch({
      m <- glmer(as.formula(formula_str),
                 data = df_j, family = binomial,
                 control = glmerControl(optimizer = "bobyqa"))
      cat("Fixed effects:\n")
      print(fixef(m))
      cat("Odds ratios:\n")
      print(exp(fixef(m)))
    }, error = function(e) {
      cat(sprintf("Model failed: %s\n", e$message))
    })
  }
}

run_judge_specific(e1, "correct ~ condition + (1 | instance_idx)", "E1")
run_judge_specific(e2, "correct ~ poisoning * has_nle + (1 | instance_idx)", "E2")
run_judge_specific(e3, "correct ~ condition + (1 | instance_idx)", "E3")
run_judge_specific(e4, "correct ~ condition + (1 | instance_idx)", "E4")
run_judge_specific(e5, "correct ~ condition + (1 | instance_idx)", "E5")

# ==============================================================================
# 2. Generator Effects (NLE conditions only)
# ==============================================================================

cat("\n\n=== 2. GENERATOR EFFECTS ===\n")
cat("Testing if NLE source (GPT vs DeepSeek) matters within NLE conditions.\n")

run_generator_test <- function(df, nle_filter_expr, exp_name) {
  cat(sprintf("\n--- %s ---\n", exp_name))
  df_nle <- df %>% filter(!!nle_filter_expr, generator != "none")
  cat(sprintf("NLE rows: %d\n", nrow(df_nle)))

  if (nrow(df_nle) < 30) {
    cat("Too few NLE rows to test generator effect.\n")
    return(NULL)
  }

  tryCatch({
    m <- glmer(correct ~ generator + judge + (1 | instance_idx),
               data = df_nle, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
    cat("Generator effect:\n")
    print(summary(m)$coefficients)
    cat("OR for generator:\n")
    print(exp(fixef(m)))
  }, error = function(e) {
    cat(sprintf("Model failed: %s\n", e$message))
  })
}

run_generator_test(e1, expr(condition == "E+X+T"), "E1 (E+X+T only)")
run_generator_test(e2, expr(has_nle == "True"), "E2 (has_nle=True)")
run_generator_test(e3, expr(condition == "E+X"), "E3 (E+X only)")
run_generator_test(e4, expr(condition == "E"), "E4 (E condition only)")
run_generator_test(e5, expr(condition != "Baseline"), "E5 (Real + Placebo)")

# ==============================================================================
# 3. Same-Family Bias Check
# ==============================================================================

cat("\n\n=== 3. SAME-FAMILY BIAS CHECK ===\n")
cat("Testing if accuracy is higher when generator == judge.\n")

run_same_family <- function(df, nle_filter_expr, exp_name) {
  cat(sprintf("\n--- %s ---\n", exp_name))
  df_nle <- df %>%
    filter(!!nle_filter_expr, generator != "none") %>%
    mutate(same_family = factor(ifelse(as.character(generator) == as.character(judge),
                                       "same", "different")))

  cat(sprintf("Same-family rows: %d, Different: %d\n",
              sum(df_nle$same_family == "same"),
              sum(df_nle$same_family == "different")))

  cat("Accuracy by same_family:\n")
  print(df_nle %>%
    group_by(same_family) %>%
    summarise(accuracy = mean(correct), n = n(), .groups = "drop"))

  tryCatch({
    m <- glmer(correct ~ same_family + judge + (1 | instance_idx),
               data = df_nle, family = binomial,
               control = glmerControl(optimizer = "bobyqa"))
    cat("Same-family effect:\n")
    print(summary(m)$coefficients)
  }, error = function(e) {
    cat(sprintf("Model failed: %s\n", e$message))
  })
}

run_same_family(e1, expr(condition == "E+X+T"), "E1")
run_same_family(e2, expr(has_nle == "True"), "E2")
run_same_family(e3, expr(condition == "E+X"), "E3")
run_same_family(e5, expr(condition != "Baseline"), "E5")

# ==============================================================================
# 4. Bayesian Robustness (brms)
# ==============================================================================

cat("\n\n=== 4. BAYESIAN ROBUSTNESS ===\n")
cat("Fitting Bayesian GLMMs with weakly informative priors.\n")
cat("This section requires the 'brms' package and can take several minutes.\n")

library(brms)

run_bayesian <- function(formula, data, exp_name) {
  cat(sprintf("\n--- %s Bayesian GLMM ---\n", exp_name))
  bm <- brm(
    formula,
    data = data,
    family = bernoulli(link = "logit"),
    prior = c(
      prior(normal(0, 1.5), class = "b"),
      prior(student_t(3, 0, 2.5), class = "sd")
    ),
    chains = 4, iter = 4000, cores = 4,
    seed = 42, silent = 2, refresh = 0
  )
  cat("Fixed effects (posterior):\n")
  fe <- fixef(bm)
  print(round(fe, 3))
  cat("\nOdds Ratios (posterior median + 95% CI):\n")
  or <- exp(fe)
  print(round(or, 3))
  cat("\n")
  return(bm)
}

bm1 <- run_bayesian(correct ~ condition + judge + (1 | instance_idx), e1, "E1")
bm2 <- run_bayesian(correct ~ poisoning * has_nle + judge + (1 | instance_idx), e2, "E2")
bm3 <- run_bayesian(correct ~ condition + judge + (1 | instance_idx), e3, "E3")
bm4 <- run_bayesian(correct ~ condition + judge + (1 | instance_idx), e4, "E4")
bm5 <- run_bayesian(correct ~ condition + judge + (1 | instance_idx), e5, "E5")

# ==============================================================================
# 5. Averaged-Data Analysis (Nonparametric Backup)
# ==============================================================================

cat("\n\n=== 5. AVERAGED-DATA ANALYSIS (Nonparametric) ===\n")
cat("Collapsing to one accuracy per instance per condition, then Friedman/Wilcoxon.\n")

# E1: Friedman test (5 conditions)
cat("\n--- E1: Friedman test ---\n")
e1_avg <- e1 %>%
  group_by(instance_idx, condition) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  pivot_wider(names_from = condition, values_from = accuracy)

friedman_matrix <- as.matrix(e1_avg[, -1])
tryCatch({
  ft <- friedman.test(friedman_matrix)
  print(ft)
}, error = function(e) {
  cat(sprintf("Friedman test failed: %s\n", e$message))
})

# E3: Wilcoxon signed-rank (2 conditions)
cat("\n--- E3: Wilcoxon signed-rank ---\n")
e3_avg <- e3 %>%
  group_by(instance_idx, condition) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  pivot_wider(names_from = condition, values_from = accuracy)

tryCatch({
  wt <- wilcox.test(e3_avg$`E+X`, e3_avg$X, paired = TRUE)
  print(wt)
}, error = function(e) {
  cat(sprintf("Wilcoxon test failed: %s\n", e$message))
})

# E4: Wilcoxon signed-rank (2 conditions)
cat("\n--- E4: Wilcoxon signed-rank ---\n")
e4_avg <- e4 %>%
  group_by(instance_idx, condition) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  pivot_wider(names_from = condition, values_from = accuracy)

tryCatch({
  wt4 <- wilcox.test(e4_avg$E, e4_avg$Baseline, paired = TRUE)
  print(wt4)
}, error = function(e) {
  cat(sprintf("Wilcoxon test failed: %s\n", e$message))
})

# E5: Friedman test (3 conditions)
cat("\n--- E5: Friedman test ---\n")
e5_avg <- e5 %>%
  group_by(instance_idx, condition) %>%
  summarise(accuracy = mean(correct), .groups = "drop") %>%
  pivot_wider(names_from = condition, values_from = accuracy)

friedman_matrix5 <- as.matrix(e5_avg[, -1])
tryCatch({
  ft5 <- friedman.test(friedman_matrix5)
  print(ft5)
}, error = function(e) {
  cat(sprintf("Friedman test failed: %s\n", e$message))
})

# ==============================================================================
# 6. Random Slopes Attempt
# ==============================================================================

cat("\n\n=== 6. RANDOM SLOPES ATTEMPT ===\n")
cat("Trying random slopes for condition by instance. Keep if converges.\n")

try_random_slopes <- function(df, formula_full, formula_slope, exp_name) {
  cat(sprintf("\n--- %s ---\n", exp_name))
  tryCatch({
    m_slope <- glmer(as.formula(formula_slope),
                     data = df, family = binomial,
                     control = glmerControl(optimizer = "bobyqa",
                                            optCtrl = list(maxfun = 50000)))
    sing <- isSingular(m_slope)
    cat(sprintf("Random slopes model: %s\n",
                ifelse(sing, "SINGULAR — drop back to intercept only",
                       "CONVERGED — consider keeping")))
    if (!sing) {
      m_int <- glmer(as.formula(formula_full),
                     data = df, family = binomial,
                     control = glmerControl(optimizer = "bobyqa"))
      cat("LRT: random slopes vs intercept only:\n")
      print(anova(m_int, m_slope))
    }
  }, error = function(e) {
    cat(sprintf("Random slopes failed to converge: %s\n", e$message))
    cat("Sticking with random intercept only.\n")
  })
}

try_random_slopes(e1,
  "correct ~ condition + judge + (1 | instance_idx)",
  "correct ~ condition + judge + (1 + condition | instance_idx)",
  "E1")

try_random_slopes(e3,
  "correct ~ condition + judge + (1 | instance_idx)",
  "correct ~ condition + judge + (1 + condition | instance_idx)",
  "E3")

try_random_slopes(e5,
  "correct ~ condition + judge + (1 | instance_idx)",
  "correct ~ condition + judge + (1 + condition | instance_idx)",
  "E5")

cat("\n================================================================\n")
cat("  SENSITIVITY ANALYSES COMPLETE\n")
cat("================================================================\n")
