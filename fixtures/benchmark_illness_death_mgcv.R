# Benchmark: mgcv PAM fit for Illness-Death Model
# Compare with MultistateModels.jl spline results
#
# Model structure:
#   State 1 (Healthy) → State 2 (Illness) → State 3 (Death)
#                    ↘ State 3 (Death)

library(mgcv)
library(jsonlite)
library(survival)
library(mstate)
library(expm)

cat(paste(rep("=", 70), collapse=""), "\n")
cat("Illness-Death Benchmark: mgcv PAM Fit\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# ============================================================================
# Load Data
# ============================================================================

dat <- read.csv("illness_death_data.csv")
meta <- fromJSON("illness_death_metadata.json")

cat("\nData summary:\n")
cat("  N subjects:", meta$n_subjects, "\n")
cat("  Max time:", meta$max_time, "\n")
cat("  Transitions 1→2:", meta$trans_12, "\n")
cat("  Transitions 1→3:", meta$trans_13, "\n")
cat("  Transitions 2→3:", meta$trans_23, "\n")

# True hazard function (our rate parameterization)
true_hazard <- function(t, shape, rate) {
  shape * rate * t^(shape - 1)
}

# ============================================================================
# Prepare PAM Data Structure
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Preparing PAM Data Structure\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Create separate datasets for each transition
n_intervals <- 25
max_t <- max(dat$tstop)
breaks <- seq(0, max_t * 1.01, length.out = n_intervals + 1)

# Function to create PAM data for a single transition
create_pam_data <- function(dat, from_state, to_state, breaks) {
  # Filter to subjects who were in from_state
  trans_dat <- dat[dat$from == from_state, ]
  
  pam_rows <- list()
  
  for (i in 1:nrow(trans_dat)) {
    row <- trans_dat[i, ]
    t_start <- row$tstart
    t_end <- row$tstop
    is_event <- (row$status == 1) && (row$to == to_state)
    
    # Find intervals at risk
    for (j in 1:(length(breaks)-1)) {
      int_start <- breaks[j]
      int_end <- breaks[j+1]
      
      # Subject at risk if interval overlaps their at-risk time
      if (int_end <= t_start) next  # Interval before subject's at-risk time
      if (int_start >= t_end) break  # Past subject's observation
      
      # Compute actual at-risk time in this interval
      risk_start <- max(int_start, t_start)
      risk_end <- min(int_end, t_end)
      offset_val <- log(risk_end - risk_start)
      
      # Event in this interval?
      event_in_interval <- is_event && (t_end <= int_end) && (t_end > int_start)
      
      pam_rows[[length(pam_rows) + 1]] <- data.frame(
        id = row$id,
        interval = j,
        t_mid = (risk_start + risk_end) / 2,
        t_start = risk_start,
        t_end = risk_end,
        offset = offset_val,
        event = as.integer(event_in_interval)
      )
    }
  }
  
  do.call(rbind, pam_rows)
}

# Create PAM data for each transition
pam_12 <- create_pam_data(dat, 1, 2, breaks)
pam_13 <- create_pam_data(dat, 1, 3, breaks)
pam_23 <- create_pam_data(dat, 2, 3, breaks)

cat("\nPAM data created:\n")
cat("  Transition 1→2: ", nrow(pam_12), " rows, ", sum(pam_12$event), " events\n")
cat("  Transition 1→3: ", nrow(pam_13), " rows, ", sum(pam_13$event), " events\n")
cat("  Transition 2→3: ", nrow(pam_23), " rows, ", sum(pam_23$event), " events\n")

# ============================================================================
# Fit mgcv GAMs
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Fitting mgcv GAMs (REML)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

t_start <- Sys.time()

# Fit transition 1→2
cat("\nFitting h12 (Healthy→Illness)...\n")
fit_12 <- gam(event ~ s(t_mid, bs="cr", k=8), 
              family = poisson(), 
              offset = offset,
              data = pam_12,
              method = "NCV")
cat("  EDF:", round(sum(fit_12$edf), 2), "\n")

# Fit transition 1→3
cat("\nFitting h13 (Healthy→Death)...\n")
fit_13 <- gam(event ~ s(t_mid, bs="cr", k=8), 
              family = poisson(), 
              offset = offset,
              data = pam_13,
              method = "NCV")
cat("  EDF:", round(sum(fit_13$edf), 2), "\n")

# Fit transition 2→3
cat("\nFitting h23 (Illness→Death)...\n")
fit_23 <- gam(event ~ s(t_mid, bs="cr", k=8), 
              family = poisson(), 
              offset = offset,
              data = pam_23,
              method = "NCV")
cat("  EDF:", round(sum(fit_23$edf), 2), "\n")

t_elapsed <- as.numeric(Sys.time() - t_start, units = "secs")
cat("\nTotal fit time:", round(t_elapsed, 1), "seconds\n")

# ============================================================================
# Predict Hazards
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Predicting Hazards\n")
cat(paste(rep("=", 70), collapse=""), "\n")

eval_times <- meta$eval_times
pred_data <- data.frame(t_mid = eval_times, offset = 0)

# Predict log-hazards
log_h12_mgcv <- predict(fit_12, newdata = pred_data, type = "link")
log_h13_mgcv <- predict(fit_13, newdata = pred_data, type = "link")
log_h23_mgcv <- predict(fit_23, newdata = pred_data, type = "link")

# Convert to hazards
h12_mgcv <- exp(log_h12_mgcv)
h13_mgcv <- exp(log_h13_mgcv)
h23_mgcv <- exp(log_h23_mgcv)

# True hazards
h12_true <- true_hazard(eval_times, meta$true_params$h12$shape, meta$true_params$h12$rate)
h13_true <- true_hazard(eval_times, meta$true_params$h13$shape, meta$true_params$h13$rate)
h23_true <- true_hazard(eval_times, meta$true_params$h23$shape, meta$true_params$h23$rate)

# RMSE for hazards
rmse <- function(a, b) sqrt(mean((a - b)^2))
cat("\nHazard RMSE (mgcv vs true):\n")
cat("  h12:", round(rmse(h12_mgcv, h12_true), 5), "\n")
cat("  h13:", round(rmse(h13_mgcv, h13_true), 5), "\n")
cat("  h23:", round(rmse(h23_mgcv, h23_true), 5), "\n")

# ============================================================================
# Compute Transition Probabilities via Product Integral
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Computing Transition Probabilities\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Function to compute transition probability matrix using product integral
compute_tpm <- function(t, h12_fn, h13_fn, h23_fn, dt = 0.01) {
  if (t <= 0) return(diag(3))
  
  n_steps <- max(1, ceiling(t / dt))
  actual_dt <- t / n_steps
  
  P <- diag(3)
  
  for (i in 1:n_steps) {
    s <- (i - 0.5) * actual_dt  # midpoint
    
    # Hazards at midpoint
    h12 <- h12_fn(s)
    h13 <- h13_fn(s)
    h23 <- h23_fn(s)
    
    # Generator matrix
    Q <- matrix(c(-(h12 + h13), h12, h13,
                  0, -h23, h23,
                  0, 0, 0), nrow = 3, byrow = TRUE)
    
    # Matrix exponential
    dP <- expm(Q * actual_dt)
    P <- P %*% dP
  }
  
  return(P)
}

# Create hazard interpolation functions for mgcv predictions
h12_mgcv_fn <- approxfun(eval_times, h12_mgcv, rule = 2)
h13_mgcv_fn <- approxfun(eval_times, h13_mgcv, rule = 2)
h23_mgcv_fn <- approxfun(eval_times, h23_mgcv, rule = 2)

# True hazard functions
h12_true_fn <- function(t) true_hazard(t, meta$true_params$h12$shape, meta$true_params$h12$rate)
h13_true_fn <- function(t) true_hazard(t, meta$true_params$h13$shape, meta$true_params$h13$rate)
h23_true_fn <- function(t) true_hazard(t, meta$true_params$h23$shape, meta$true_params$h23$rate)

# Compute state prevalence at eval times
cat("\nComputing state prevalence...\n")

prev_true <- matrix(0, nrow = length(eval_times), ncol = 3)
prev_mgcv <- matrix(0, nrow = length(eval_times), ncol = 3)

for (i in 1:length(eval_times)) {
  t <- eval_times[i]
  
  # True transition probabilities
  P_true <- compute_tpm(t, h12_true_fn, h13_true_fn, h23_true_fn)
  prev_true[i, ] <- P_true[1, ]  # Row 1 = starting from state 1
  
  # mgcv transition probabilities
  P_mgcv <- compute_tpm(t, h12_mgcv_fn, h13_mgcv_fn, h23_mgcv_fn)
  prev_mgcv[i, ] <- P_mgcv[1, ]
}

# ============================================================================
# Compute Cumulative Incidence
# ============================================================================

cat("Computing cumulative incidence...\n")

# Cumulative incidence for transition 1→2 (illness)
# CI_12(t) = integral_0^t P11(s) * h12(s) ds
compute_ci <- function(times, h_fn, P_fn, dt = 0.05) {
  ci <- numeric(length(times))
  
  for (i in 1:length(times)) {
    t <- times[i]
    if (t <= 0) {
      ci[i] <- 0
      next
    }
    
    n_steps <- max(1, ceiling(t / dt))
    actual_dt <- t / n_steps
    
    integral <- 0
    for (j in 1:n_steps) {
      s <- (j - 0.5) * actual_dt
      P_s <- P_fn(s)  # P11(s)
      h_s <- h_fn(s)
      integral <- integral + P_s * h_s * actual_dt
    }
    ci[i] <- integral
  }
  
  return(ci)
}

# P11 functions (probability of staying in state 1)
P11_true_fn <- function(t) compute_tpm(t, h12_true_fn, h13_true_fn, h23_true_fn)[1, 1]
P11_mgcv_fn <- function(t) compute_tpm(t, h12_mgcv_fn, h13_mgcv_fn, h23_mgcv_fn)[1, 1]

# This is slow, so use vectorized approximation
# Pre-compute P11 at finer grid
fine_times <- seq(0.01, max(eval_times), by = 0.1)
P11_true_vec <- sapply(fine_times, function(t) compute_tpm(t, h12_true_fn, h13_true_fn, h23_true_fn)[1, 1])
P11_mgcv_vec <- sapply(fine_times, function(t) compute_tpm(t, h12_mgcv_fn, h13_mgcv_fn, h23_mgcv_fn)[1, 1])

P11_true_approx <- approxfun(fine_times, P11_true_vec, rule = 2)
P11_mgcv_approx <- approxfun(fine_times, P11_mgcv_vec, rule = 2)

ci_illness_true <- compute_ci(eval_times, h12_true_fn, P11_true_approx)
ci_illness_mgcv <- compute_ci(eval_times, h12_mgcv_fn, P11_mgcv_approx)

ci_death_direct_true <- compute_ci(eval_times, h13_true_fn, P11_true_approx)
ci_death_direct_mgcv <- compute_ci(eval_times, h13_mgcv_fn, P11_mgcv_approx)

# Total death CI is just prevalence in state 3 (absorbing)
ci_death_total_true <- prev_true[, 3]
ci_death_total_mgcv <- prev_mgcv[, 3]

# ============================================================================
# Compare Results
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Results Comparison\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Load Julia results for comparison
julia_results <- fromJSON("illness_death_results.json")

cat("\n--- State Prevalence RMSE ---\n")
cat("True vs mgcv:\n")
cat("  P(Healthy):", round(rmse(prev_true[,1], prev_mgcv[,1]), 5), "\n")
cat("  P(Illness):", round(rmse(prev_true[,2], prev_mgcv[,2]), 5), "\n")
cat("  P(Death):  ", round(rmse(prev_true[,3], prev_mgcv[,3]), 5), "\n")

cat("\nTrue vs Empirical:\n")
cat("  P(Healthy):", round(rmse(prev_true[,1], julia_results$empirical$prevalence_healthy), 5), "\n")
cat("  P(Illness):", round(rmse(prev_true[,2], julia_results$empirical$prevalence_illness), 5), "\n")
cat("  P(Death):  ", round(rmse(prev_true[,3], julia_results$empirical$prevalence_death), 5), "\n")

cat("\n--- Cumulative Incidence RMSE ---\n")
cat("CI Illness (True vs mgcv):", round(rmse(ci_illness_true, ci_illness_mgcv), 5), "\n")
cat("CI Death Direct (True vs mgcv):", round(rmse(ci_death_direct_true, ci_death_direct_mgcv), 5), "\n")
cat("CI Death Total (True vs mgcv):", round(rmse(ci_death_total_true, ci_death_total_mgcv), 5), "\n")

# ============================================================================
# Sample Values Table
# ============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Sample State Prevalence Values\n")
cat(paste(rep("=", 70), collapse=""), "\n")

cat("\nTime\tTrue P1\t\tmgcv P1\t\tEmp P1\n")
idx <- c(1, 10, 20, 30, 40)
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t\t%.4f\t\t%.4f\n", 
                eval_times[i], 
                prev_true[i, 1], 
                prev_mgcv[i, 1],
                julia_results$empirical$prevalence_healthy[i]))
  }
}

cat("\nTime\tTrue P2\t\tmgcv P2\t\tEmp P2\n")
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t\t%.4f\t\t%.4f\n", 
                eval_times[i], 
                prev_true[i, 2], 
                prev_mgcv[i, 2],
                julia_results$empirical$prevalence_illness[i]))
  }
}

cat("\nTime\tTrue P3\t\tmgcv P3\t\tEmp P3\n")
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t\t%.4f\t\t%.4f\n", 
                eval_times[i], 
                prev_true[i, 3], 
                prev_mgcv[i, 3],
                julia_results$empirical$prevalence_death[i]))
  }
}

# ============================================================================
# Save mgcv Results
# ============================================================================

results <- list(
  hazards = list(
    h12_mgcv = as.numeric(h12_mgcv),
    h13_mgcv = as.numeric(h13_mgcv),
    h23_mgcv = as.numeric(h23_mgcv),
    h12_true = h12_true,
    h13_true = h13_true,
    h23_true = h23_true
  ),
  prevalence = list(
    mgcv_healthy = as.numeric(prev_mgcv[, 1]),
    mgcv_illness = as.numeric(prev_mgcv[, 2]),
    mgcv_death = as.numeric(prev_mgcv[, 3]),
    true_healthy = as.numeric(prev_true[, 1]),
    true_illness = as.numeric(prev_true[, 2]),
    true_death = as.numeric(prev_true[, 3])
  ),
  cumulative_incidence = list(
    ci_illness_mgcv = ci_illness_mgcv,
    ci_illness_true = ci_illness_true,
    ci_death_direct_mgcv = ci_death_direct_mgcv,
    ci_death_direct_true = ci_death_direct_true,
    ci_death_total_mgcv = ci_death_total_mgcv,
    ci_death_total_true = ci_death_total_true
  ),
  rmse = list(
    hazard_h12 = rmse(h12_mgcv, h12_true),
    hazard_h13 = rmse(h13_mgcv, h13_true),
    hazard_h23 = rmse(h23_mgcv, h23_true),
    prev_healthy = rmse(prev_true[,1], prev_mgcv[,1]),
    prev_illness = rmse(prev_true[,2], prev_mgcv[,2]),
    prev_death = rmse(prev_true[,3], prev_mgcv[,3])
  ),
  fit_info = list(
    edf_h12 = sum(fit_12$edf),
    edf_h13 = sum(fit_13$edf),
    edf_h23 = sum(fit_23$edf),
    fit_time_seconds = t_elapsed
  ),
  eval_times = eval_times
)

write_json(results, "illness_death_mgcv_results.json", pretty = TRUE, auto_unbox = TRUE)
cat("\nSaved mgcv results to: illness_death_mgcv_results.json\n")

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Benchmark Complete!\n")
cat(paste(rep("=", 70), collapse=""), "\n")
