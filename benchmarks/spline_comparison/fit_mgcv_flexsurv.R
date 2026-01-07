# =============================================================================
# Penalized Spline Benchmark: mgcv and flexsurv Fits
# =============================================================================
#
# This script fits illness-death models using:
# 1. mgcv: Piecewise-exponential model (PAM) with smooth baseline hazards
# 2. flexsurv: Royston-Parmar flexible survival model with splines
#
# Both approaches use cubic splines with automatic smoothing selection.
# =============================================================================

library(mgcv)
library(flexsurv)
library(survival)
library(jsonlite)

cat(paste(rep("=", 70), collapse=""), "\n")
cat("Penalized Spline Benchmark: mgcv and flexsurv Fits\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Set working directory to script location
# Handle both interactive and Rscript execution
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0 && nchar(script_path) > 0) {
  setwd(dirname(script_path))
} else if (exists("sys.frame") && tryCatch({sys.frame(1)$ofile; TRUE}, error = function(e) FALSE)) {
  setwd(dirname(sys.frame(1)$ofile))
}
# Otherwise assume we're already in the right directory

# =============================================================================
# Load Data
# =============================================================================

cat("\n--- Loading Data ---\n")

dat <- read.csv("benchmark_data.csv")
meta <- fromJSON("benchmark_metadata.json")

cat("N subjects:", meta$n_subjects, "\n")
cat("Max time:", meta$max_time, "\n")
cat("Transitions 1→2:", meta$n_12, "\n")
cat("Transitions 1→3:", meta$n_13, "\n")
cat("Transitions 2→3:", meta$n_23, "\n")

eval_times <- meta$eval_times
h12_true <- meta$true_hazards$h12
h13_true <- meta$true_hazards$h13
h23_true <- meta$true_hazards$h23

# Helper functions
rmse <- function(a, b) sqrt(mean((a - b)^2))

# True hazard functions (for verification)
true_h12 <- function(t) ifelse(t > 0, 0.3 * sqrt(t), 0.3 * sqrt(0.01))
true_h13 <- function(t) 0.1 + 0.02 * t
true_h23 <- function(t) 0.4 * exp(-0.1 * t)

# =============================================================================
# Prepare Data for mgcv PAM (Piecewise Exponential Model)
# =============================================================================

cat("\n--- Preparing PAM Data for mgcv ---\n")

# Create separate datasets for each transition
# PAM splits follow-up time into intervals with Poisson events

n_intervals <- 30
max_t <- max(dat$tstop)
breaks <- seq(0, max_t * 1.01, length.out = n_intervals + 1)

create_pam_data <- function(dat, from_state, to_state, breaks) {
  # Filter to rows where subject is in from_state
  trans_dat <- dat[dat$statefrom == from_state, ]
  
  pam_rows <- list()
  
  for (i in 1:nrow(trans_dat)) {
    row <- trans_dat[i, ]
    t_start <- row$tstart
    t_end <- row$tstop
    is_event <- (row$obstype == 1) && (row$stateto == to_state)
    
    # Find intervals at risk
    for (j in 1:(length(breaks)-1)) {
      int_start <- breaks[j]
      int_end <- breaks[j+1]
      
      if (int_end <= t_start) next
      if (int_start >= t_end) break
      
      risk_start <- max(int_start, t_start)
      risk_end <- min(int_end, t_end)
      offset_val <- log(risk_end - risk_start)
      
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

pam_12 <- create_pam_data(dat, 1, 2, breaks)
pam_13 <- create_pam_data(dat, 1, 3, breaks)
pam_23 <- create_pam_data(dat, 2, 3, breaks)

cat("PAM data created:\n")
cat("  Transition 1→2:", nrow(pam_12), "rows,", sum(pam_12$event), "events\n")
cat("  Transition 1→3:", nrow(pam_13), "rows,", sum(pam_13$event), "events\n")
cat("  Transition 2→3:", nrow(pam_23), "rows,", sum(pam_23$event), "events\n")

# =============================================================================
# Fit mgcv GAMs (PAM with smooth baseline)
# =============================================================================

cat("\n--- Fitting mgcv GAMs (NCV/REML) ---\n")

t_mgcv_start <- Sys.time()

# Use cubic regression splines (cr) with k=8 basis functions
# NCV = Neighbourhood Cross-Validation (Wood 2024) - similar to PIJCV
cat("Fitting h12 (1→2)...\n")
fit_12_mgcv <- gam(event ~ s(t_mid, bs="cr", k=8), 
                   family = poisson(), 
                   offset = offset,
                   data = pam_12,
                   method = "NCV")

cat("Fitting h13 (1→3)...\n")
fit_13_mgcv <- gam(event ~ s(t_mid, bs="cr", k=8), 
                   family = poisson(), 
                   offset = offset,
                   data = pam_13,
                   method = "NCV")

cat("Fitting h23 (2→3)...\n")
fit_23_mgcv <- gam(event ~ s(t_mid, bs="cr", k=8), 
                   family = poisson(), 
                   offset = offset,
                   data = pam_23,
                   method = "NCV")

t_mgcv <- as.numeric(Sys.time() - t_mgcv_start, units = "secs")

cat("\nmgcv Results:\n")
cat("  h12 EDF:", round(sum(fit_12_mgcv$edf), 3), "\n")
cat("  h13 EDF:", round(sum(fit_13_mgcv$edf), 3), "\n")
cat("  h23 EDF:", round(sum(fit_23_mgcv$edf), 3), "\n")
cat("  Total time:", round(t_mgcv, 2), "seconds\n")

# Extract smoothing parameters (λ in mgcv notation)
lambda_12_mgcv <- fit_12_mgcv$sp
lambda_13_mgcv <- fit_13_mgcv$sp
lambda_23_mgcv <- fit_23_mgcv$sp

cat("\nmgcv Smoothing Parameters (sp):\n")
cat("  h12:", round(lambda_12_mgcv, 4), "\n")
cat("  h13:", round(lambda_13_mgcv, 4), "\n")
cat("  h23:", round(lambda_23_mgcv, 4), "\n")

# Predict hazards at evaluation times
pred_data <- data.frame(t_mid = eval_times, offset = 0)

h12_mgcv <- exp(predict(fit_12_mgcv, newdata = pred_data, type = "link"))
h13_mgcv <- exp(predict(fit_13_mgcv, newdata = pred_data, type = "link"))
h23_mgcv <- exp(predict(fit_23_mgcv, newdata = pred_data, type = "link"))

cat("\nmgcv Hazard RMSE vs True:\n")
cat("  h12:", round(rmse(h12_mgcv, h12_true), 5), "\n")
cat("  h13:", round(rmse(h13_mgcv, h13_true), 5), "\n")
cat("  h23:", round(rmse(h23_mgcv, h23_true), 5), "\n")

# =============================================================================
# Fit flexsurv Royston-Parmar Models
# =============================================================================

cat("\n--- Fitting flexsurv Royston-Parmar Models ---\n")

# Prepare survival data for each transition
# For flexsurv, we need (time, status) format

# Transition 1→2: competing risk with 1→3
surv_12_dat <- dat[dat$statefrom == 1, ]
surv_12_dat$time <- surv_12_dat$tstop - surv_12_dat$tstart
surv_12_dat$status <- as.integer(surv_12_dat$stateto == 2 & surv_12_dat$obstype == 1)

# Transition 1→3: competing risk with 1→2
surv_13_dat <- dat[dat$statefrom == 1, ]
surv_13_dat$time <- surv_13_dat$tstop - surv_13_dat$tstart
surv_13_dat$status <- as.integer(surv_13_dat$stateto == 3 & surv_13_dat$obstype == 1)

# Transition 2→3: simple survival from state 2
surv_23_dat <- dat[dat$statefrom == 2, ]
surv_23_dat$time <- surv_23_dat$tstop - surv_23_dat$tstart
surv_23_dat$status <- as.integer(surv_23_dat$stateto == 3 & surv_23_dat$obstype == 1)

t_flexsurv_start <- Sys.time()

# Fit using flexsurvspline with 2 internal knots (4 df total = 2 + 2 boundary)
# scale = "hazard" gives proportional hazards spline model
cat("Fitting h12 (1→2)...\n")
fit_12_fs <- tryCatch({
  flexsurvspline(Surv(time, status) ~ 1, data = surv_12_dat, 
                 k = 2, scale = "hazard")
}, error = function(e) {
  cat("  Warning: flexsurv h12 failed, trying simpler model\n")
  flexsurvspline(Surv(time, status) ~ 1, data = surv_12_dat, 
                 k = 1, scale = "hazard")
})

cat("Fitting h13 (1→3)...\n")
fit_13_fs <- tryCatch({
  flexsurvspline(Surv(time, status) ~ 1, data = surv_13_dat, 
                 k = 2, scale = "hazard")
}, error = function(e) {
  cat("  Warning: flexsurv h13 failed, trying simpler model\n")
  flexsurvspline(Surv(time, status) ~ 1, data = surv_13_dat, 
                 k = 1, scale = "hazard")
})

cat("Fitting h23 (2→3)...\n")
fit_23_fs <- tryCatch({
  flexsurvspline(Surv(time, status) ~ 1, data = surv_23_dat, 
                 k = 2, scale = "hazard")
}, error = function(e) {
  cat("  Warning: flexsurv h23 failed, trying simpler model\n")
  flexsurvspline(Surv(time, status) ~ 1, data = surv_23_dat, 
                 k = 1, scale = "hazard")
})

t_flexsurv <- as.numeric(Sys.time() - t_flexsurv_start, units = "secs")

cat("\nflexsurv Results:\n")
cat("  h12 df:", length(coef(fit_12_fs)), "\n")
cat("  h13 df:", length(coef(fit_13_fs)), "\n")
cat("  h23 df:", length(coef(fit_23_fs)), "\n")
cat("  Total time:", round(t_flexsurv, 2), "seconds\n")

# Extract hazard predictions
# Note: flexsurv uses cumulative hazard parameterization, need to differentiate
h12_flexsurv <- numeric(length(eval_times))
h13_flexsurv <- numeric(length(eval_times))
h23_flexsurv <- numeric(length(eval_times))

for (i in seq_along(eval_times)) {
  t <- eval_times[i]
  # flexsurv summary gives hazard directly
  h12_flexsurv[i] <- summary(fit_12_fs, t = t, type = "hazard")[[1]]$est
  h13_flexsurv[i] <- summary(fit_13_fs, t = t, type = "hazard")[[1]]$est
  h23_flexsurv[i] <- summary(fit_23_fs, t = t, type = "hazard")[[1]]$est
}

cat("\nflexsurv Hazard RMSE vs True:\n")
cat("  h12:", round(rmse(h12_flexsurv, h12_true), 5), "\n")
cat("  h13:", round(rmse(h13_flexsurv, h13_true), 5), "\n")
cat("  h23:", round(rmse(h23_flexsurv, h23_true), 5), "\n")

# =============================================================================
# Load Julia Results for Comparison
# =============================================================================

cat("\n--- Loading Julia Results ---\n")

julia_results <- tryCatch({
  fromJSON("julia_results.json")
}, error = function(e) {
  cat("  Julia results not found - run Julia benchmark first\n")
  NULL
})

# =============================================================================
# Summary Comparison
# =============================================================================

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("SUMMARY COMPARISON\n")
cat(paste(rep("=", 70), collapse=""), "\n")

cat("\n--- Hazard RMSE vs True ---\n")
cat(sprintf("%-20s %10s %10s %10s\n", "Method", "h12", "h13", "h23"))
cat(paste(rep("-", 55), collapse=""), "\n")
cat(sprintf("%-20s %10.5f %10.5f %10.5f\n", "mgcv (NCV)", 
            rmse(h12_mgcv, h12_true), rmse(h13_mgcv, h13_true), rmse(h23_mgcv, h23_true)))
cat(sprintf("%-20s %10.5f %10.5f %10.5f\n", "flexsurv", 
            rmse(h12_flexsurv, h12_true), rmse(h13_flexsurv, h13_true), rmse(h23_flexsurv, h23_true)))
if (!is.null(julia_results)) {
  cat(sprintf("%-20s %10.5f %10.5f %10.5f\n", "Julia (PIJCV)", 
              julia_results$rmse$h12, julia_results$rmse$h13, julia_results$rmse$h23))
}

cat("\n--- Computation Time (seconds) ---\n")
cat(sprintf("%-20s %10.2f\n", "mgcv", t_mgcv))
cat(sprintf("%-20s %10.2f\n", "flexsurv", t_flexsurv))
if (!is.null(julia_results)) {
  cat(sprintf("%-20s %10.2f\n", "Julia", julia_results$time_seconds))
}

cat("\n--- Effective Degrees of Freedom ---\n")
cat(sprintf("%-20s %10s %10s %10s\n", "Method", "h12", "h13", "h23"))
cat(paste(rep("-", 55), collapse=""), "\n")
cat(sprintf("%-20s %10.2f %10.2f %10.2f\n", "mgcv", 
            sum(fit_12_mgcv$edf), sum(fit_13_mgcv$edf), sum(fit_23_mgcv$edf)))
cat(sprintf("%-20s %10.2f %10.2f %10.2f\n", "flexsurv", 
            length(coef(fit_12_fs)), length(coef(fit_13_fs)), length(coef(fit_23_fs))))

cat("\n--- Smoothing Parameters ---\n")
cat(sprintf("%-20s %12s %12s %12s\n", "Method", "h12", "h13", "h23"))
cat(paste(rep("-", 60), collapse=""), "\n")
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "mgcv (sp)", 
            lambda_12_mgcv, lambda_13_mgcv, lambda_23_mgcv))
if (!is.null(julia_results)) {
  cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "Julia (λ)", 
              julia_results$lambda[1], julia_results$lambda[2], julia_results$lambda[3]))
}

# =============================================================================
# Export R Results
# =============================================================================

r_results <- list(
  mgcv = list(
    hazards = list(
      h12 = as.numeric(h12_mgcv),
      h13 = as.numeric(h13_mgcv),
      h23 = as.numeric(h23_mgcv)
    ),
    lambda = list(
      h12 = lambda_12_mgcv,
      h13 = lambda_13_mgcv,
      h23 = lambda_23_mgcv
    ),
    edf = list(
      h12 = sum(fit_12_mgcv$edf),
      h13 = sum(fit_13_mgcv$edf),
      h23 = sum(fit_23_mgcv$edf)
    ),
    rmse = list(
      h12 = rmse(h12_mgcv, h12_true),
      h13 = rmse(h13_mgcv, h13_true),
      h23 = rmse(h23_mgcv, h23_true)
    ),
    time_seconds = t_mgcv
  ),
  flexsurv = list(
    hazards = list(
      h12 = h12_flexsurv,
      h13 = h13_flexsurv,
      h23 = h23_flexsurv
    ),
    coefficients = list(
      h12 = as.numeric(coef(fit_12_fs)),
      h13 = as.numeric(coef(fit_13_fs)),
      h23 = as.numeric(coef(fit_23_fs))
    ),
    rmse = list(
      h12 = rmse(h12_flexsurv, h12_true),
      h13 = rmse(h13_flexsurv, h13_true),
      h23 = rmse(h23_flexsurv, h23_true)
    ),
    time_seconds = t_flexsurv
  ),
  eval_times = eval_times,
  true_hazards = list(
    h12 = h12_true,
    h13 = h13_true,
    h23 = h23_true
  )
)

write_json(r_results, "r_results.json", pretty = TRUE, auto_unbox = TRUE)

cat("\n--- Results saved to r_results.json ---\n")
cat("\nBenchmark complete!\n")
