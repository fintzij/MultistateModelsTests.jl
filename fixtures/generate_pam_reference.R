# =============================================================================
# PIJCV/NCV Reference: Piecewise Additive Model (PAM) Survival Context
# =============================================================================
#
# Generate reference data using mgcv's NCV method in a survival analysis context
# via piecewise-exponential additive models (PAMMs).
#
# This provides a direct comparison with MultistateModels.jl's penalized splines:
# - Same context: exact/right-censored survival data
# - Same method: mgcv's NCV (= PIJCV per Wood 2024)
# - Same machinery: Poisson likelihood on expanded data
#
# Reference: 
#   - Wood (2024) "NCV for Smoothness Selection" arXiv:2404.16490
#   - Bender et al. (2018) "A generalized additive model approach to 
#     time-to-event analysis" Statistical Modelling
#
# =============================================================================

library(mgcv)
library(survival)
library(jsonlite)

set.seed(42)

# =============================================================================
# 1. Generate survival data with smooth baseline hazard
# =============================================================================

cat("=== Generating survival data with smooth hazard ===\n")

n <- 500  # Sample size

# True baseline hazard: h0(t) = 0.5 * (1 + 0.5*sin(2*pi*t/10))
# This gives a sinusoidal hazard over [0, 10]
true_hazard <- function(t) {
  0.5 * (1 + 0.5 * sin(2 * pi * t / 10))
}

# Cumulative hazard
true_cumhaz <- function(t) {
  # Integral of h0 from 0 to t
  # = 0.5 * t - 0.5 * (10/(2*pi)) * (cos(2*pi*t/10) - 1)
  # = 0.5 * t + (10/(4*pi)) * (1 - cos(2*pi*t/10))
  0.5 * t + (10/(4*pi)) * (1 - cos(2 * pi * t / 10))
}

# Inverse cumulative hazard for sampling (numerical)
inv_cumhaz <- function(u) {
  sapply(u, function(ui) {
    uniroot(function(t) true_cumhaz(t) - ui, c(0, 100), extendInt = "yes")$root
  })
}

# Generate event times via inverse transform sampling
U <- runif(n)
event_times <- inv_cumhaz(-log(U))

# Administrative censoring at t=12
cens_time <- 12
obs_times <- pmin(event_times, cens_time)
status <- as.integer(event_times <= cens_time)

cat(sprintf("Events: %d, Censored: %d\n", sum(status), sum(1 - status)))
cat(sprintf("Time range: [%.3f, %.3f]\n", min(obs_times), max(obs_times)))

# Create survival data frame
surv_data <- data.frame(
  id = 1:n,
  time = obs_times,
  status = status
)

# =============================================================================
# 2. Transform to piecewise-exponential data (PED)
# =============================================================================

cat("\n=== Transforming to piecewise-exponential format ===\n")

# Create cut points for the piecewise-exponential model
# More cuts = better approximation of continuous hazard
n_intervals <- 50
cut_points <- seq(0, max(obs_times) * 1.01, length.out = n_intervals + 1)

# Manual PED transformation
# For each subject, expand into intervals
ped_list <- lapply(1:n, function(i) {
  t_i <- obs_times[i]
  d_i <- status[i]
  
  # Find which intervals this subject contributes to
  intervals <- which(cut_points[-length(cut_points)] < t_i)
  
  if (length(intervals) == 0) return(NULL)
  
  # Create one row per interval
  data.frame(
    id = i,
    tstart = cut_points[intervals],
    tend = pmin(cut_points[intervals + 1], t_i),
    interval = intervals,
    ped_status = c(rep(0, length(intervals) - 1), d_i),  # Event only in last interval
    offset = log(pmin(cut_points[intervals + 1], t_i) - cut_points[intervals])
  )
})

ped <- do.call(rbind, ped_list)
ped$tend_mid <- (ped$tstart + ped$tend) / 2  # Midpoint for smooth evaluation

cat(sprintf("PED rows: %d (expanded from %d subjects)\n", nrow(ped), n))

# =============================================================================
# 3. Fit PAM with NCV smoothing selection
# =============================================================================

cat("\n=== Fitting PAM with NCV (PIJCV) smoothing selection ===\n")

# Fit with NCV - this is the PIJCV algorithm
# Note: NCV may be slow, so we use a reasonable basis dimension
pam_ncv <- gam(
  ped_status ~ s(tend_mid, k = 15, bs = "ps"),  # P-spline basis
  family = poisson(),
  offset = offset,
  data = ped,
  method = "NCV"  # Neighborhood Cross-Validation = PIJCV
)

cat("\n--- NCV fit summary ---\n")
print(summary(pam_ncv))

# Also fit with REML for comparison
pam_reml <- gam(
  ped_status ~ s(tend_mid, k = 15, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "REML"
)

cat("\n--- REML fit summary ---\n")
print(summary(pam_reml))

# =============================================================================
# 4. Extract smoothing parameters and model details
# =============================================================================

cat("\n=== Extracting model details ===\n")

# Get smoothing parameters (lambda)
sp_ncv <- pam_ncv$sp[1]
sp_reml <- pam_reml$sp[1]

cat(sprintf("NCV smoothing parameter (sp): %.6f\n", sp_ncv))
cat(sprintf("REML smoothing parameter (sp): %.6f\n", sp_reml))
cat(sprintf("Ratio (NCV/REML): %.3f\n", sp_ncv / sp_reml))

# Get effective degrees of freedom
edf_ncv <- sum(pam_ncv$edf)
edf_reml <- sum(pam_reml$edf)

cat(sprintf("NCV EDF: %.3f\n", edf_ncv))
cat(sprintf("REML EDF: %.3f\n", edf_reml))

# =============================================================================
# 5. Extract penalty matrix structure
# =============================================================================

cat("\n=== Extracting penalty matrix ===\n")

# Get the smooth object
smooth_obj <- pam_ncv$smooth[[1]]

# Penalty matrix (S)
S <- smooth_obj$S[[1]]
cat(sprintf("Penalty matrix dimension: %d x %d\n", nrow(S), ncol(S)))

# Basis matrix at data points
X <- predict(pam_ncv, type = "lpmatrix")
# Remove intercept column
X_smooth <- X[, grepl("s\\(tend_mid\\)", colnames(X))]
cat(sprintf("Basis matrix dimension: %d x %d\n", nrow(X_smooth), ncol(X_smooth)))

# Get knots
knots <- smooth_obj$knots

# =============================================================================
# 6. Evaluate fitted hazards vs truth
# =============================================================================

cat("\n=== Evaluating fit quality ===\n")

# Create evaluation grid
t_eval <- seq(0.1, 11, length.out = 100)

# True log-hazard
log_h0_true <- log(true_hazard(t_eval))

# Predicted log-hazard from models
# Need to create prediction data with unit offset
pred_data <- data.frame(
  tend_mid = t_eval,
  offset = 0  # Log(1) = 0, so we get log(hazard) directly
)

log_h0_ncv <- predict(pam_ncv, newdata = pred_data, type = "response")
log_h0_reml <- predict(pam_reml, newdata = pred_data, type = "response")

# Actually, predict with type="response" gives hazard rate, not log-hazard
# For Poisson with log link: E[Y] = exp(Xβ + offset), so response = hazard
h0_ncv <- predict(pam_ncv, newdata = pred_data, type = "response")
h0_reml <- predict(pam_reml, newdata = pred_data, type = "response")
h0_true <- true_hazard(t_eval)

# RMSE
rmse_ncv <- sqrt(mean((h0_ncv - h0_true)^2))
rmse_reml <- sqrt(mean((h0_reml - h0_true)^2))

cat(sprintf("RMSE vs true hazard (NCV): %.6f\n", rmse_ncv))
cat(sprintf("RMSE vs true hazard (REML): %.6f\n", rmse_reml))

# Relative RMSE
rel_rmse_ncv <- rmse_ncv / mean(h0_true)
rel_rmse_reml <- rmse_reml / mean(h0_true)

cat(sprintf("Relative RMSE (NCV): %.4f (%.1f%%)\n", rel_rmse_ncv, rel_rmse_ncv * 100))
cat(sprintf("Relative RMSE (REML): %.4f (%.1f%%)\n", rel_rmse_reml, rel_rmse_reml * 100))

# =============================================================================
# 7. Export reference data to JSON
# =============================================================================

cat("\n=== Exporting to JSON ===\n")

reference_data <- list(
  description = "PAM/PAMM reference data for PIJCV comparison with MultistateModels.jl",
  date_generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  mgcv_version = as.character(packageVersion("mgcv")),
  r_version = R.version.string,
  
  config = list(
    n = n,
    n_intervals = n_intervals,
    k = 15,  # Basis dimension
    m = 2,   # Penalty order (default for ps)
    bs = "ps",
    cens_time = cens_time,
    true_hazard = "h0(t) = 0.5 * (1 + 0.5*sin(2*pi*t/10))"
  ),
  
  data = list(
    time = surv_data$time,
    status = surv_data$status
  ),
  
  ncv = list(
    sp = sp_ncv,
    edf = edf_ncv,
    deviance = pam_ncv$deviance,
    null_deviance = pam_ncv$null.deviance,
    aic = AIC(pam_ncv)
  ),
  
  reml = list(
    sp = sp_reml,
    edf = edf_reml,
    deviance = pam_reml$deviance,
    null_deviance = pam_reml$null.deviance,
    aic = AIC(pam_reml)
  ),
  
  fit_quality = list(
    rmse_ncv = rmse_ncv,
    rmse_reml = rmse_reml,
    rel_rmse_ncv = rel_rmse_ncv,
    rel_rmse_reml = rel_rmse_reml
  ),
  
  predictions = list(
    t_eval = t_eval,
    h0_true = h0_true,
    h0_ncv = as.vector(h0_ncv),
    h0_reml = as.vector(h0_reml)
  ),
  
  penalty_matrix = list(
    nrow = nrow(S),
    ncol = ncol(S),
    values = as.vector(S)
  ),
  
  knots = knots,
  
  coefficients = list(
    intercept_ncv = coef(pam_ncv)[1],
    smooth_ncv = coef(pam_ncv)[-1],
    intercept_reml = coef(pam_reml)[1],
    smooth_reml = coef(pam_reml)[-1]
  )
)

# Write to JSON
output_path <- "pam_pijcv_reference.json"
write_json(reference_data, output_path, pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("Reference data written to: %s\n", output_path))
cat(sprintf("File size: %.1f KB\n", file.size(output_path) / 1024))

# =============================================================================
# 8. Summary
# =============================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("SUMMARY: PAM-based PIJCV Reference Data\n")
cat(strrep("=", 70), "\n")
cat(sprintf("Sample size: %d subjects, %d events\n", n, sum(status)))
cat(sprintf("Model: Poisson GAM with P-spline basis (k=15)\n"))
cat(sprintf("\nNCV (PIJCV) results:\n"))
cat(sprintf("  Smoothing parameter: %.4f\n", sp_ncv))
cat(sprintf("  Effective df: %.2f\n", edf_ncv))
cat(sprintf("  Hazard RMSE: %.6f (%.1f%% relative)\n", rmse_ncv, rel_rmse_ncv * 100))
cat(sprintf("\nREML results:\n"))
cat(sprintf("  Smoothing parameter: %.4f\n", sp_reml))
cat(sprintf("  Effective df: %.2f\n", edf_reml))
cat(sprintf("  Hazard RMSE: %.6f (%.1f%% relative)\n", rmse_reml, rel_rmse_reml * 100))
cat(sprintf("\nSmoothing parameter ratio (NCV/REML): %.3f\n", sp_ncv / sp_reml))
cat(strrep("=", 70), "\n")
