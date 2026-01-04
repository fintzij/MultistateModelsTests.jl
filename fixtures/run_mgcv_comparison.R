# =============================================================================
# R mgcv PAM Comparison Script
# =============================================================================

library(mgcv)
library(survival)
library(jsonlite)

setwd("/Users/fintzij/Library/CloudStorage/OneDrive-BristolMyersSquibb/Documents/Julia packages/MultistateModels.jl/MultistateModelsTests/scripts/../fixtures")

# Load Julia's data
data <- read.csv("comparison_data.csv")
hazard_grid <- read.csv("comparison_hazard_grid.csv")
julia_results <- fromJSON("julia_fit_results.json")

cat("\n=== R mgcv PAM Fitting ===\n")
cat(sprintf("n=%d, events=%d, censored=%d\n", 
            nrow(data), sum(data$status), sum(1 - data$status)))

# -----------------------------------------------------------------------------
# Transform to piecewise-exponential format
# -----------------------------------------------------------------------------

n_intervals <- 50
cut_points <- seq(0, max(data$time) * 1.01, length.out = n_intervals + 1)

ped_list <- lapply(1:nrow(data), function(i) {
  t_i <- data$time[i]
  d_i <- data$status[i]
  
  intervals <- which(cut_points[-length(cut_points)] < t_i)
  if (length(intervals) == 0) return(NULL)
  
  data.frame(
    id = i,
    tstart = cut_points[intervals],
    tend = pmin(cut_points[intervals + 1], t_i),
    interval = intervals,
    ped_status = c(rep(0, length(intervals) - 1), d_i),
    offset = log(pmin(cut_points[intervals + 1], t_i) - cut_points[intervals])
  )
})

ped <- do.call(rbind, ped_list)
ped$tend_mid <- (ped$tstart + ped$tend) / 2

cat(sprintf("PED rows: %d\n", nrow(ped)))

# -----------------------------------------------------------------------------
# Fit PAM with REML (standard) - match Julia's knot count
# -----------------------------------------------------------------------------

# Use k = number of Julia basis functions + 1 (for mgcv's constraint)
k_mgcv <- length(julia_results$spline_coeffs) + 1

cat(sprintf("\nFitting with k=%d basis functions (to match Julia's %d coefficients)...\n",
            k_mgcv, length(julia_results$spline_coeffs)))

cat("Fitting with REML...\n")
pam_reml <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "REML"
)

cat("Fitting with NCV (PIJCV)...\n")
pam_ncv <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "NCV"
)

# Also fit with GCV for comparison
cat("Fitting with GCV...\n")
pam_gcv <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "GCV.Cp"
)

# -----------------------------------------------------------------------------
# Extract fitted hazards at evaluation points
# -----------------------------------------------------------------------------

pred_data <- data.frame(
  tend_mid = hazard_grid$t,
  offset = 0
)

h_reml <- predict(pam_reml, newdata = pred_data, type = "response")
h_ncv <- predict(pam_ncv, newdata = pred_data, type = "response")
h_gcv <- predict(pam_gcv, newdata = pred_data, type = "response")

# -----------------------------------------------------------------------------
# Compare to truth
# -----------------------------------------------------------------------------

rmse_reml <- sqrt(mean((h_reml - hazard_grid$h_true)^2))
rmse_ncv <- sqrt(mean((h_ncv - hazard_grid$h_true)^2))
rmse_gcv <- sqrt(mean((h_gcv - hazard_grid$h_true)^2))
rmse_julia <- sqrt(mean((hazard_grid$h_julia - hazard_grid$h_true)^2))

cat(sprintf("\n=== RMSE vs True Hazard ===\n"))
cat(sprintf("Julia:     %.6f\n", rmse_julia))
cat(sprintf("mgcv REML: %.6f\n", rmse_reml))
cat(sprintf("mgcv NCV:  %.6f\n", rmse_ncv))
cat(sprintf("mgcv GCV:  %.6f\n", rmse_gcv))

cat(sprintf("\n=== Smoothing Parameters ===\n"))
cat(sprintf("REML: sp=%.4f, EDF=%.2f\n", pam_reml$sp[1], sum(pam_reml$edf)))
cat(sprintf("NCV:  sp=%.4f, EDF=%.2f\n", pam_ncv$sp[1], sum(pam_ncv$edf)))
cat(sprintf("GCV:  sp=%.4f, EDF=%.2f\n", pam_gcv$sp[1], sum(pam_gcv$edf)))

# -----------------------------------------------------------------------------
# Export results
# -----------------------------------------------------------------------------

results <- list(
  reml = list(
    sp = pam_reml$sp[1],
    edf = sum(pam_reml$edf),
    rmse = rmse_reml,
    fitted_hazard = as.vector(h_reml)
  ),
  ncv = list(
    sp = pam_ncv$sp[1],
    edf = sum(pam_ncv$edf),
    rmse = rmse_ncv,
    fitted_hazard = as.vector(h_ncv)
  ),
  gcv = list(
    sp = pam_gcv$sp[1],
    edf = sum(pam_gcv$edf),
    rmse = rmse_gcv,
    fitted_hazard = as.vector(h_gcv)
  ),
  julia = list(
    rmse = rmse_julia,
    fitted_hazard = hazard_grid$h_julia
  ),
  t_eval = hazard_grid$t,
  h_true = hazard_grid$h_true
)

write_json(results, "mgcv_fit_results.json", pretty = TRUE, auto_unbox = TRUE)
cat("\nResults exported to mgcv_fit_results.json\n")
