# =============================================================================
# Generate Reference Data in R for Julia Comparison
# =============================================================================
#
# Generate survival data with known Weibull hazard, fit with mgcv PAM,
# then export for Julia to fit and compare.
#
# =============================================================================

library(mgcv)
library(survival)
library(jsonlite)

set.seed(2024)

# =============================================================================
# 1. Generate survival data with Weibull hazard
# =============================================================================

cat("=== Generating Weibull Survival Data ===\n")

# Weibull parameters (flexsurv convention)
weibull_shape <- 1.5
weibull_scale <- 5.0

# True hazard: h(t) = (shape/scale) * (t/scale)^(shape-1)
true_hazard <- function(t) {
  (weibull_shape / weibull_scale) * (t / weibull_scale)^(weibull_shape - 1)
}

# Generate event times
n <- 500
U <- runif(n)
# For Weibull: T = scale * (-log(U))^(1/shape)
event_times <- weibull_scale * (-log(U))^(1/weibull_shape)

# Administrative censoring at t=15
cens_time <- 15
obs_times <- pmin(event_times, cens_time)
status <- as.integer(event_times <= cens_time)

cat(sprintf("Sample size: %d\n", n))
cat(sprintf("Events: %d, Censored: %d\n", sum(status), sum(1 - status)))
cat(sprintf("Time range: [%.3f, %.3f]\n", min(obs_times), max(obs_times)))
cat(sprintf("True hazard: Weibull(shape=%.1f, scale=%.1f)\n", weibull_shape, weibull_scale))

# Create data frame
surv_data <- data.frame(
  id = 1:n,
  time = obs_times,
  status = status
)

# =============================================================================
# 2. Transform to PED format and fit PAM
# =============================================================================

cat("\n=== Fitting mgcv PAM ===\n")

n_intervals <- 50
cut_points <- seq(0, max(obs_times) * 1.01, length.out = n_intervals + 1)

ped_list <- lapply(1:n, function(i) {
  t_i <- obs_times[i]
  d_i <- status[i]
  
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

# Fit PAM with REML
pam_reml <- gam(
  ped_status ~ s(tend_mid, k = 10, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "REML"
)

cat(sprintf("REML: sp=%.4f, EDF=%.2f\n", pam_reml$sp[1], sum(pam_reml$edf)))

# =============================================================================
# 3. Evaluate and export
# =============================================================================

cat("\n=== Evaluating fit ===\n")

# Evaluation grid
t_eval <- seq(0.5, max(obs_times) * 0.95, length.out = 50)
h_true <- sapply(t_eval, true_hazard)

# Predict from PAM
pred_data <- data.frame(tend_mid = t_eval, offset = 0)
h_mgcv <- predict(pam_reml, newdata = pred_data, type = "response")

rmse_mgcv <- sqrt(mean((h_mgcv - h_true)^2))
cat(sprintf("mgcv RMSE vs truth: %.6f\n", rmse_mgcv))

# Export everything
results <- list(
  config = list(
    n = n,
    weibull_shape = weibull_shape,
    weibull_scale = weibull_scale,
    cens_time = cens_time,
    n_events = sum(status),
    n_censored = sum(1 - status)
  ),
  data = list(
    id = surv_data$id,
    time = surv_data$time,
    status = surv_data$status
  ),
  mgcv = list(
    sp = pam_reml$sp[1],
    edf = sum(pam_reml$edf),
    rmse = rmse_mgcv,
    fitted_hazard = as.vector(h_mgcv)
  ),
  evaluation = list(
    t_eval = t_eval,
    h_true = h_true
  )
)

write_json(results, "r_generated_data.json", pretty = TRUE, auto_unbox = TRUE)
write.csv(surv_data, "r_generated_data.csv", row.names = FALSE)

cat("\nExported to r_generated_data.json and r_generated_data.csv\n")

# =============================================================================
# Summary
# =============================================================================

cat("\n")
cat(strrep("=", 60), "\n")
cat("SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Data: n=%d, events=%d\n", n, sum(status)))
cat(sprintf("True hazard: Weibull(shape=%.1f, scale=%.1f)\n", weibull_shape, weibull_scale))
cat(sprintf("mgcv PAM RMSE: %.6f\n", rmse_mgcv))
cat(strrep("=", 60), "\n")
