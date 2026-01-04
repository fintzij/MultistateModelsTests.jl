# =============================================================================
# Generate PIJCV/NCV Reference Data from R mgcv
# =============================================================================
#
# This script generates reference data for validating MultistateModels.jl's
# PIJCV implementation against R mgcv's NCV (Neighborhood Cross-Validation).
#
# PIJCV (Prediction-based Integrated Jackknife Cross-Validation) is called
# NCV in mgcv. See Wood (2024) arXiv:2404.16490 for details.
#
# Usage:
#   Rscript generate_pijcv_reference.R
#
# Output:
#   pijcv_reference.json - Reference data for Julia tests
#
# =============================================================================

library(mgcv)
library(jsonlite)

cat(strrep("=", 70), "\n")
cat("Generating PIJCV/NCV Reference Data\n")
cat(strrep("=", 70), "\n\n")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

set.seed(12345)
n <- 200
sigma <- 0.3  # Noise standard deviation

# True function
f_true <- function(x) sin(2 * pi * x)

# Spline settings (matching Julia defaults)
k <- 10        # Number of basis functions
m <- 2         # Penalty order (second derivative)
bs <- "ps"     # P-spline basis

# Evaluation grid
n_eval <- 50
x_eval <- seq(0, 1, length.out = n_eval)

# -----------------------------------------------------------------------------
# Generate Data
# -----------------------------------------------------------------------------

cat("Generating synthetic data...\n")
x <- runif(n, 0, 1)
f_x <- f_true(x)
y <- f_x + rnorm(n, 0, sigma)

cat(sprintf("  n = %d observations\n", n))
cat(sprintf("  sigma = %.2f noise SD\n", sigma))
cat(sprintf("  x range: [%.3f, %.3f]\n", min(x), max(x)))

# -----------------------------------------------------------------------------
# Fit GAM with NCV (PIJCV)
# -----------------------------------------------------------------------------

cat("\nFitting GAM with NCV (PIJCV)...\n")

fit_ncv <- gam(y ~ s(x, k = k, bs = bs, m = m), method = "NCV")

lambda_ncv <- fit_ncv$sp
edf_ncv <- sum(fit_ncv$edf)
gcv_ncv <- fit_ncv$gcv.ubre

cat(sprintf("  NCV lambda: %.6f\n", lambda_ncv))
cat(sprintf("  NCV EDF: %.3f\n", edf_ncv))

# -----------------------------------------------------------------------------
# Fit GAM with REML for comparison
# -----------------------------------------------------------------------------

cat("\nFitting GAM with REML for comparison...\n")

fit_reml <- gam(y ~ s(x, k = k, bs = bs, m = m), method = "REML")

lambda_reml <- fit_reml$sp
edf_reml <- sum(fit_reml$edf)

cat(sprintf("  REML lambda: %.6f\n", lambda_reml))
cat(sprintf("  REML EDF: %.3f\n", edf_reml))
cat(sprintf("  Lambda ratio (NCV/REML): %.3f\n", lambda_ncv / lambda_reml))

# -----------------------------------------------------------------------------
# Extract Penalty Matrix
# -----------------------------------------------------------------------------

cat("\nExtracting penalty matrix...\n")

# Get smooth object
sm <- fit_ncv$smooth[[1]]

# Penalty matrix
S <- sm$S[[1]]

cat(sprintf("  Penalty matrix dimension: %d x %d\n", nrow(S), ncol(S)))

# Knots used by mgcv
# Note: mgcv stores knots differently - we need the full knot vector
knots_mgcv <- sm$knots

cat(sprintf("  Number of knots: %d\n", length(knots_mgcv)))

# -----------------------------------------------------------------------------
# Predictions at Evaluation Points
# -----------------------------------------------------------------------------

cat("\nComputing predictions at evaluation points...\n")

pred_ncv <- predict(fit_ncv, newdata = data.frame(x = x_eval))
pred_reml <- predict(fit_reml, newdata = data.frame(x = x_eval))
f_true_eval <- f_true(x_eval)

# Compute RMSE vs truth
rmse_ncv <- sqrt(mean((pred_ncv - f_true_eval)^2))
rmse_reml <- sqrt(mean((pred_reml - f_true_eval)^2))

cat(sprintf("  RMSE vs truth (NCV):  %.4f\n", rmse_ncv))
cat(sprintf("  RMSE vs truth (REML): %.4f\n", rmse_reml))

# -----------------------------------------------------------------------------
# Extract Basis Matrix for Comparison
# -----------------------------------------------------------------------------

cat("\nExtracting basis matrix at data points...\n")

# Predict with type="lpmatrix" to get design matrix
X <- predict(fit_ncv, type = "lpmatrix")

# The smooth part (excluding intercept)
X_smooth <- X[, 2:ncol(X)]

cat(sprintf("  Full design matrix: %d x %d\n", nrow(X), ncol(X)))
cat(sprintf("  Smooth design matrix: %d x %d\n", nrow(X_smooth), ncol(X_smooth)))

# Also get design matrix at evaluation points
X_eval <- predict(fit_ncv, newdata = data.frame(x = x_eval), type = "lpmatrix")
X_smooth_eval <- X_eval[, 2:ncol(X_eval)]

# -----------------------------------------------------------------------------
# Assemble Reference Data
# -----------------------------------------------------------------------------

cat("\nAssembling reference data...\n")

reference <- list(
  # Metadata
  description = "PIJCV/NCV reference data from R mgcv",
  date_generated = as.character(Sys.time()),
  mgcv_version = as.character(packageVersion("mgcv")),
  r_version = R.version.string,
  
  # Configuration
  config = list(
    seed = 12345,
    n = n,
    sigma = sigma,
    k = k,
    m = m,
    bs = bs,
    n_eval = n_eval
  ),
  
  # Raw data
  data = list(
    x = x,
    y = y,
    f_true = f_x
  ),
  
  # NCV (PIJCV) results
  ncv = list(
    lambda = lambda_ncv,
    edf = edf_ncv,
    gcv_score = gcv_ncv
  ),
  
  # REML results for comparison
  reml = list(
    lambda = lambda_reml,
    edf = edf_reml
  ),
  
  # Predictions
  predictions = list(
    x_eval = x_eval,
    f_true_eval = f_true_eval,
    pred_ncv = as.vector(pred_ncv),
    pred_reml = as.vector(pred_reml),
    rmse_ncv = rmse_ncv,
    rmse_reml = rmse_reml
  ),
  
  # Penalty matrix (as flat vector for JSON, row-major)
  penalty_matrix = list(
    nrow = nrow(S),
    ncol = ncol(S),
    values = as.vector(t(S))  # Row-major for easier Julia reshape
  ),
  
  # Knot vector
  knots = knots_mgcv,
  
  # Basis matrices (for detailed comparison)
  basis_at_data = list(
    nrow = nrow(X_smooth),
    ncol = ncol(X_smooth),
    values = as.vector(t(X_smooth))
  ),
  
  basis_at_eval = list(
    nrow = nrow(X_smooth_eval),
    ncol = ncol(X_smooth_eval),
    values = as.vector(t(X_smooth_eval))
  ),
  
  # Fitted coefficients
  coefficients = list(
    intercept = coef(fit_ncv)[1],
    smooth = coef(fit_ncv)[2:length(coef(fit_ncv))]
  )
)

# -----------------------------------------------------------------------------
# Save Reference Data
# -----------------------------------------------------------------------------

output_file <- "pijcv_reference.json"
cat(sprintf("\nSaving to %s...\n", output_file))

write_json(reference, output_file, pretty = TRUE, digits = 15, auto_unbox = TRUE)

cat("\nDone!\n")
cat(strrep("=", 70), "\n")

# -----------------------------------------------------------------------------
# Summary Statistics for Documentation
# -----------------------------------------------------------------------------

cat("\n")
cat("SUMMARY FOR JULIA TEST IMPLEMENTATION\n")
cat(strrep("-", 40), "\n")
cat(sprintf("Expected NCV lambda:  %.10f\n", lambda_ncv))
cat(sprintf("Expected NCV EDF:     %.6f\n", edf_ncv))
cat(sprintf("Lambda ratio NCV/REML: %.3f\n", lambda_ncv / lambda_reml))
cat("\n")
cat("Acceptance criteria suggestions:\n")
cat("  - Lambda match: within 10%% relative error\n")
cat("  - EDF match: within ±0.5\n")
cat("  - Penalty matrix: element-wise rel error < 1e-6\n")
cat("  - Fitted curve RMSE: < 0.05 between Julia and R\n")
