# =============================================================================
# flexmsm Benchmark: Compare smoothing parameter selection methods
# =============================================================================
#
# This script attempts to fit a survival model using flexmsm's PERF and EFS 
# methods and exports results for comparison with MultistateModels.jl.
#
# NOTE: flexmsm is designed for panel-observed multistate models with multiple
# observation times per subject. Simple survival data (one observation per 
# subject) may not work correctly due to how the package infers state pairs.
#
# Reference: Eletti, Marra & Radice (2024). arXiv:2312.05345v4
# =============================================================================

library(flexmsm)
library(jsonlite)
library(survival)
library(mgcv)

cat(paste(rep("=", 70), collapse=""), "\n")
cat("flexmsm Benchmark: PERF and EFS Smoothing Selection\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# ============================================================================
# Load Data from Julia GCV reference
# ============================================================================

script_dir <- tryCatch({
  dirname(rstudioapi::getSourceEditorContext()$path)
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(sub("--file=", "", file_arg))
  } else {
    getwd()
  }
})

csv_path <- file.path(script_dir, "gcv_reference_data.csv")
julia_path <- file.path(script_dir, "gcv_reference_julia.json")

if (!file.exists(csv_path)) {
  stop("Data file not found. Run generate_gcv_reference.jl first.")
}

dat <- read.csv(csv_path)
julia_ref <- fromJSON(julia_path)

cat("\nData loaded:\n")
cat("  N:", nrow(dat), "\n")
cat("  Events:", sum(dat$status), "\n")
cat("  Censored:", sum(1 - dat$status), "\n")
cat("  True hazard: Weibull(shape=", julia_ref$config$true_shape,
    ", rate=", julia_ref$config$true_rate, ")\n")

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Fitting models\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# ============================================================================
# flexmsm - Note: Currently fails on simple survival data
# ============================================================================

cat("\n--- flexmsm PERF (known issue with simple survival data) ---\n")
cat("  Status: Skipped - flexmsm requires panel-observed multistate data\n")
cat("  For a proper comparison, use illness-death or multi-state data\n")

perf_results <- list(
  method = "perf",
  note = "flexmsm not tested - requires panel-observed multistate data",
  converged = FALSE
)

efs_results <- list(
  method = "efs",
  note = "flexmsm not tested - requires panel-observed multistate data",
  converged = FALSE
)

# ============================================================================
# mgcv GCV - Poisson PAM approach (working baseline)
# ============================================================================

cat("\n--- mgcv GCV (Poisson PAM approach) ---\n")

n_intervals <- 20
breaks <- seq(0, max(dat$time) * 1.01, length.out = n_intervals + 1)

ped_rows <- list()
for (i in 1:nrow(dat)) {
  t_end <- dat$time[i]
  is_event <- dat$status[i]
  
  for (j in 1:(length(breaks) - 1)) {
    int_start <- breaks[j]
    int_end <- breaks[j + 1]
    
    if (int_end > t_end && int_start >= t_end) break
    
    risk_end <- min(int_end, t_end)
    offset_val <- log(risk_end - int_start + 1e-10)
    event_in_interval <- is_event && (t_end <= int_end) && (t_end > int_start)
    
    ped_rows[[length(ped_rows) + 1]] <- data.frame(
      id = i,
      interval = j,
      t_mid = (int_start + risk_end) / 2,
      offset = offset_val,
      event = as.integer(event_in_interval)
    )
  }
}

ped <- do.call(rbind, ped_rows)

tryCatch({
  fit_gcv <- gam(
    event ~ s(t_mid, bs = "ps", k = 5),
    family = poisson(),
    data = ped,
    offset = offset,
    method = "GCV.Cp"
  )
  
  gcv_sp <- fit_gcv$sp
  gcv_edf <- sum(fit_gcv$edf)
  
  cat("  Optimal sp:", round(gcv_sp, 6), "\n")
  cat("  EDF:", round(gcv_edf, 2), "\n")
  
  mgcv_gcv_results <- list(
    method = "gcv",
    sp = as.vector(gcv_sp),
    edf = as.numeric(gcv_edf),
    converged = TRUE
  )
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
  mgcv_gcv_results <<- list(method = "gcv", error = conditionMessage(e), converged = FALSE)
})

# ============================================================================
# mgcv REML - for comparison with EFS
# ============================================================================

cat("\n--- mgcv REML (for EFS comparison) ---\n")

tryCatch({
  fit_reml <- gam(
    event ~ s(t_mid, bs = "ps", k = 5),
    family = poisson(),
    data = ped,
    offset = offset,
    method = "REML"
  )
  
  reml_sp <- fit_reml$sp
  reml_edf <- sum(fit_reml$edf)
  
  cat("  Optimal sp:", round(reml_sp, 6), "\n")
  cat("  EDF:", round(reml_edf, 2), "\n")
  
  mgcv_reml_results <- list(
    method = "reml",
    sp = as.vector(reml_sp),
    edf = as.numeric(reml_edf),
    converged = TRUE
  )
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
  mgcv_reml_results <<- list(method = "reml", error = conditionMessage(e), converged = FALSE)
})

# ============================================================================
# Export results
# ============================================================================

output <- list(
  description = "Smoothing parameter selection benchmark for MultistateModels.jl",
  date_generated = as.character(Sys.time()),
  r_version = R.version.string,
  flexmsm_version = as.character(packageVersion("flexmsm")),
  mgcv_version = as.character(packageVersion("mgcv")),
  data_config = julia_ref$config,
  julia_optimal = julia_ref$optimal,
  flexmsm_perf = perf_results,
  flexmsm_efs = efs_results,
  mgcv_gcv = if (exists("mgcv_gcv_results")) mgcv_gcv_results else list(error = "not computed"),
  mgcv_reml = if (exists("mgcv_reml_results")) mgcv_reml_results else list(error = "not computed"),
  notes = list(
    "flexmsm requires panel-observed multistate data (multiple observation times per subject)",
    "For proper flexmsm comparison, use illness-death model with intermittent observations",
    "mgcv GCV and REML use Poisson PAM (piecewise-exponential additive model) approach",
    "Julia MultistateModels.jl uses exact likelihood, so optimal lambda may differ"
  )
)

output_path <- file.path(script_dir, "flexmsm_benchmark_results.json")
write_json(output, output_path, pretty = TRUE, auto_unbox = TRUE)
cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Results written to:", output_path, "\n")
cat(paste(rep("=", 70), collapse=""), "\n")
