# Final Comparison: MultistateModels.jl Splines vs mgcv PAM
# Illness-Death Model Benchmark Results

library(jsonlite)

cat(paste(rep("=", 70), collapse=""), "\n")
cat("FINAL BENCHMARK COMPARISON\n")
cat("MultistateModels.jl Splines vs mgcv PAM: Illness-Death Model\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Load all results
julia_results <- fromJSON("illness_death_results.json")
julia_pred <- fromJSON("illness_death_julia_predictions.json")
mgcv_results <- fromJSON("illness_death_mgcv_results.json")
metadata <- fromJSON("illness_death_metadata.json")

cat("\n--- Model Setup ---\n")
cat("N subjects:", metadata$n_subjects, "\n")
cat("Max time:", metadata$max_time, "\n")
cat("\nTrue Parameters:\n")
cat("  h12 (Healthy→Illness): shape=", metadata$true_params$h12$shape, 
    ", rate=", metadata$true_params$h12$rate, "\n")
cat("  h13 (Healthy→Death):   shape=", metadata$true_params$h13$shape, 
    ", rate=", metadata$true_params$h13$rate, "\n")
cat("  h23 (Illness→Death):   shape=", metadata$true_params$h23$shape, 
    ", rate=", metadata$true_params$h23$rate, "\n")

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("HAZARD ESTIMATION ACCURACY (vs True)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

cat("\n                  h12         h13         h23\n")
cat("--------------------------------------------------\n")
cat(sprintf("Julia Splines:   %.5f     %.5f     %.5f\n", 
            julia_pred$rmse$hazard_h12, julia_pred$rmse$hazard_h13, julia_pred$rmse$hazard_h23))
cat(sprintf("mgcv PAM:        %.5f     %.5f     %.5f\n", 
            mgcv_results$rmse$hazard_h12, mgcv_results$rmse$hazard_h13, mgcv_results$rmse$hazard_h23))

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("STATE PREVALENCE ACCURACY (RMSE vs True)\n")
cat(paste(rep("=", 70), collapse=""), "\n")

# Helper function
rmse <- function(a, b) sqrt(mean((a - b)^2))

true_p1 <- julia_results$true$prevalence_healthy
true_p2 <- julia_results$true$prevalence_illness
true_p3 <- julia_results$true$prevalence_death

emp_p1 <- julia_results$empirical$prevalence_healthy
emp_p2 <- julia_results$empirical$prevalence_illness
emp_p3 <- julia_results$empirical$prevalence_death

julia_p1 <- julia_pred$prevalence$julia_healthy
julia_p2 <- julia_pred$prevalence$julia_illness
julia_p3 <- julia_pred$prevalence$julia_death

mgcv_p1 <- mgcv_results$prevalence$mgcv_healthy
mgcv_p2 <- mgcv_results$prevalence$mgcv_illness
mgcv_p3 <- mgcv_results$prevalence$mgcv_death

cat("\n                    P(Healthy)    P(Illness)    P(Death)\n")
cat("----------------------------------------------------------\n")
cat(sprintf("Empirical:          %.5f       %.5f       %.5f\n", 
            rmse(true_p1, emp_p1), rmse(true_p2, emp_p2), rmse(true_p3, emp_p3)))
cat(sprintf("Julia Splines:      %.5f       %.5f       %.5f\n", 
            rmse(true_p1, julia_p1), rmse(true_p2, julia_p2), rmse(true_p3, julia_p3)))
cat(sprintf("mgcv PAM:           %.5f       %.5f       %.5f\n", 
            rmse(true_p1, mgcv_p1), rmse(true_p2, mgcv_p2), rmse(true_p3, mgcv_p3)))

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("SAMPLE VALUES AT KEY TIMEPOINTS\n")
cat(paste(rep("=", 70), collapse=""), "\n")

eval_times <- julia_results$eval_times
idx <- c(1, 10, 20, 30, 40)  # t = 0.5, 5, 10, 15, 20

cat("\n--- P(Healthy) ---\n")
cat("Time\tTrue\tJulia\tmgcv\tEmpirical\n")
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                eval_times[i], true_p1[i], julia_p1[i], mgcv_p1[i], emp_p1[i]))
  }
}

cat("\n--- P(Illness) ---\n")
cat("Time\tTrue\tJulia\tmgcv\tEmpirical\n")
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                eval_times[i], true_p2[i], julia_p2[i], mgcv_p2[i], emp_p2[i]))
  }
}

cat("\n--- P(Death) ---\n")
cat("Time\tTrue\tJulia\tmgcv\tEmpirical\n")
for (i in idx) {
  if (i <= length(eval_times)) {
    cat(sprintf("%.1f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                eval_times[i], true_p3[i], julia_p3[i], mgcv_p3[i], emp_p3[i]))
  }
}

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("FIT PERFORMANCE\n")
cat(paste(rep("=", 70), collapse=""), "\n")

cat("\nJulia MultistateModels.jl:\n")
cat("  Fit time:", round(julia_results$julia_fit$fit_time_seconds, 1), "seconds\n")
cat("  Interior knots:", paste(round(julia_results$julia_fit$interior_knots, 2), collapse=", "), "\n")
cat("  Joint likelihood estimation (all transitions simultaneously)\n")

cat("\nmgcv PAM:\n")
cat("  Fit time:", round(mgcv_results$fit_info$fit_time_seconds, 1), "seconds\n")
cat("  EDF h12:", round(mgcv_results$fit_info$edf_h12, 2), "\n")
cat("  EDF h13:", round(mgcv_results$fit_info$edf_h13, 2), "\n")
cat("  EDF h23:", round(mgcv_results$fit_info$edf_h23, 2), "\n")
cat("  3 separate gam() calls (independent estimation)\n")

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 70), collapse=""), "\n")

cat("\nHazard RMSE comparison:\n")
cat("  Julia splines: h12=", round(julia_pred$rmse$hazard_h12, 4),
    ", h13=", round(julia_pred$rmse$hazard_h13, 4),
    ", h23=", round(julia_pred$rmse$hazard_h23, 4), "\n")
cat("  mgcv PAM:      h12=", round(mgcv_results$rmse$hazard_h12, 4),
    ", h13=", round(mgcv_results$rmse$hazard_h13, 4),
    ", h23=", round(mgcv_results$rmse$hazard_h23, 4), "\n")

cat("\nPrevalence RMSE comparison:\n")
cat("  Julia splines: P1=", round(rmse(true_p1, julia_p1), 4),
    ", P2=", round(rmse(true_p2, julia_p2), 4),
    ", P3=", round(rmse(true_p3, julia_p3), 4), "\n")
cat("  mgcv PAM:      P1=", round(rmse(true_p1, mgcv_p1), 4),
    ", P2=", round(rmse(true_p2, mgcv_p2), 4),
    ", P3=", round(rmse(true_p3, mgcv_p3), 4), "\n")

# Winner determination
julia_wins <- sum(c(
  julia_pred$rmse$hazard_h12 < mgcv_results$rmse$hazard_h12,
  julia_pred$rmse$hazard_h13 < mgcv_results$rmse$hazard_h13,
  julia_pred$rmse$hazard_h23 < mgcv_results$rmse$hazard_h23,
  rmse(true_p1, julia_p1) < rmse(true_p1, mgcv_p1),
  rmse(true_p2, julia_p2) < rmse(true_p2, mgcv_p2),
  rmse(true_p3, julia_p3) < rmse(true_p3, mgcv_p3)
))
mgcv_wins <- 6 - julia_wins

cat("\nComparison summary: Julia wins ", julia_wins, "/6 metrics, mgcv wins ", mgcv_wins, "/6 metrics\n")

cat("\nConclusions:\n")
if (julia_wins > mgcv_wins) {
  cat("  - MultistateModels.jl splines outperform mgcv PAM on this illness-death benchmark\n")
} else if (mgcv_wins > julia_wins) {
  cat("  - mgcv PAM outperforms MultistateModels.jl splines on this illness-death benchmark\n")
} else {
  cat("  - Both approaches perform similarly on this illness-death benchmark\n")
}
cat("  - Julia benefits from joint estimation (all transitions fit simultaneously)\n")
cat("  - mgcv PAM fits each transition independently, ignoring cross-transition structure\n")

cat("\n", paste(rep("=", 70), collapse=""), "\n")
cat("Comparison Complete!\n")
cat(paste(rep("=", 70), collapse=""), "\n")
