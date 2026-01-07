# =============================================================================
# Penalized Spline Benchmark: Visualization
# =============================================================================
#
# Creates comparison plots of fitted hazard functions across:
# - mgcv (R)
# - flexsurv (R)
# - MultistateModels.jl
#
# Requires: CairoMakie, JSON, DataFrames
#
# Run with: julia --project=../../.. visualize_comparison.jl
# =============================================================================

using CairoMakie
using JSON
using DataFrames
using Statistics
using Printf

const OUTPUT_DIR = @__DIR__

CairoMakie.activate!(type = "png", px_per_unit = 2.0)

function load_results()
    # Load metadata
    metadata = JSON.parsefile(joinpath(OUTPUT_DIR, "benchmark_metadata.json"))
    eval_times = Vector{Float64}(metadata["eval_times"])
    
    # Load true hazards
    h12_true = Vector{Float64}(metadata["true_hazards"]["h12"])
    h13_true = Vector{Float64}(metadata["true_hazards"]["h13"])
    h23_true = Vector{Float64}(metadata["true_hazards"]["h23"])
    
    # Load Julia results
    julia_results = try
        JSON.parsefile(joinpath(OUTPUT_DIR, "julia_results.json"))
    catch e
        @warn "Julia results not found" exception=e
        nothing
    end
    
    # Load R results
    r_results = try
        JSON.parsefile(joinpath(OUTPUT_DIR, "r_results.json"))
    catch e
        @warn "R results not found" exception=e
        nothing
    end
    
    return (
        eval_times = eval_times,
        true_haz = (h12 = h12_true, h13 = h13_true, h23 = h23_true),
        julia = julia_results,
        r = r_results,
        metadata = metadata
    )
end

function create_hazard_comparison_plot(results)
    fig = Figure(size = (1200, 900))
    
    t = results.eval_times
    
    # Color scheme
    true_color = :black
    julia_color = :blue
    mgcv_color = :red
    flexsurv_color = :green
    
    # h12 plot
    ax1 = Axis(fig[1, 1],
               title = "Transition 1→2 (Healthy → Illness)",
               xlabel = "Time",
               ylabel = "Hazard h(t)",
               titlesize = 14)
    
    lines!(ax1, t, results.true_haz.h12, color = true_color, linewidth = 2, 
           label = "True: 0.3√t")
    
    if !isnothing(results.julia)
        h12_julia = Vector{Float64}(results.julia["hazards"]["h12"])
        lines!(ax1, t, h12_julia, color = julia_color, linewidth = 2, 
               linestyle = :dash, label = "Julia (PIJCV)")
    end
    
    if !isnothing(results.r)
        h12_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h12"])
        h12_flexsurv = Vector{Float64}(results.r["flexsurv"]["hazards"]["h12"])
        lines!(ax1, t, h12_mgcv, color = mgcv_color, linewidth = 2, 
               linestyle = :dot, label = "mgcv (NCV)")
        lines!(ax1, t, h12_flexsurv, color = flexsurv_color, linewidth = 2, 
               linestyle = :dashdot, label = "flexsurv")
    end
    
    axislegend(ax1, position = :lt)
    
    # h13 plot
    ax2 = Axis(fig[1, 2],
               title = "Transition 1→3 (Healthy → Death)",
               xlabel = "Time",
               ylabel = "Hazard h(t)",
               titlesize = 14)
    
    lines!(ax2, t, results.true_haz.h13, color = true_color, linewidth = 2, 
           label = "True: 0.1 + 0.02t")
    
    if !isnothing(results.julia)
        h13_julia = Vector{Float64}(results.julia["hazards"]["h13"])
        lines!(ax2, t, h13_julia, color = julia_color, linewidth = 2, 
               linestyle = :dash, label = "Julia (PIJCV)")
    end
    
    if !isnothing(results.r)
        h13_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h13"])
        h13_flexsurv = Vector{Float64}(results.r["flexsurv"]["hazards"]["h13"])
        lines!(ax2, t, h13_mgcv, color = mgcv_color, linewidth = 2, 
               linestyle = :dot, label = "mgcv (NCV)")
        lines!(ax2, t, h13_flexsurv, color = flexsurv_color, linewidth = 2, 
               linestyle = :dashdot, label = "flexsurv")
    end
    
    axislegend(ax2, position = :lt)
    
    # h23 plot
    ax3 = Axis(fig[2, 1],
               title = "Transition 2→3 (Illness → Death)",
               xlabel = "Time",
               ylabel = "Hazard h(t)",
               titlesize = 14)
    
    lines!(ax3, t, results.true_haz.h23, color = true_color, linewidth = 2, 
           label = "True: 0.4e^(-0.1t)")
    
    if !isnothing(results.julia)
        h23_julia = Vector{Float64}(results.julia["hazards"]["h23"])
        lines!(ax3, t, h23_julia, color = julia_color, linewidth = 2, 
               linestyle = :dash, label = "Julia (PIJCV)")
    end
    
    if !isnothing(results.r)
        h23_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h23"])
        h23_flexsurv = Vector{Float64}(results.r["flexsurv"]["hazards"]["h23"])
        lines!(ax3, t, h23_mgcv, color = mgcv_color, linewidth = 2, 
               linestyle = :dot, label = "mgcv (NCV)")
        lines!(ax3, t, h23_flexsurv, color = flexsurv_color, linewidth = 2, 
               linestyle = :dashdot, label = "flexsurv")
    end
    
    axislegend(ax3, position = :rt)
    
    # Summary statistics panel
    ax4 = Axis(fig[2, 2],
               title = "RMSE Comparison",
               xlabel = "Transition",
               ylabel = "RMSE vs True",
               titlesize = 14,
               xticks = (1:3, ["h12", "h13", "h23"]))
    
    # Collect RMSE values
    if !isnothing(results.julia) && !isnothing(results.r)
        julia_rmse = [results.julia["rmse"]["h12"], results.julia["rmse"]["h13"], results.julia["rmse"]["h23"]]
        mgcv_rmse = [results.r["mgcv"]["rmse"]["h12"], results.r["mgcv"]["rmse"]["h13"], results.r["mgcv"]["rmse"]["h23"]]
        flexsurv_rmse = [results.r["flexsurv"]["rmse"]["h12"], results.r["flexsurv"]["rmse"]["h13"], results.r["flexsurv"]["rmse"]["h23"]]
        
        x_pos = [1, 2, 3]
        bar_width = 0.25
        
        barplot!(ax4, x_pos .- bar_width, julia_rmse, width = bar_width, 
                 color = julia_color, label = "Julia")
        barplot!(ax4, x_pos, mgcv_rmse, width = bar_width, 
                 color = mgcv_color, label = "mgcv")
        barplot!(ax4, x_pos .+ bar_width, flexsurv_rmse, width = bar_width, 
                 color = flexsurv_color, label = "flexsurv")
        
        axislegend(ax4, position = :rt)
    end
    
    # Add title
    Label(fig[0, :], "Penalized Spline Benchmark: mgcv vs flexsurv vs MultistateModels.jl",
          fontsize = 18, font = :bold)
    
    # Add metadata footnote
    n_subjects = results.metadata["n_subjects"]
    Label(fig[3, :], "N = $n_subjects subjects | Illness-Death Model | Cubic B-splines with automatic smoothing",
          fontsize = 10, color = :gray)
    
    return fig
end

function create_residual_plot(results)
    fig = Figure(size = (1200, 400))
    
    t = results.eval_times
    
    # h12 residuals
    ax1 = Axis(fig[1, 1],
               title = "h12 Residuals",
               xlabel = "Time",
               ylabel = "Fitted - True")
    
    hlines!(ax1, [0], color = :gray, linestyle = :dash)
    
    if !isnothing(results.julia)
        h12_julia = Vector{Float64}(results.julia["hazards"]["h12"])
        lines!(ax1, t, h12_julia .- results.true_haz.h12, color = :blue, label = "Julia")
    end
    if !isnothing(results.r)
        h12_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h12"])
        lines!(ax1, t, h12_mgcv .- results.true_haz.h12, color = :red, label = "mgcv")
    end
    
    axislegend(ax1, position = :rb)
    
    # h13 residuals
    ax2 = Axis(fig[1, 2],
               title = "h13 Residuals",
               xlabel = "Time",
               ylabel = "Fitted - True")
    
    hlines!(ax2, [0], color = :gray, linestyle = :dash)
    
    if !isnothing(results.julia)
        h13_julia = Vector{Float64}(results.julia["hazards"]["h13"])
        lines!(ax2, t, h13_julia .- results.true_haz.h13, color = :blue, label = "Julia")
    end
    if !isnothing(results.r)
        h13_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h13"])
        lines!(ax2, t, h13_mgcv .- results.true_haz.h13, color = :red, label = "mgcv")
    end
    
    axislegend(ax2, position = :rb)
    
    # h23 residuals
    ax3 = Axis(fig[1, 3],
               title = "h23 Residuals",
               xlabel = "Time",
               ylabel = "Fitted - True")
    
    hlines!(ax3, [0], color = :gray, linestyle = :dash)
    
    if !isnothing(results.julia)
        h23_julia = Vector{Float64}(results.julia["hazards"]["h23"])
        lines!(ax3, t, h23_julia .- results.true_haz.h23, color = :blue, label = "Julia")
    end
    if !isnothing(results.r)
        h23_mgcv = Vector{Float64}(results.r["mgcv"]["hazards"]["h23"])
        lines!(ax3, t, h23_mgcv .- results.true_haz.h23, color = :red, label = "mgcv")
    end
    
    axislegend(ax3, position = :rt)
    
    return fig
end

function print_summary_table(results)
    println("\n" * "="^70)
    println("BENCHMARK SUMMARY")
    println("="^70)
    
    println("\n--- Dataset ---")
    println("N subjects: $(results.metadata["n_subjects"])")
    println("Max time: $(results.metadata["max_time"])")
    println("Transitions: 1→2 ($(results.metadata["n_12"])), 1→3 ($(results.metadata["n_13"])), 2→3 ($(results.metadata["n_23"]))")
    
    println("\n--- RMSE vs True Hazard ---")
    @printf("%-20s %10s %10s %10s\n", "Method", "h12", "h13", "h23")
    println("-"^55)
    
    if !isnothing(results.julia)
        @printf("%-20s %10.5f %10.5f %10.5f\n", "Julia (PIJCV)",
                results.julia["rmse"]["h12"], results.julia["rmse"]["h13"], results.julia["rmse"]["h23"])
    end
    
    if !isnothing(results.r)
        @printf("%-20s %10.5f %10.5f %10.5f\n", "mgcv (NCV)",
                results.r["mgcv"]["rmse"]["h12"], results.r["mgcv"]["rmse"]["h13"], results.r["mgcv"]["rmse"]["h23"])
        @printf("%-20s %10.5f %10.5f %10.5f\n", "flexsurv",
                results.r["flexsurv"]["rmse"]["h12"], results.r["flexsurv"]["rmse"]["h13"], results.r["flexsurv"]["rmse"]["h23"])
    end
    
    println("\n--- Computation Time (seconds) ---")
    if !isnothing(results.julia)
        @printf("%-20s %10.2f\n", "Julia", results.julia["time_seconds"])
    end
    if !isnothing(results.r)
        @printf("%-20s %10.2f\n", "mgcv", results.r["mgcv"]["time_seconds"])
        @printf("%-20s %10.2f\n", "flexsurv", results.r["flexsurv"]["time_seconds"])
    end
    
    if !isnothing(results.julia) && !isnothing(results.r)
        println("\n--- Smoothing Parameters ---")
        @printf("%-20s %12s %12s %12s\n", "Method", "λ_h12", "λ_h13", "λ_h23")
        println("-"^60)
        lambda = results.julia["lambda"]
        @printf("%-20s %12.4f %12.4f %12.4f\n", "Julia",
                lambda[1], lambda[2], lambda[3])
        @printf("%-20s %12.4f %12.4f %12.4f\n", "mgcv (sp)",
                results.r["mgcv"]["lambda"]["h12"], results.r["mgcv"]["lambda"]["h13"], results.r["mgcv"]["lambda"]["h23"])
    end
    
    println()
end

function main()
    println("Loading results...")
    results = load_results()
    
    # Print summary table
    print_summary_table(results)
    
    # Create plots
    println("Creating hazard comparison plot...")
    fig1 = create_hazard_comparison_plot(results)
    save(joinpath(OUTPUT_DIR, "hazard_comparison.png"), fig1)
    println("  Saved: hazard_comparison.png")
    
    println("Creating residual plot...")
    fig2 = create_residual_plot(results)
    save(joinpath(OUTPUT_DIR, "residual_comparison.png"), fig2)
    println("  Saved: residual_comparison.png")
    
    println("\nVisualization complete!")
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
