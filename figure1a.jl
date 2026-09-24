using CairoMakie
using Dates
using LinearAlgebra
using Optim
using Printf
using QuadGK
using Random
using SpecialFunctions
using Statistics
using TOML

##############################
# Parameters
##############################

const SEED = 20260912
const ALPHA = 0.7
const LAMBDAS = [0.01, 1.0]
const EXPERIMENT_KAPPAS = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5]
const THEORY_KAPPAS = collect(0.5:0.2:4.5)
const DELTA = 1.0
const SIGMA = 0.5
const TAU = 1.0
const THEORY = (
    population_size=1024,
    grid_points=71,
    grid_radius=7.0,
    grid_scale=0.25,
    mean_explicit_points=8.0,
    response_samples=512,
    response_step=0.01,
    min_iterations=60,
    max_iterations=160,
    check_window=15,
    relative_tolerance=0.005,
    bisection_steps=16,
)
const EXPERIMENT = (
    dimension=256,
    trials=8,
    coordinate_warmup_sweeps=20,
    lbfgs_iterations=2500,
    trust_region_iterations=120,
    duality_gap_tolerance=1.0e-8,
)
const OUTPUT_DIR = joinpath(@__DIR__, "output")

##############################
# Stable distribution
##############################

levy_constant(alpha) = gamma(alpha + 1) * sinpi(alpha / 2) / pi

function stable!(rng, output, alpha)
    for i in eachindex(output)
        angle = pi * (rand(rng) - 0.5)
        exponential = randexp(rng)
        output[i] = sin(alpha * angle) / cos(angle)^(1 / alpha) *
                    (cos((1 - alpha) * angle) / exponential)^((1 - alpha) / alpha)
    end
    output
end

##############################
# Theory
##############################

function population_grid()
    extent = asinh(THEORY.grid_radius / THEORY.grid_scale)
    THEORY.grid_scale .* sinh.(range(-extent, extent, length=THEORY.grid_points))
end

@inline function inverse_derivative(population, row, grid, field, ridge)
    field <= population[row, 1] &&
        return grid[1] + (field - population[row, 1]) / ridge
    field >= population[row, end] &&
        return grid[end] + (field - population[row, end]) / ridge
    lo, hi = 1, length(grid)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if population[row, mid] <= field
            lo = mid
        else
            hi = mid
        end
    end
    weight = (field - population[row, lo]) /
             (population[row, hi] - population[row, lo])
    grid[lo] + weight * (grid[hi] - grid[lo])
end

function population_roots(population, grid, ridge)
    roots = Vector{Float64}(undef, size(population, 1))
    for row in axes(population, 1)
        roots[row] = inverse_derivative(population, row, grid, 0.0, ridge)
    end
    roots
end

function roots_and_response(population, grid, ridge)
    roots = population_roots(population, grid, ridge)
    responses = similar(roots)
    for row in axes(population, 1)
        hi = searchsortedfirst(view(population, row, :), 0.0)
        responses[row] = if hi == 1 || hi > length(grid)
            1 / ridge
        else
            (grid[hi] - grid[hi - 1]) /
            (population[row, hi] - population[row, hi - 1])
        end
    end
    mean(roots .^ 2), mean(responses), mean(abs.(roots) .^ ALPHA)
end

function cutoff_points(rng, cutoff, rows)
    coefficient = 2 * levy_constant(ALPHA) / ALPHA
    active_mean = coefficient / cutoff^ALPHA
    row_points = Vector{Vector{Float64}}(undef, rows)
    for row in 1:rows
        points = Float64[]
        arrival = 0.0
        while true
            arrival += randexp(rng)
            arrival > THEORY.mean_explicit_points && break
            magnitude = (coefficient / arrival)^(1 / ALPHA)
            push!(points, arrival <= active_mean ?
                          ifelse(rand(rng, Bool), magnitude, -magnitude) : 0.0)
        end
        row_points[row] = points
    end
    points = zeros(rows, max(maximum(length.(row_points)), 1))
    for row in 1:rows
        points[row, 1:length(row_points[row])] .= row_points[row]
    end
    points
end

function compensated_factor_duals(population, external, grid, ridge, cutoff,
                                  weak_variance, q2, susceptibility,
                                  evaluation_grid, rng)
    factors = length(external)
    inner = cutoff_points(rng, cutoff, factors)
    children = rand(rng, 1:size(population, 1), size(inner))
    noise = SIGMA .* randn(rng, factors)
    weak_field = sqrt(max(weak_variance * q2, 0.0)) .* randn(rng, factors)
    result = Matrix{Float64}(undef, factors, length(evaluation_grid))

    function residual(factor, x, dual)
        marked_sum = 0.0
        for edge in axes(inner, 2)
            point = inner[factor, edge]
            point == 0 && continue
            marked_sum += point * inverse_derivative(
                population, children[factor, edge], grid, -point * dual, ridge)
        end
        (1 + weak_variance * susceptibility) * dual - marked_sum -
        external[factor] * x - weak_field[factor] + noise[factor]
    end

    for factor in 1:factors, (column, x) in enumerate(evaluation_grid)
        lower, upper = -DELTA, DELTA
        if residual(factor, x, lower) >= 0
            result[factor, column] = lower
        elseif residual(factor, x, upper) <= 0
            result[factor, column] = upper
        else
            for _ in 1:THEORY.bisection_steps
                midpoint = (lower + upper) / 2
                if residual(factor, x, midpoint) < 0
                    lower = midpoint
                else
                    upper = midpoint
                end
            end
            result[factor, column] = (lower + upper) / 2
        end
    end
    result
end

stream_seed(iteration, stream) = SEED + 10_000_000 + 100_000 + 10iteration + stream

function solve_theory(kappa, ridge)
    grid = population_grid()
    coefficient = 2 * levy_constant(ALPHA) / ALPHA
    cutoff = (coefficient / THEORY.mean_explicit_points)^(1 / ALPHA)
    weak_variance = 2 * levy_constant(ALPHA) / (2 - ALPHA) * cutoff^(2 - ALPHA)
    response_grid = [-THEORY.response_step, 0.0, THEORY.response_step]

    rng = MersenneTwister(stream_seed(0, 1))
    teacher = randn(rng, THEORY.population_size)
    population = ridge .* (grid' .+ teacher)
    history = Float64[]
    stationarity_gap = Inf
    converged = false

    for iteration in 1:THEORY.max_iterations
        q2, susceptibility, _ = roots_and_response(population, grid, ridge)
        rng = MersenneTwister(stream_seed(iteration, 1))
        response_duals = compensated_factor_duals(
            population, ones(THEORY.response_samples), grid, ridge, cutoff,
            weak_variance, q2, susceptibility, response_grid, rng)
        response_zero = view(response_duals, :, 2)
        response_derivative = (view(response_duals, :, 3) .-
                               view(response_duals, :, 1)) ./
                              (response_grid[3] - response_grid[1])
        kappa_weak_scale = kappa^(2 / ALPHA)
        q_hat = kappa_weak_scale * weak_variance * mean(response_zero .^ 2)
        a_hat = kappa_weak_scale * weak_variance * mean(response_derivative)

        rng = MersenneTwister(stream_seed(iteration, 2))
        outer = cutoff_points(rng, cutoff, THEORY.population_size)
        outer .*= kappa^(1 / ALPHA)
        rng = MersenneTwister(stream_seed(iteration, 3))
        flat_duals = compensated_factor_duals(
            population, vec(outer), grid, ridge, cutoff, weak_variance, q2,
            susceptibility, grid, rng)
        duals = reshape(flat_duals, size(outer, 1), size(outer, 2), length(grid))
        rng = MersenneTwister(stream_seed(iteration, 4))
        teacher = randn(rng, THEORY.population_size)
        weak_field = sqrt(max(q_hat, 0.0)) .* randn(rng, THEORY.population_size)
        updated = ridge .* (grid' .+ teacher) .+ weak_field .+ a_hat .* grid'
        for row in 1:THEORY.population_size, edge in axes(outer, 2)
            point = outer[row, edge]
            point == 0 && continue
            @views updated[row, :] .+= point .* duals[row, edge, :]
        end
        minimum(diff(updated, dims=2)) >= -1e-8 ||
            error("non-monotone saddle update at lambda=$ridge, kappa=$kappa")
        population = updated
        _, _, q_alpha = roots_and_response(population, grid, ridge)
        push!(history, q_alpha)

        window = THEORY.check_window
        if iteration >= max(THEORY.min_iterations, 2window)
            current = mean(view(history, iteration - window + 1:iteration))
            previous = mean(view(history, iteration - 2window + 1:iteration - window))
            stationarity_gap = abs(current - previous) / max(abs(current), eps())
            if stationarity_gap <= THEORY.relative_tolerance
                converged = true
                break
            end
        end
    end
    window = min(THEORY.check_window, length(history))
    q_alpha = mean(view(history, length(history) - window + 1:length(history)))
    (; alpha=ALPHA, kappa, ridge, q_alpha, iterations=length(history),
       stationarity_gap, converged)
end

##############################
# Experiments
##############################

function huber_objective_gradient(weights, design, response, ridge)
    residual = design * weights - response
    objective = sum(ifelse(abs(x) <= DELTA, x^2 / 2,
                           DELTA * abs(x) - DELTA^2 / 2) for x in residual) +
                ridge * dot(weights, weights) / 2
    score = clamp.(residual, -DELTA, DELTA)
    objective, design' * score + ridge * weights, residual
end

function coordinate_warmup(design, response, ridge)
    sample_count, dimension = size(design)
    dual = zeros(sample_count)
    transformed = zeros(dimension)
    curvature = 1 .+ vec(sum(abs2, design; dims=2)) ./ ridge
    for _ in 1:EXPERIMENT.coordinate_warmup_sweeps, sample in 1:sample_count
        row = view(design, sample, :)
        old = dual[sample]
        gradient = old + response[sample] + dot(row, transformed) / ridge
        new = clamp(old - gradient / curvature[sample], -DELTA, DELTA)
        difference = new - old
        dual[sample] = new
        BLAS.axpy!(difference, row, transformed)
    end
    -transformed ./ ridge
end

function fit_huber_ridge(design, response, ridge)
    weights = coordinate_warmup(design, response, ridge)
    tolerance = EXPERIMENT.duality_gap_tolerance
    function objective_gradient!(objective_requested, gradient, candidate)
        objective, computed_gradient, _ =
            huber_objective_gradient(candidate, design, response, ridge)
        gradient === nothing || copyto!(gradient, computed_gradient)
        objective_requested === nothing ? nothing : objective
    end
    result = optimize(Optim.only_fg!(objective_gradient!), weights, LBFGS(),
        Optim.Options(g_tol=tolerance, iterations=EXPERIMENT.lbfgs_iterations))
    weights = Optim.minimizer(result)

    function diagnostics(candidate)
        objective, gradient, residual =
            huber_objective_gradient(candidate, design, response, ridge)
        dual = clamp.(residual, -DELTA, DELTA)
        transformed = design' * dual
        dual_value = -dot(dual, response) - dot(dual, dual) / 2 -
                     dot(transformed, transformed) / (2ridge)
        relative_gap = max(0.0, objective - dual_value) / (1 + abs(objective))
        stationarity = norm(gradient, Inf) / (1 + abs(objective))
        relative_gap, stationarity
    end

    iterations = Optim.iterations(result)
    relative_gap, stationarity = diagnostics(weights)
    if relative_gap > tolerance || stationarity > sqrt(tolerance)
        objective(candidate) =
            huber_objective_gradient(candidate, design, response, ridge)[1]
        function gradient!(gradient, candidate)
            copyto!(gradient,
                huber_objective_gradient(candidate, design, response, ridge)[2])
        end
        function hessian!(hessian, candidate)
            residual = design * candidate - response
            active_design = view(design, abs.(residual) .< DELTA, :)
            mul!(hessian, active_design', active_design)
            hessian[diagind(hessian)] .+= ridge
        end
        twice_differentiable = Optim.TwiceDifferentiable(
            objective, gradient!, hessian!, weights)
        fallback = optimize(twice_differentiable, weights, NewtonTrustRegion(),
            Optim.Options(g_tol=tolerance,
                          iterations=EXPERIMENT.trust_region_iterations))
        weights = Optim.minimizer(fallback)
        iterations += Optim.iterations(fallback)
        relative_gap, stationarity = diagnostics(weights)
    end
    if relative_gap > tolerance || stationarity > sqrt(tolerance)
        scales = sqrt.(ridge .+ vec(sum(abs2, design; dims=1)))
        function scaled_objective_gradient!(objective_requested, gradient,
                                            scaled_weights)
            candidate = scaled_weights ./ scales
            objective, computed_gradient, _ =
                huber_objective_gradient(candidate, design, response, ridge)
            gradient === nothing || copyto!(gradient, computed_gradient ./ scales)
            objective_requested === nothing ? nothing : objective
        end
        scaled_result = optimize(
            Optim.only_fg!(scaled_objective_gradient!), zeros(length(weights)), LBFGS(),
            Optim.Options(g_tol=tolerance, iterations=EXPERIMENT.lbfgs_iterations))
        weights = Optim.minimizer(scaled_result) ./ scales
        iterations += Optim.iterations(scaled_result)
        relative_gap, stationarity = diagnostics(weights)
    end
    success = relative_gap <= tolerance && stationarity <= sqrt(tolerance)
    weights, success, iterations, relative_gap, stationarity
end

function run_experiment(kappa, ridge, trial)
    sample_count = round(Int, kappa * EXPERIMENT.dimension)
    seed = SEED + 100_000_000 + 1_000_000 +
           10_000 * round(Int, 10kappa) + trial
    rng = MersenneTwister(seed)
    design = Matrix{Float64}(undef, sample_count, EXPERIMENT.dimension)
    stable!(rng, design, ALPHA)
    design ./= EXPERIMENT.dimension^(1 / ALPHA)
    teacher = randn(rng, EXPERIMENT.dimension)
    response = design * teacher + SIGMA .* randn(rng, sample_count)
    weights, success, iterations, gap, stationarity =
        fit_huber_ridge(design, response, ridge)
    q_alpha = mean(abs.(weights .- teacher) .^ ALPHA)
    (; alpha=ALPHA, kappa, ridge, trial, q_alpha, success, iterations,
       relative_duality_gap=gap, relative_stationarity=stationarity, seed)
end

##############################
# Bayes benchmark
##############################

function bayes_curvature_moment(beta, linear, stable_scale)
    integral, _ = quadgk(
        u -> exp(-linear * u^(1 / beta) - stable_scale * u), 0, Inf;
        rtol=1e-11)
    integral / gamma(1 + beta)
end

function bayes_risk(kappa; tolerance=1e-10)
    beta = ALPHA / 2
    lambda = SIGMA^2 / TAU^2
    levy_scale = 2 * levy_constant(ALPHA) / ALPHA * gamma(1 - beta)
    hatted_moment = lambda^(-beta)
    unhatted_moment = bayes_curvature_moment(beta, 1.0,
                                              levy_scale * hatted_moment)
    for _ in 1:1_000
        new_hatted = bayes_curvature_moment(
            beta, lambda, kappa * levy_scale * unhatted_moment)
        new_unhatted = bayes_curvature_moment(
            beta, 1.0, levy_scale * new_hatted)
        gap = max(abs(new_hatted - hatted_moment) / new_hatted,
                  abs(new_unhatted - unhatted_moment) / new_unhatted)
        hatted_moment, unhatted_moment = new_hatted, new_unhatted
        if gap <= tolerance
            gaussian_moment = 2^beta * gamma((ALPHA + 1) / 2) / sqrt(pi)
            return gaussian_moment * SIGMA^ALPHA * hatted_moment
        end
    end
    error("Bayes curvature moments did not converge at kappa=$kappa")
end

##############################
# Output
##############################

csv_value(x::Bool) = x ? "true" : "false"
csv_value(x::AbstractFloat) = isfinite(x) ? @sprintf("%.12g", x) : string(x)
csv_value(x) = string(x)

function write_csv(path, rows, fields)
    open(path, "w") do io
        println(io, join(string.(fields), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, field)) for field in fields), ','))
        end
    end
end

function summarize(rows)
    summary = NamedTuple[]
    for ridge in LAMBDAS, kappa in EXPERIMENT_KAPPAS
        selected = [row.q_alpha for row in rows
                    if row.ridge == ridge && row.kappa == kappa && row.success]
        isempty(selected) && continue
        push!(summary, (; alpha=ALPHA, kappa, ridge, mean=mean(selected),
            se=length(selected) < 2 ? NaN : std(selected) / sqrt(length(selected)),
            count=length(selected)))
    end
    summary
end

function plot_figure(theory_rows, experiment_summary, bayes_rows)
    colors = Makie.wong_colors()
    figure = Figure(size=(650, 540), fontsize=27)
    axis = Axis(figure[1, 1], xlabel=L"\kappa=N/D", ylabel=L"q_\alpha",
        title=L"\alpha=0.7", titlesize=34, xlabelsize=32, ylabelsize=32,
        xticklabelsize=25, yticklabelsize=25,
        xgridcolor=(:gray, 0.14), ygridcolor=(:gray, 0.14))
    styles = [(LAMBDAS[1], colors[1], :circle), (LAMBDAS[2], colors[3], :diamond)]
    for (ridge, color, marker) in styles
        theory = [row for row in theory_rows if row.ridge == ridge]
        finite = [row for row in experiment_summary if row.ridge == ridge]
        label = L"\lambda=%$ridge"
        lines!(axis, getproperty.(theory, :kappa), getproperty.(theory, :q_alpha);
               color, linewidth=3, label)
        errorbars!(axis, getproperty.(finite, :kappa), getproperty.(finite, :mean),
                   getproperty.(finite, :se); color, whiskerwidth=10)
        scatter!(axis, getproperty.(finite, :kappa), getproperty.(finite, :mean);
                 color, marker, markersize=12, strokecolor=:white,
                 strokewidth=0.8, label)
    end
    lines!(axis, getproperty.(bayes_rows, :kappa), getproperty.(bayes_rows, :q_alpha);
           color=:black, linewidth=3, label=L"\mathrm{Bayes\ optimal}")
    xlims!(axis, minimum(THEORY_KAPPAS), maximum(THEORY_KAPPAS))
    axislegend(axis; position=:rt, labelsize=24,
               merge=true, unique=true, framevisible=false)
    Label(figure[1, 1, TopLeft()], "A"; fontsize=42, font=:bold,
          halign=:right, padding=(0, 8, 5, 0))
    save(joinpath(OUTPUT_DIR, "figure1a.png"), figure; px_per_unit=2)
    save(joinpath(OUTPUT_DIR, "figure1a.pdf"), figure)
end

##############################
# Run
##############################

function main()
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    mkpath(OUTPUT_DIR)

    theory_jobs = [(ridge, kappa) for ridge in LAMBDAS for kappa in THEORY_KAPPAS]
    theory_rows = Vector{NamedTuple}(undef, length(theory_jobs))
    Threads.@threads for index in eachindex(theory_jobs)
        ridge, kappa = theory_jobs[index]
        theory_rows[index] = solve_theory(kappa, ridge)
        @printf("theory %d/%d: lambda=%.2g kappa=%.2f q=%.5g\n",
                index, length(theory_jobs), ridge, kappa, theory_rows[index].q_alpha)
    end

    experiment_jobs = [(ridge, kappa, trial) for ridge in LAMBDAS
                       for kappa in EXPERIMENT_KAPPAS for trial in 1:EXPERIMENT.trials]
    experiment_rows = Vector{NamedTuple}(undef, length(experiment_jobs))
    Threads.@threads for index in eachindex(experiment_jobs)
        ridge, kappa, trial = experiment_jobs[index]
        experiment_rows[index] = run_experiment(kappa, ridge, trial)
        @printf("ERM %d/%d: lambda=%.2g kappa=%.2f q=%.5g\n",
                index, length(experiment_jobs), ridge, kappa,
                experiment_rows[index].q_alpha)
    end
    for ridge in LAMBDAS, kappa in EXPERIMENT_KAPPAS
        successes = count(row -> row.ridge == ridge && row.kappa == kappa &&
                                 row.success, experiment_rows)
        successes >= EXPERIMENT.trials - 1 ||
            error("only $successes ERM fits converged at lambda=$ridge, kappa=$kappa")
    end

    sort!(theory_rows; by=row -> (row.ridge, row.kappa))
    sort!(experiment_rows; by=row -> (row.ridge, row.kappa, row.trial))
    experiment_summary = summarize(experiment_rows)
    bayes_rows = [(; alpha=ALPHA, kappa, q_alpha=bayes_risk(kappa))
                  for kappa in range(first(THEORY_KAPPAS), last(THEORY_KAPPAS);
                                     length=161)]
    write_csv(joinpath(OUTPUT_DIR, "theory.csv"), theory_rows,
        (:alpha, :kappa, :ridge, :q_alpha, :iterations, :stationarity_gap, :converged))
    write_csv(joinpath(OUTPUT_DIR, "experiment_trials.csv"), experiment_rows,
        (:alpha, :kappa, :ridge, :trial, :q_alpha, :success, :iterations,
         :relative_duality_gap, :relative_stationarity, :seed))
    write_csv(joinpath(OUTPUT_DIR, "experiment_summary.csv"), experiment_summary,
        (:alpha, :kappa, :ridge, :mean, :se, :count))
    write_csv(joinpath(OUTPUT_DIR, "bayes_theory.csv"), bayes_rows,
        (:alpha, :kappa, :q_alpha))
    metadata = Dict(
        "generated_at" => string(now()),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "seed" => SEED,
        "alpha" => ALPHA,
        "lambdas" => LAMBDAS,
        "theory_kappas" => THEORY_KAPPAS,
        "experiment_kappas" => EXPERIMENT_KAPPAS,
        "delta" => DELTA,
        "sigma" => SIGMA,
        "tau" => TAU,
        "theory" => Dict(string(key) => value for (key, value) in pairs(THEORY)),
        "experiment" => Dict(string(key) => value for (key, value) in pairs(EXPERIMENT)),
    )
    open(joinpath(OUTPUT_DIR, "run_metadata.toml"), "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    plot_figure(theory_rows, experiment_summary, bayes_rows)
    @printf("wrote %s\n", OUTPUT_DIR)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
