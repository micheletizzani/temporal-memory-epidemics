#!/usr/bin/env julia
"""
Identifiability phase diagram over (N, T, n_node, kappa) -- the heavy compute driver.

Julia mirror of `scripts/run_identifiability_sweep.py`. This is the version to run for
production sweeps: it multithreads over replicates and has no per-event Python overhead.

    julia --project=. -t auto scripts/run_identifiability_sweep.jl --reps 50
    julia --project=. -t auto scripts/run_identifiability_sweep.jl --quick

RUN scripts/check_parity.jl FIRST. The Julia code here was written without a Julia
runtime available and is unverified until that check passes.

For every cell of the design, `reps` independent realisations are fitted and the script
records, per parameter: bias, rmse, across-replicate sd, mean asymptotic se, coverage of
the 95% Wald interval, and power of the corresponding boundary-null LR test.

`coverage` is the headline diagnostic. Cells below the nominal 0.95 are the degenerate
regime: the estimator is confidently wrong. A large gap between `sd` and `mean_se` says
the asymptotics have not kicked in at that sample size, which is itself a result.
"""

using Printf
using Random
using Statistics
using Base.Threads

include(joinpath(@__DIR__, "GenerativeModels.jl"))
include(joinpath(@__DIR__, "Inference.jl"))
using .GenerativeModels
using .Inference

function parse_args(args)
    opts = Dict{String,Any}(
        "N" => [25, 50], "T" => [150.0, 600.0, 2400.0],
        "n_node" => [0.0, 0.3, 0.6], "kappa" => [0.0, 5.0, 20.0],
        "gamma_n" => 0.5, "gamma_e" => 0.05, "reps" => 10, "seed" => 0,
        "out" => joinpath(@__DIR__, "..", "results", "identifiability_sweep_julia.csv"),
        "quick" => false)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--quick"
            opts["quick"] = true; i += 1
        elseif a in ("--reps", "--seed")
            opts[a[3:end]] = parse(Int, args[i+1]); i += 2
        elseif a in ("--gamma-n", "--gamma-e")
            opts[replace(a[3:end], "-" => "_")] = parse(Float64, args[i+1]); i += 2
        elseif a == "--out"
            opts["out"] = args[i+1]; i += 2
        elseif a in ("--N", "--T", "--n-node", "--kappa")
            key = a == "--N" ? "N" : a == "--T" ? "T" :
                  a == "--n-node" ? "n_node" : "kappa"
            vals = String[]
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(vals, args[i]); i += 1
            end
            opts[key] = key == "N" ? parse.(Int, vals) : parse.(Float64, vals)
        else
            error("unknown argument $a")
        end
    end
    if opts["quick"]
        opts["N"] = [25]; opts["T"] = [150.0, 600.0]
        opts["n_node"] = [0.0, 0.6]; opts["kappa"] = [0.0, 20.0]; opts["reps"] = 4
    end
    return opts
end

struct RepResult
    ok::Bool
    n_node::Float64
    kappa::Float64
    se_n::Float64
    se_k::Float64
    rej_n::Bool
    rej_k::Bool
    n_events::Int
end

function run_rep(N::Int, T::Float64, truth::MechanismParams, seed::Int)
    rng = MersenneTwister(seed)
    act = 0.02 .+ 0.03 .* rand(rng, N)
    model = TemporalNetworkModel(N, act, truth)
    events = simulate(model, T; rng=MersenneTwister(seed + 100_000))
    length(events) < 30 && return RepResult(false, NaN, NaN, NaN, NaN, false, false,
                                            length(events))
    res = try
        fit_observed_initiator(events, N, T)
    catch
        return RepResult(false, NaN, NaN, NaN, NaN, false, false, length(events))
    end
    contacts, initiator, side = Inference.split_events(events)
    p_k = likelihood_ratio_test(res.nll_selection,
                                selection_nll(0.0, res.gamma_e, contacts, side, N), 1)
    p_n = likelihood_ratio_test(res.nll_timing,
                                timing_nll(0.0, res.gamma_n, contacts, initiator, N, T), 1)
    return RepResult(true, res.n_node, res.kappa,
                     res.se["n_node"], res.se["kappa"],
                     isfinite(p_n) && p_n < 0.05, isfinite(p_k) && p_k < 0.05,
                     length(events))
end

function summarise_cell(reps::Vector{RepResult}, truth::MechanismParams)
    ok = filter(r -> r.ok, reps)
    row = Dict{String,Any}("n_ok" => length(ok), "n_failed" => length(reps) - length(ok),
                           "mean_events" => mean([r.n_events for r in reps]))
    for (name, getter, se_getter, rej_getter, tv) in (
            ("n_node", r -> r.n_node, r -> r.se_n, r -> r.rej_n, truth.n_node),
            ("kappa", r -> r.kappa, r -> r.se_k, r -> r.rej_k, truth.kappa))
        if isempty(ok)
            for suffix in ("bias", "rmse", "sd", "mean_se", "coverage", "power")
                row["$(name)_$(suffix)"] = NaN
            end
            continue
        end
        e = Float64[getter(r) for r in ok]
        s = Float64[se_getter(r) for r in ok]
        good = findall(x -> isfinite(x) && x > 0, s)
        row["$(name)_bias"] = mean(e) - tv
        row["$(name)_rmse"] = sqrt(mean((e .- tv) .^ 2))
        row["$(name)_sd"] = length(e) > 1 ? std(e) : NaN
        row["$(name)_mean_se"] = isempty(good) ? NaN : mean(s[good])
        row["$(name)_coverage"] = isempty(good) ? NaN :
            mean(abs.(e[good] .- tv) .< 1.96 .* s[good])
        row["$(name)_power"] = mean(Float64[rej_getter(r) for r in ok])
    end
    return row
end

function main()
    opts = parse_args(ARGS)
    design = [(N, T, n, k) for N in opts["N"], T in opts["T"],
              n in opts["n_node"], k in opts["kappa"]]
    design = vec(design)
    @printf("design: %d cells x %d reps = %d fits on %d thread(s)\n",
            length(design), opts["reps"], length(design) * opts["reps"], nthreads())

    header = ["N", "T", "true_n_node", "true_kappa", "gamma_n", "gamma_e", "reps",
              "n_ok", "n_failed", "mean_events",
              "n_node_bias", "n_node_rmse", "n_node_sd", "n_node_mean_se",
              "n_node_coverage", "n_node_power",
              "kappa_bias", "kappa_rmse", "kappa_sd", "kappa_mean_se",
              "kappa_coverage", "kappa_power"]

    mkpath(dirname(opts["out"]))
    open(opts["out"], "w") do io
        println(io, join(header, ","))
        for (c, (N, T, n, k)) in enumerate(design)
            truth = MechanismParams(n, opts["gamma_n"], k, opts["gamma_e"])
            seed0 = opts["seed"] + 1000c
            reps = Vector{RepResult}(undef, opts["reps"])
            @threads for r in 1:opts["reps"]
                reps[r] = run_rep(N, T, truth, seed0 + r)
            end
            row = summarise_cell(reps, truth)
            row["N"] = N; row["T"] = T; row["true_n_node"] = n; row["true_kappa"] = k
            row["gamma_n"] = opts["gamma_n"]; row["gamma_e"] = opts["gamma_e"]
            row["reps"] = opts["reps"]
            println(io, join([string(row[h]) for h in header], ","))
            @printf("[%3d/%3d] N=%3d T=%7.0f n=%.2f k=%5.1f | E~%7.0f | n_bias=%+.3f cov=%.2f | k_bias=%+.2f cov=%.2f\n",
                    c, length(design), N, T, n, k, row["mean_events"],
                    row["n_node_bias"], row["n_node_coverage"],
                    row["kappa_bias"], row["kappa_coverage"])
        end
    end
    println("\nwrote $(opts["out"])")
    println("Inspect cells with coverage < 0.80: those are the degenerate regime, where")
    println("the estimator is confidently wrong rather than merely imprecise.")
end

main()
