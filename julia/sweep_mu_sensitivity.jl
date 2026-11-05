# sweep_mu_sensitivity.jl — Test stability of naive-fit under-prediction across recovery speeds
using Random, Statistics, JSON, Printf

include("epidemic_poles.jl")

function run_mu_sweep(N=55, E=4565, T=3000.0, reps=300)
    mus = [1/300.0, 1/120.0, 1/60.0, 1/30.0, 1/15.0]
    betas = [0.08, 0.14, 0.20, 0.30, 0.45]
    
    results = Dict{String, Any}()

    for mu in mus
        res_mu = Dict{String, Any}()
        for (name, kind) in [("naive", "node"), ("corrected", "link")]
            rng = MersenneTwister(42)
            ev = gen_pole(kind, N, T, E, 42)
            
            final_sizes = Float64[]
            for be in betas
                m, _ = run_sir(ev, N, be, mu, rng, reps)
                push!(final_sizes, m)
            end
            res_mu[name] = final_sizes
        end
        results[@sprintf("mu_%.4f", mu)] = res_mu
        
        ratio = res_mu["corrected"][4] / max(res_mu["naive"][4], 1e-6)
        @printf("mu = %.4f | Corrected/Naive Ratio at beta=0.30: %.2fx\n", mu, ratio)
    end
    
    outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)
    open(joinpath(outdir, "mu_sensitivity_results.json"), "w") do io
        JSON.print(io, results)
    end
    println("Saved -> results/mu_sensitivity_results.json")
end

run_mu_sweep()
