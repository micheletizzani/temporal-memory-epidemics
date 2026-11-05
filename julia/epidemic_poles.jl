# epidemic_poles.jl
# Julia port of epidemic_poles.py -- same logic, scaled up.
#
# Self-contained: inlines the two-channel Ogata simulator (from joint_kernel_map.py)
# and the SIR / SIS engines (from epidemic_divergence.py/.jl) so this file has NO
# cross-experiment imports beyond GenerativeModels.jl (which is only used for gen_pole
# in epidemic_divergence; here we use the mixture-kernel simulator directly).
#
# Usage:
#   julia epidemic_poles.jl                             # quick sandbox (SIS, small N)
#   julia epidemic_poles.jl sis  60,100,160,260 reps=200 G=10
#   julia epidemic_poles.jl sir  60,100,160,260 reps=200 G=10
#
# Outputs results/poles_<model>_N<N>_<kind>.json

using Random, Statistics, JSON, Printf
using DataStructures   # BinaryMinHeap for exact SIS recoveries
using LinearAlgebra: mul!, dot

# ── constants ────────────────────────────────────────────────────────────────
const EVENTS_PER_NODE = 83
const POLES = Dict("node" => (0.85, 0.0),
                   "link" => (0.0, 40.0),
                   "poisson" => (0.0, 0.0))

# ── two-channel Ogata thinning (from joint_kernel_map.py) ────────────────────
"""
    simulate_two_channel(N, a, b, n_node, node_ws, node_gm, kappa, tie_vs, tie_dl,
                         T, rng; max_events=200_000)

Exact O(E) Ogata thinning for node self-excitation (mixture kernel, unit integral)
+ tie reinforcement (mixture kernel, unit peak).
Returns `Vector{Tuple{Float64,Int,Int}}` — events `(t, initiator, partner)`, 1-indexed.
"""
function simulate_two_channel(N::Int, a::Vector{Float64}, b::Vector{Float64},
                              n_node::Float64, node_ws::Vector{Float64},
                              node_gm::Vector{Float64}, kappa::Float64,
                              tie_vs::Vector{Float64}, tie_dl::Vector{Float64},
                              T::Float64, rng::AbstractRNG;
                              max_events::Int=200_000)
    Kn = length(node_gm)
    Ke = length(tie_dl)
    S  = zeros(N, Kn)                                   # per-node excitation components
    tie = Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Float64}}()   # (i,j) => (vec, t_last)
    ev = Tuple{Float64,Int,Int}[]
    t  = 0.0

    # node_intensity: a + n_node * (S * node_ws)
    lam_buf = similar(a)

    while length(ev) < max_events
        # compute intensity upper bound (valid: only decays until next event)
        mul!(lam_buf, S, node_ws)    # lam_buf[i] = sum_m S[i,m] * node_ws[m]
        @. lam_buf = a + n_node * lam_buf
        lam_star = sum(lam_buf)
        lam_star <= 0 && break

        dt = randexp(rng) / lam_star
        t += dt
        t >= T && break

        # decay S by dt
        @inbounds for m in 1:Kn
            decay = exp(-node_gm[m] * dt)
            @simd for i in 1:N
                S[i, m] *= decay
            end
        end

        # recompute intensity at candidate time
        mul!(lam_buf, S, node_ws)
        @. lam_buf = a + n_node * lam_buf
        lam_tot = sum(lam_buf)
        rand(rng) * lam_star >= lam_tot && continue     # thinned

        # choose initiator
        i = _sample_index(lam_buf, lam_tot, rng)

        # partner weights: b_l + kappa * m_il(t)
        w = copy(b)
        w[i] = 0.0
        if kappa > 0
            for ((p, q), (vec, tp)) in tie
                local l::Int
                if p == i
                    l = q
                elseif q == i
                    l = p
                else
                    continue
                end
                m_val = dot(tie_vs, vec .* exp.(.-tie_dl .* (t - tp)))
                w[l] += kappa * m_val
            end
        end
        wt = sum(w)
        wt <= 0 && continue
        j = _sample_index(w, wt, rng)

        # register: bump node-i excitation, bump tie memory
        @inbounds for m in 1:Kn
            S[i, m] += node_gm[m]
        end
        key = i < j ? (i, j) : (j, i)
        if haskey(tie, key)
            vec_old, tp_old = tie[key]
            vec_new = vec_old .* exp.(.-tie_dl .* (t - tp_old)) .+ 1.0
            tie[key] = (vec_new, t)
        else
            tie[key] = (ones(Ke), t)
        end
        push!(ev, (t, i, j))
    end
    return ev
end

# helper shared with GenerativeModels
function _sample_index(w::AbstractVector{Float64}, total::Float64, rng::AbstractRNG)
    u = rand(rng) * total
    c = 0.0
    @inbounds for k in eachindex(w)
        c += w[k]
        c >= u && return k
    end
    @inbounds for k in length(w):-1:1
        w[k] > 0 && return k
    end
    error("cannot sample: all weights zero")
end


# ── pole generation ──────────────────────────────────────────────────────────
"""
    gen_pole(kind, N, T, E, seed)

Matched-activity contact sequence for a pole (n_node, kappa), trimmed to E events.
"""
function gen_pole(kind::String, N::Int, T::Float64, E::Int, seed::Int)
    rng = MersenneTwister(seed)
    a0 = exp.(0.6 .* randn(rng, N))
    n_node, kappa = POLES[kind]
    a = a0 .* ((E / T) * (1.0 - n_node) / sum(a0))
    b = ones(N)
    exp1_ws = [1.0]; exp1_gm = [1.0]                   # single-exponential kernels
    ev = simulate_two_channel(N, a, b, n_node, exp1_ws, exp1_gm,
                              kappa, exp1_ws, exp1_gm, T, rng;
                              max_events=6 * E)
    sort!(ev, by=x -> x[1])
    return ev[1:min(E, length(ev))]
end

# ── SIR engine ───────────────────────────────────────────────────────────────
"""
    run_sir(events, N, beta, mu, rng, reps) -> (mean_final_size, ...)

Temporal-network SIR; returns (mean, se) of final epidemic size.
"""
function run_sir_samples(events::Vector{Tuple{Float64,Int,Int}}, N::Int,
                         beta::Float64, mu::Float64, rng::AbstractRNG, reps::Int)
    finals = zeros(reps)
    t0 = events[1][1]
    @inbounds for r in 1:reps
        state = zeros(Int8, N)
        rec   = fill(Inf, N)
        seed  = rand(rng, 1:N)
        state[seed] = 1; rec[seed] = t0 + randexp(rng) / mu
        ever  = 1
        for (t, i, j) in events
            if state[i] == 1 && t >= rec[i]; state[i] = 2; end
            if state[j] == 1 && t >= rec[j]; state[j] = 2; end
            si, sj = state[i], state[j]
            if si == 1 && sj == 0 && rand(rng) < beta
                state[j] = 1; rec[j] = t + randexp(rng) / mu; ever += 1
            elseif sj == 1 && si == 0 && rand(rng) < beta
                state[i] = 1; rec[i] = t + randexp(rng) / mu; ever += 1
            end
        end
        finals[r] = ever / N
    end
    return finals
end

# mean/se convenience wrapper (kept for compatibility)
function run_sir(events, N, beta, mu, rng, reps)
    f = run_sir_samples(events, N, beta, mu, rng, reps)
    return mean(f), std(f) / sqrt(reps)
end

# ── SIS engine ───────────────────────────────────────────────────────────────
"""
    sis_prevalences(events, N, beta, mu, rng, reps; tail=0.4)

Per-run endemic prevalence (time-averaged over the final `tail` of the window).
Returns a vector of length `reps`.
"""
function sis_prevalences(events::Vector{Tuple{Float64,Int,Int}}, N::Int,
                         beta::Float64, mu::Float64, rng::AbstractRNG, reps::Int;
                         tail::Float64=0.4)
    t0   = events[1][1]
    tend = events[end][1]
    tcut = t0 + (1.0 - tail) * (tend - t0)
    prevs = zeros(reps)
    @inbounds for r in 1:reps
        state = zeros(Int8, N)
        rec   = fill(Inf, N)
        heap  = BinaryMinHeap{Tuple{Float64,Int}}()
        seed  = rand(rng, 1:N)
        state[seed] = 1
        rec[seed]   = t0 + randexp(rng) / mu
        push!(heap, (rec[seed], seed))
        infc = 1; psum = 0.0; pn = 0
        for (t, i, j) in events
            while !isempty(heap) && first(heap)[1] <= t
                rt, n = pop!(heap)
                if state[n] == 1 && rt == rec[n]
                    state[n] = 0; infc -= 1
                end
            end
            si, sj = state[i], state[j]
            if si == 1 && sj == 0 && rand(rng) < beta
                state[j] = 1; infc += 1
                rec[j] = t + randexp(rng) / mu
                push!(heap, (rec[j], j))
            elseif sj == 1 && si == 0 && rand(rng) < beta
                state[i] = 1; infc += 1
                rec[i] = t + randexp(rng) / mu
                push!(heap, (rec[i], i))
            end
            if t >= tcut
                psum += infc; pn += 1
            end
        end
        prevs[r] = (psum / max(pn, 1)) / N
    end
    return prevs
end

# ── pole sweep ───────────────────────────────────────────────────────────────
function run_pole(model::String, kind::String, N::Int, betas::Vector{Float64},
                  mu::Float64, reps::Int, G::Int; T::Float64=3000.0)
    E  = EVENTS_PER_NODE * N
    nb = length(betas)
    # Each graph realisation is generated ONCE and reused across all beta (paired across beta,
    # far cheaper at large N). samples[b] pools the per-run order parameter over all G graphs
    # (length G*reps, ordered graph-major: [g0 reps..., g1 reps..., ...]).
    samples = [Float64[] for _ in 1:nb]
    for g in 0:(G-1)
        ev  = gen_pole(kind, N, T, E, 10 + g)
        rng = MersenneTwister(1000 + g)
        for (b, be) in enumerate(betas)
            s = model == "sir" ? run_sir_samples(ev, N, be, mu, rng, reps) :
                                 sis_prevalences(ev, N, be, mu, rng, reps)
            append!(samples[b], s)
        end
    end

    out = Dict{String,Any}("model"=>model, "kind"=>kind, "N"=>N, "E"=>E, "mu"=>mu,
                           "reps"=>reps, "G"=>G, "T"=>T,
                           "events_per_node"=>EVENTS_PER_NODE, "tail"=>0.4,
                           "betas"=>betas,
                           "samples"=>samples)            # RAW per-run order parameter (future-proof)

    # moments of the order parameter (rho for SIS, final size for SIR)
    m1 = [mean(s)          for s in samples]
    m2 = [mean(abs2, s)    for s in samples]
    m4 = [mean(x->x^4, s)  for s in samples]
    vr = [var(s)           for s in samples]
    sv = [mean(>(0), s)    for s in samples]              # survival / outbreak fraction
    out["mean"]=m1; out["m2"]=m2; out["m4"]=m4; out["var"]=vr; out["surv"]=sv
    # prefactor-free dimensionless ratios -- their CROSSINGS across N locate beta_c
    out["ratio_m2"] = [m1[b] > 0 ? m2[b]/m1[b]^2 : 0.0 for b in 1:nb]              # <O^2>/<O>^2
    out["binder"]   = [m2[b] > 0 ? 1 - m4[b]/(3*m2[b]^2) : 0.0 for b in 1:nb]      # 1 - <O^4>/3<O^2>^2

    if model == "sis"
        out["rho"] = m1
        out["chi"] = [m1[b] > 1e-9 ? N*vr[b]/m1[b] : 0.0 for b in 1:nb]           # full-ensemble susceptibility
        # quasi-stationary (survival-conditioned) statistics -- remove the absorbing-state bimodality
        qm=Float64[]; qv=Float64[]; qchi=Float64[]
        for b in 1:nb
            surv = filter(>(0), samples[b])
            mq = isempty(surv) ? 0.0 : mean(surv)
            vq = length(surv) > 1 ? var(surv) : 0.0
            push!(qm, mq); push!(qv, vq); push!(qchi, mq > 0 ? N*vq/mq : 0.0)
        end
        out["qs_mean"]=qm; out["qs_var"]=qv; out["qs_chi"]=qchi
    else
        out["final_size"] = m1
        out["chi_sir"]    = [N*vr[b] for b in 1:nb]                                # outbreak-size fluctuations
        thr = 5.0 / N
        out["p_macro"]    = [mean(>(thr), samples[b]) for b in 1:nb]               # macroscopic-outbreak probability
    end
    return out
end

# ── CLI ──────────────────────────────────────────────────────────────────────
function main()
    model = length(ARGS) >= 1 ? ARGS[1] : "sis"
    Ns    = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [40, 60, 90]

    # keyword args: reps=55 G=3 recov=60
    kw = Dict{String,String}()
    for a in ARGS[3:end]
        if occursin("=", a)
            k, v = split(a, "="; limit=2)
            kw[String(k)] = String(v)
        end
    end
    reps  = parse(Int,     get(kw, "reps",  "55"))
    G     = parse(Int,     get(kw, "G",     "3"))
    mu    = 1.0 / parse(Float64, get(kw, "recov", "60"))

    # default grids bracket the thresholds so dimensionless-ratio CROSSINGS across N can locate
    # beta_c; override with e.g. betas=0.05,0.1,0.15,...
    default_betas = model == "sir" ?
        [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.24, 0.30] :
        [0.06, 0.10, 0.14, 0.18, 0.24, 0.30, 0.38, 0.48, 0.62, 0.80]
    betas = haskey(kw, "betas") ? parse.(Float64, split(kw["betas"], ",")) : default_betas

    outdir = joinpath(@__DIR__, "..", "results"); mkpath(outdir)

    for N in Ns
        for kind in ("node", "link", "poisson")
            d   = run_pole(model, kind, N, betas, mu, reps, G)
            fn  = joinpath(outdir, "poles_$(model)_N$(N)_$(kind).json")
            open(fn, "w") do io
                JSON.print(io, d)          # compact: files carry raw per-run samples
            end
            key = model == "sir" ? "final_size" : "chi"
            vals = join([@sprintf("%.3f", v) for v in d[key]], " ")
            println("$model N=$(lpad(N,5)) $(rpad(kind,8)) $key: $vals")
        end
    end
end

main()
