#!/usr/bin/env julia
# =====================================================================================
# Metropolis-within-Gibbs initiator-label sampler for the latent-initiator confound test,
# fast enough for FULL proximity records. This is the demanding inner loop of the E-step
# check (mwg_real.py is the Python reference on small subsamples). It samples the label
# posterior P(sigma | data, theta_hat) at the FIXED observed-data MLE theta_hat (n_node,
# gamma_n, kappa, gamma_e, activities a, attractiveness b), which the Python driver supplies.
# It writes label draws only; the per-draw (n_node, kappa) fit is done back in the tested
# Python estimator, so nothing correctness-critical is reimplemented here.
#
# Likelihood (matches inference.py exactly):
#   timing  : sum_e log(a_{i_e} + n_node * S_{i_e}(t_e^-)),  S_i(t)=sum_{init by i} gamma_n e^{-gamma_n(t-t_k)}
#             (the compensator and baseline are label-independent and drop out of sigma-ratios)
#   selection: sum_e [ log(b_{j_e} + kappa*m_pair_e) - log((sum b - b_{i_e}) + kappa*Mrow_{i_e,e}) ]
#             with m_pair and Mrow label-free (unit-peak tie kernel), precomputed once.
#
# I/O contract (plain text, no packages beyond stdlib):
#   in : events.tsv (E x 3: t u v, nodes 0-indexed), scalars.tsv (n_node gamma_n kappa
#        gamma_e N T M burnin seed thin, one per line), a.tsv (N), b.tsv (N)
#   out: filter.tsv, smoother.tsv  (each M x E, entries in {0,1}; 0 => initiator = u = col2)
#
# Runtime: O(E^2) per sweep (full timing recompute per flip; selection update is O(1)).
# Fine for E up to ~1.5e4 (minutes). For much larger E, recompute timing from the flipped
# event forward instead of from scratch.
# =====================================================================================
using DelimitedFiles, Random

function main()
    indir, outdir = ARGS[1], ARGS[2]
    ev  = readdlm(joinpath(indir, "events.tsv"))
    sc  = vec(readdlm(joinpath(indir, "scalars.tsv")))
    a   = vec(readdlm(joinpath(indir, "a.tsv")))
    b   = vec(readdlm(joinpath(indir, "b.tsv")))
    t = Float64.(ev[:, 1])
    u = Int.(ev[:, 2]) .+ 1                     # 0-indexed file -> 1-indexed Julia
    v = Int.(ev[:, 3]) .+ 1
    a = max.(a, 1e-12)                          # floor zero-MLE activities to keep rates finite
    n_node, gamma_n, kappa, gamma_e = sc[1], sc[2], sc[3], sc[4]
    N, T = Int(sc[5]), sc[6]
    M, burnin, seed, thin = Int(sc[7]), Int(sc[8]), Int(sc[9]), Int(sc[10])
    E = length(t); Bsum = sum(b)

    # ---- label-free selection statistics: m_pair[e], Mrow at both endpoints ----------
    mpair = zeros(E); MrowU = zeros(E); MrowV = zeros(E)
    Mrows = zeros(N)
    pairmem = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    tprev = 0.0
    for e in 1:E
        Mrows .*= exp(-gamma_e * (t[e] - tprev)); tprev = t[e]
        iu, iv = u[e], v[e]; key = iu < iv ? (iu, iv) : (iv, iu)
        pm = get(pairmem, key, (0.0, t[e]))
        m_now = pm[1] * exp(-gamma_e * (t[e] - pm[2]))
        mpair[e] = m_now; MrowU[e] = Mrows[iu]; MrowV[e] = Mrows[iv]
        pairmem[key] = (m_now + 1.0, t[e])
        Mrows[iu] += 1.0; Mrows[iv] += 1.0
    end

    sel_term(e, s) = begin                       # s=0: init=u,part=v ; s=1: init=v,part=u
        i, j, Mi = s == 0 ? (u[e], v[e], MrowU[e]) : (v[e], u[e], MrowV[e])
        log(b[j] + kappa * mpair[e]) - log((Bsum - b[i]) + kappa * Mi)
    end

    Sval = zeros(N); Slast = zeros(N)
    timing_logsum(sigma) = begin                 # O(E), lazy per-node decay
        fill!(Sval, 0.0); fill!(Slast, 0.0); tot = 0.0
        @inbounds for e in 1:E
            i = sigma[e] == 0 ? u[e] : v[e]
            Sval[i] *= exp(-gamma_n * (t[e] - Slast[i])); Slast[i] = t[e]
            tot += log(a[i] + n_node * Sval[i])
            Sval[i] += gamma_n
        end
        tot
    end

    # ---- sequential-imputation filter (M independent causal draws) -------------------
    filter_draw(rng) = begin
        sigma = Vector{Int}(undef, E)
        fill!(Sval, 0.0); fill!(Slast, 0.0)
        @inbounds for e in 1:E
            iu, iv = u[e], v[e]
            Su = Sval[iu] * exp(-gamma_n * (t[e] - Slast[iu]))
            Sv = Sval[iv] * exp(-gamma_n * (t[e] - Slast[iv]))
            c0 = log(a[iu] + n_node * Su) + sel_term(e, 0)
            c1 = log(a[iv] + n_node * Sv) + sel_term(e, 1)
            mmax = max(c0, c1)
            p0 = mmax == -Inf ? 0.5 :
                 exp(c0 - mmax) / (exp(c0 - mmax) + exp(c1 - mmax))
            s = rand(rng) < p0 ? 0 : 1
            Sval[iu] = Su; Slast[iu] = t[e]; Sval[iv] = Sv; Slast[iv] = t[e]
            Sval[s == 0 ? iu : iv] += gamma_n
            sigma[e] = s
        end
        sigma
    end

    rng = MersenneTwister(seed)
    FIL = Array{Int}(undef, M, E)
    for m in 1:M
        FIL[m, :] = filter_draw(MersenneTwister(seed + 100 + m))
    end

    # ---- Metropolis-within-Gibbs smoother (one chain, M samples after burnin) --------
    sigma = filter_draw(MersenneTwister(seed + 7))
    sel_tot = sum(sel_term(e, sigma[e]) for e in 1:E)
    tim_tot = timing_logsum(sigma)
    SMO = Array{Int}(undef, M, E)
    kept = 0
    total_sweeps = burnin + M * thin
    flips = 0; tries = 0
    for sweep in 1:total_sweeps
        @inbounds for e in 1:E
            snew = 1 - sigma[e]
            d_sel = sel_term(e, snew) - sel_term(e, sigma[e])
            old = sigma[e]; sigma[e] = snew
            tim_new = timing_logsum(sigma)        # O(E) exact recompute
            dL = (tim_new - tim_tot) + d_sel
            tries += 1
            if log(rand(rng) + 1e-300) < dL
                tim_tot = tim_new; sel_tot += d_sel; flips += 1
            else
                sigma[e] = old                    # reject
            end
        end
        if sweep > burnin && (sweep - burnin) % thin == 0 && kept < M
            kept += 1; SMO[kept, :] = sigma
        end
    end

    writedlm(joinpath(outdir, "filter.tsv"), FIL)
    writedlm(joinpath(outdir, "smoother.tsv"), SMO)
    open(joinpath(outdir, "smoother_acceptance.txt"), "w") do io
        write(io, string(flips / max(tries, 1)))
    end
    println("mwg_full.jl: wrote $(M) filter and $(kept) smoother draws over E=$E events.")
end

main()
