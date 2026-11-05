"""
    Inference

MLE and identifiability diagnostics for the three-mechanism model. Faithful port of
`inference/inference.py`; see that file for the derivation.

Structure of the likelihood (initiator observed):

    log L = TIMING + SELECTION

    TIMING    = sum_e log(a_{i_e} + n_node S_{i_e}(t_e^-)) - sum_i [a_i T + n_node C_i]
    C_i       = sum_{t_k in H_i} (1 - exp(-gamma_n (T - t_k)))
    SELECTION = sum_e [log(b_{j_e} + kappa m_{i_e j_e}(t_e^-))
                       - log((B - b_{i_e}) + kappa M_{i_e}(t_e^-))]

TIMING involves only `(a, n_node, gamma_n)`, SELECTION only `(b, kappa, gamma_e)`. The
blocks share no parameters, so with initiator labels the observed information is exactly
block-diagonal between node memory and tie memory: A and B cannot trade off. Without the
labels they mix, and `fit_latent_initiator` measures by how much.

The `N` baseline activities are profiled out exactly by bisection on the per-node score
equation `sum_{e: i_e=i} 1/(a_i + n_node S_e) = T`, whose root lies in `[0, K_i/T]`.

NOT EXECUTED during development (no Julia available in the authoring environment).
Run `julia --project=. scripts/check_parity.jl` first; it reproduces a Python-generated
fixture to 1e-8 and will catch any porting error.
"""
module Inference

using Optim
using LinearAlgebra
using Random
using Statistics

export TimingStats, SelectionStats, FitResult,
       replay_timing, replay_selection, profile_activities,
       timing_nll, selection_nll, fit_observed_initiator, fit_latent_initiator,
       profile_likelihood, likelihood_ratio_test, time_resolution, summary_string

const Contact = Tuple{Float64,Int,Int}

# ---------------------------------------------------------------------------------
# Sufficient-statistic replays: the only O(E) passes.
# ---------------------------------------------------------------------------------

struct TimingStats
    S_at_event::Vector{Float64}
    initiator::Vector{Int}
    compensator::Vector{Float64}
    counts::Vector{Float64}
    T::Float64
end

"""
    replay_timing(contacts, initiator, N, gamma_n, T)

One O(E) pass giving `S_{i_e}(t_e^-)` and the compensator terms `C_i`.
"""
function replay_timing(contacts::AbstractVector{Contact}, initiator::AbstractVector{Int},
                       N::Int, gamma_n::Float64, T::Float64)
    gamma_n > 0 || error("gamma_n must be positive, got $gamma_n")
    E = length(contacts)
    S = zeros(N); S_at = zeros(E); C = zeros(N); counts = zeros(N)
    t_prev = 0.0
    @inbounds for e in 1:E
        t = contacts[e][1]
        t < t_prev && error("contacts must be sorted by time")
        S .*= exp(-gamma_n * (t - t_prev))
        t_prev = t
        i = initiator[e]
        S_at[e] = S[i]
        S[i] += gamma_n
        counts[i] += 1.0
        C[i] += 1.0 - exp(-gamma_n * (T - t))
    end
    return TimingStats(S_at, collect(initiator), C, counts, T)
end

struct SelectionStats
    m_pair::Vector{Float64}
    M_row::Matrix{Float64}      # E x 2
    endpoints::Matrix{Int}      # E x 2
end

"""
    replay_selection(contacts, N, gamma_e)

One O(E) pass giving the dyad memory and the dyad-memory row sums at each event.
Label-free, so it is reused across every imputation draw.
"""
function replay_selection(contacts::AbstractVector{Contact}, N::Int, gamma_e::Float64)
    gamma_e > 0 || error("gamma_e must be positive, got $gamma_e")
    E = length(contacts)
    M = zeros(N)
    nbr = [Dict{Int,Tuple{Float64,Float64}}() for _ in 1:N]
    m_pair = zeros(E); M_row = zeros(E, 2); endpoints = zeros(Int, E, 2)
    t_prev = 0.0
    @inbounds for e in 1:E
        (t, i, j) = contacts[e]
        t < t_prev && error("contacts must be sorted by time")
        M .*= exp(-gamma_e * (t - t_prev))
        t_prev = t
        ent = get(nbr[i], j, nothing)
        m_now = ent === nothing ? 0.0 : ent[1] * exp(-gamma_e * (t - ent[2]))
        m_pair[e] = m_now
        M_row[e, 1] = M[i]; M_row[e, 2] = M[j]
        endpoints[e, 1] = i; endpoints[e, 2] = j
        m_new = m_now + 1.0             # unit-peak tie kernel, matches the simulator
        nbr[i][j] = (m_new, t); nbr[j][i] = (m_new, t)
        M[i] += 1.0; M[j] += 1.0
    end
    return SelectionStats(m_pair, M_row, endpoints)
end

# ---------------------------------------------------------------------------------
# TIMING block
# ---------------------------------------------------------------------------------

"""
    profile_activities(stats, n_node, N)

Exact profile MLE of the baseline activities. Bisection on `[0, K_i/T]`; the left
endpoint is the boundary solution taken when node memory alone already over-explains
the node's events.
"""
function profile_activities(stats::TimingStats, n_node::Float64, N::Int;
                            tol::Float64=1e-12, max_iter::Int=200)
    T = stats.T
    a = zeros(N)
    idx = [Int[] for _ in 1:N]
    for e in eachindex(stats.initiator)
        push!(idx[stats.initiator[e]], e)
    end
    for i in 1:N
        K = stats.counts[i]
        K == 0 && continue
        base = [n_node * stats.S_at_event[e] for e in idx[i]]
        deriv(x) = sum(1.0 / (x + bb) for bb in base) - T
        if minimum(base) > 0 && deriv(0.0) <= 0
            a[i] = 0.0
            continue
        end
        lo, hi = 0.0, K / T
        if deriv(hi) > 0            # numerical safety; unreachable analytically
            a[i] = hi
            continue
        end
        for _ in 1:max_iter
            mid = 0.5 * (lo + hi)
            deriv(mid) > 0 ? (lo = mid) : (hi = mid)
            hi - lo < tol * max(1.0, hi) && break
        end
        a[i] = 0.5 * (lo + hi)
    end
    return a
end

"""
    timing_nll(n_node, gamma_n, contacts, initiator, N, T)

Concentrated negative log-likelihood of the TIMING block. Returns `Inf` outside the
stationary region rather than throwing, so optimisers can walk into it safely.
"""
function timing_nll(n_node::Float64, gamma_n::Float64, contacts::AbstractVector{Contact},
                    initiator::AbstractVector{Int}, N::Int, T::Float64)
    (0.0 <= n_node < 1.0 && gamma_n > 0) || return Inf
    stats = replay_timing(contacts, initiator, N, gamma_n, T)
    a = profile_activities(stats, n_node, N)
    nll = 0.0
    @inbounds for e in eachindex(stats.S_at_event)
        rate = a[stats.initiator[e]] + n_node * stats.S_at_event[e]
        rate <= 0 && return Inf
        nll -= log(rate)
    end
    nll += sum(a) * T + n_node * sum(stats.compensator)
    return isfinite(nll) ? nll : Inf
end

function timing_nll_with_activities(n_node::Float64, gamma_n::Float64,
                                    contacts::AbstractVector{Contact},
                                    initiator::AbstractVector{Int}, N::Int, T::Float64)
    stats = replay_timing(contacts, initiator, N, gamma_n, T)
    return profile_activities(stats, n_node, N)
end

# ---------------------------------------------------------------------------------
# SELECTION block
# ---------------------------------------------------------------------------------

function selection_nll_from_stats(kappa::Float64, sel::SelectionStats,
                                  side::AbstractVector{Int}, b::Vector{Float64})
    kappa < 0 && return Inf
    B = sum(b)
    nll = 0.0
    @inbounds for e in eachindex(side)
        c = side[e] + 1                   # 1 or 2
        i = sel.endpoints[e, c]
        j = sel.endpoints[e, 3 - c]
        num = b[j] + kappa * sel.m_pair[e]
        den = (B - b[i]) + kappa * sel.M_row[e, c]
        (num <= 0 || den <= 0) && return Inf
        nll -= log(num) - log(den)
    end
    return nll
end

"""
    estimate_attractiveness(kappa, sel, side, N)

MM fixed point for `b`, renormalised to mean 1 each sweep. Monotonicity is checked; a
sweep that fails to improve the objective stops the iteration rather than diverging.
"""
function estimate_attractiveness(kappa::Float64, sel::SelectionStats,
                                 side::AbstractVector{Int}, N::Int;
                                 n_iter::Int=60, tol::Float64=1e-10)
    E = length(side)
    b = ones(N)
    prev = selection_nll_from_stats(kappa, sel, side, b)
    for _ in 1:n_iter
        B = sum(b)
        share = zeros(N); exposure = zeros(N)
        inv_den = zeros(E)
        @inbounds for e in 1:E
            c = side[e] + 1
            i = sel.endpoints[e, c]; j = sel.endpoints[e, 3 - c]
            num = b[j] + kappa * sel.m_pair[e]
            den = (B - b[i]) + kappa * sel.M_row[e, c]
            share[j] += b[j] / max(num, 1e-300)
            inv_den[e] = 1.0 / max(den, 1e-300)
        end
        tot_inv = sum(inv_den)
        fill!(exposure, tot_inv)
        @inbounds for e in 1:E
            c = side[e] + 1
            exposure[sel.endpoints[e, c]] -= inv_den[e]
        end
        b_new = [exposure[k] > 0 ? max(share[k] / exposure[k], 1e-12) : b[k] for k in 1:N]
        b_new ./= (sum(b_new) / N)
        cur = selection_nll_from_stats(kappa, sel, side, b_new)
        (!isfinite(cur) || cur > prev + 1e-9) && break
        if abs(prev - cur) < tol * max(1.0, abs(prev))
            b = b_new
            break
        end
        b, prev = b_new, cur
    end
    return b
end

function selection_nll(kappa::Float64, gamma_e::Float64,
                       contacts::AbstractVector{Contact}, side::AbstractVector{Int},
                       N::Int; fit_b::Bool=false)
    (kappa >= 0 && gamma_e > 0) || return Inf
    sel = replay_selection(contacts, N, gamma_e)
    b = fit_b ? estimate_attractiveness(kappa, sel, side, N) : ones(N)
    return selection_nll_from_stats(kappa, sel, side, b)
end

# ---------------------------------------------------------------------------------
# Data-driven bounds and starts
# ---------------------------------------------------------------------------------

"""
    time_resolution(contacts) -> (resolution, n_ties)

Smallest strictly positive inter-event gap and the number of exactly simultaneous
events. Ties make the timing likelihood DIVERGE as `gamma_n -> Inf` (a second event at
zero lag sees self-excitation `gamma_n` times the number of coincident events, while
the compensator stays finite), so decay rates are capped at `1/resolution` throughout.
That cap is a statement about what the sampling grid can resolve, not a hack.
"""
function time_resolution(contacts::AbstractVector{Contact})
    ts = [c[1] for c in contacts]
    d = diff(ts)
    n_ties = count(==(0.0), d)
    pos = filter(>(0.0), d)
    res = isempty(pos) ? NaN : minimum(pos)
    return res, n_ties
end

function gamma_bounds(contacts::AbstractVector{Contact}, T::Float64; verbose::Bool=false)
    res, n_ties = time_resolution(contacts)
    g_lo = 1.0 / (10.0 * T)
    g_hi = (isnan(res) || res <= 0) ? length(contacts) / T : 1.0 / res
    g_hi <= g_lo && (g_hi = g_lo * 1e3)
    if verbose && n_ties > 0
        println("  [fit] $n_ties exactly simultaneous events detected; capping decay ",
                "rates at 1/$res = $g_hi per time unit. See time_resolution().")
    end
    return g_lo, g_hi
end

"""
Cheapest point of a coarse 2-D grid, used to seed the optimiser. Without it the fit
fails silently on real data: if gamma starts at 1 while the data live on a scale of
hours, every memory term is numerically zero, the gradient vanishes, and the fit
reports "no memory" regardless of the truth.
"""
function coarse_start(f, grid_a, grid_b)
    best = (grid_a[1], grid_b[1]); best_val = Inf
    for va in grid_a, vb in grid_b
        v = f(va, vb)
        if isfinite(v) && v < best_val
            best = (va, vb); best_val = v
        end
    end
    return best
end

# ---------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------

struct FitResult
    n_node::Float64
    gamma_n::Float64
    kappa::Float64
    gamma_e::Float64
    activities::Vector{Float64}
    attractiveness::Vector{Float64}
    nll_timing::Float64
    nll_selection::Float64
    se::Dict{String,Float64}
    corr_n_kappa::Float64
    n_events::Int
    T::Float64
    notes::String
end

nll(r::FitResult) = r.nll_timing + r.nll_selection

function summary_string(r::FitResult)
    pm(name, val) = (s = get(r.se, name, NaN);
                     isfinite(s) ? "$(round(val, digits=4)) +/- $(round(s, digits=4))" :
                                   "$(round(val, digits=4)) (se n/a)")
    join([
        "n_node  = $(pm("n_node", r.n_node))   [node branching ratio, A-memory]",
        "gamma_n = $(pm("gamma_n", r.gamma_n))   [node memory decay, 1/time]",
        "kappa   = $(pm("kappa", r.kappa))   [tie reinforcement, B-memory]",
        "gamma_e = $(pm("gamma_e", r.gamma_e))   [tie memory decay, 1/time]",
        "corr(n_node, kappa) = $(round(r.corr_n_kappa, digits=3))",
        "nll = $(round(nll(r), digits=3)) on $(r.n_events) events over T = $(r.T)",
        isempty(r.notes) ? "" : "notes: $(r.notes)",
    ], "\n")
end

function split_events(events::AbstractVector{<:Tuple{Float64,Int,Int}})
    E = length(events)
    contacts = Vector{Contact}(undef, E)
    initiator = zeros(Int, E)
    side = zeros(Int, E)
    @inbounds for e in 1:E
        (t, i, j) = events[e]
        initiator[e] = i
        if i < j
            contacts[e] = (t, i, j); side[e] = 0
        else
            contacts[e] = (t, j, i); side[e] = 1
        end
    end
    return contacts, initiator, side
end

function _hessian2(f, x::Vector{Float64}, step::Vector{Float64})
    H = fill(NaN, 2, 2)
    f0 = f(x)
    isfinite(f0) || return H
    for p in 1:2
        xp = copy(x); xm = copy(x)
        xp[p] += step[p]; xm[p] -= step[p]
        H[p, p] = (f(xp) - 2f0 + f(xm)) / step[p]^2
    end
    xpp = copy(x); xpm = copy(x); xmp = copy(x); xmm = copy(x)
    xpp[1] += step[1]; xpp[2] += step[2]
    xpm[1] += step[1]; xpm[2] -= step[2]
    xmp[1] -= step[1]; xmp[2] += step[2]
    xmm[1] -= step[1]; xmm[2] -= step[2]
    H[1, 2] = H[2, 1] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4step[1] * step[2])
    return H
end

function _safe_cov(H::Matrix{Float64})
    all(isfinite, H) || return fill(NaN, size(H))
    cov = try
        inv(H)
    catch
        return fill(NaN, size(H))
    end
    any(diag(cov) .<= 0) && return fill(NaN, size(H))
    return cov
end

_se(cov, k) = (v = cov[k, k]; (isfinite(v) && v > 0) ? sqrt(v) : NaN)

"""
    fit_observed_initiator(events, N, T; fit_b=false, x0=nothing, verbose=false)

Exact MLE when the initiator of every event is observed. TIMING and SELECTION are
optimised independently because they share no parameters, so `corr_n_kappa` is exactly
zero by construction.
"""
function fit_observed_initiator(events::AbstractVector{<:Tuple{Float64,Int,Int}},
                                N::Int, T::Real; fit_b::Bool=false,
                                x0::Union{Nothing,Vector{Float64}}=nothing,
                                verbose::Bool=false)
    length(events) >= 10 || error("need at least 10 events to fit, got $(length(events))")
    T = Float64(T)
    T > 0 || error("T must be positive, got $T")

    contacts, initiator, side = split_events(events)
    g_lo, g_hi = gamma_bounds(contacts, T; verbose=verbose)

    f_t(n, g) = timing_nll(n, g, contacts, initiator, N, T)
    f_s(k, g) = selection_nll(k, g, contacts, side, N; fit_b=fit_b)

    if x0 === nothing
        g_scan = exp.(range(log(g_lo), log(g_hi); length=9))
        st = coarse_start(f_t, [0.05, 0.2, 0.4, 0.6, 0.8], g_scan)
        ss = coarse_start(f_s, [0.01, 0.3, 3.0, 30.0, 300.0], g_scan)
        x0 = [st[1], st[2], ss[1], ss[2]]
    end

    lg = (log(g_lo), log(g_hi))
    clampl(v) = min(max(log(max(v, g_lo)), lg[1]), lg[2])

    r_t = optimize(z -> f_t(z[1], exp(z[2])),
                   [0.0, lg[1]], [0.999, lg[2]],
                   [min(max(x0[1], 0.0), 0.95), clampl(x0[2])],
                   Fminbox(LBFGS()), Optim.Options(iterations=200))
    r_s = optimize(z -> f_s(exp(z[1]), exp(z[2])),
                   [log(1e-8), lg[1]], [log(1e6), lg[2]],
                   [log(max(x0[3], 1e-6)), clampl(x0[4])],
                   Fminbox(LBFGS()), Optim.Options(iterations=200))

    zt = Optim.minimizer(r_t); zs = Optim.minimizer(r_s)
    n_hat = zt[1]; gn_hat = exp(zt[2]); kp_hat = exp(zs[1]); ge_hat = exp(zs[2])

    a_hat = timing_nll_with_activities(n_hat, gn_hat, contacts, initiator, N, T)
    sel = replay_selection(contacts, N, ge_hat)
    b_hat = fit_b ? estimate_attractiveness(kp_hat, sel, side, N) : ones(N)

    H_t = _hessian2(z -> f_t(z[1], z[2]), [n_hat, gn_hat],
                    [max(1e-3, 0.02 * max(n_hat, 1e-2)), 0.02 * gn_hat])
    H_s = _hessian2(z -> f_s(z[1], z[2]), [kp_hat, ge_hat],
                    [0.02 * max(kp_hat, 1e-3), 0.02 * ge_hat])
    cov_t = _safe_cov(H_t); cov_s = _safe_cov(H_s)

    se = Dict("n_node" => _se(cov_t, 1), "gamma_n" => _se(cov_t, 2),
              "kappa" => _se(cov_s, 1), "gamma_e" => _se(cov_s, 2))

    return FitResult(n_hat, gn_hat, kp_hat, ge_hat, a_hat, b_hat,
                     Optim.minimum(r_t), Optim.minimum(r_s), se, 0.0,
                     length(events), T,
                     "initiator observed; timing and selection blocks are exactly orthogonal")
end

"""
    impute_initiators(ordered, N, T, theta, rng)

Forward pass drawing an initiator label per contact, with `P(i initiated)` proportional
to `rho_i(t) * pi_{j|i}(t)` under the state accumulated from labels already drawn.
"""
function impute_initiators(ordered::AbstractVector{Contact}, N::Int, T::Float64,
                           theta::Vector{Float64}, rng)
    n_node, gamma_n, kappa, gamma_e = theta
    S = zeros(N); M = zeros(N)
    nbr = [Dict{Int,Tuple{Float64,Float64}}() for _ in 1:N]
    counts = zeros(N)
    for (_, i, j) in ordered
        counts[i] += 0.5; counts[j] += 0.5
    end
    a = max.(counts ./ max(T, 1e-12), 1e-9)
    b_total = Float64(N)
    side = zeros(Int, length(ordered))
    t_prev = 0.0
    @inbounds for e in eachindex(ordered)
        (t, i, j) = ordered[e]
        dt = t - t_prev
        S .*= exp(-gamma_n * dt); M .*= exp(-gamma_e * dt)
        t_prev = t
        ent = get(nbr[i], j, nothing)
        m_now = ent === nothing ? 0.0 : ent[1] * exp(-gamma_e * (t - ent[2]))
        num = 1.0 + kappa * m_now
        w_i = (a[i] + n_node * S[i]) * num / max((b_total - 1.0) + kappa * M[i], 1e-300)
        w_j = (a[j] + n_node * S[j]) * num / max((b_total - 1.0) + kappa * M[j], 1e-300)
        tot = w_i + w_j
        s = (tot <= 0 || rand(rng) < w_i / tot) ? 0 : 1
        side[e] = s
        init = s == 0 ? i : j
        S[init] += gamma_n
        m_new = m_now + 1.0
        nbr[i][j] = (m_new, t); nbr[j][i] = (m_new, t)
        M[i] += 1.0; M[j] += 1.0
    end
    return side
end

"""
    fit_latent_initiator(contacts, N, T; n_imputations=6, n_em=8, fit_b=false, seed=0)

Monte Carlo EM when only undirected contacts are observed.

This is Monte Carlo EM, not exact EM: sequential imputation is a filtering approximation
to the full smoothing distribution over labels, since a label drawn at event `e` is
never revised in light of later events. On synthetic data the size of that approximation
can be measured by comparing against `fit_observed_initiator` with the TRUE labels. Do
not read the observed-vs-latent gap as pure information loss without that check.

Standard errors combine within- and between-imputation variance by Rubin's rules.
"""
function fit_latent_initiator(contacts::AbstractVector{<:Tuple{Float64,Int,Int}},
                              N::Int, T::Real; n_imputations::Int=6, n_em::Int=8,
                              fit_b::Bool=false, seed::Int=0,
                              x0::Union{Nothing,Vector{Float64}}=nothing)
    length(contacts) >= 10 || error("need at least 10 contacts, got $(length(contacts))")
    T = Float64(T)
    rng = MersenneTwister(seed)
    ordered = Contact[(t, min(i, j), max(i, j)) for (t, i, j) in contacts]

    theta = if x0 === nothing
        flip = rand(rng, 0:1, length(ordered))
        seed_events = [(ordered[e][1],
                        flip[e] == 0 ? ordered[e][2] : ordered[e][3],
                        flip[e] == 0 ? ordered[e][3] : ordered[e][2])
                       for e in eachindex(ordered)]
        r0 = fit_observed_initiator(seed_events, N, T; fit_b=fit_b, verbose=true)
        [r0.n_node, r0.gamma_n, r0.kappa, r0.gamma_e]
    else
        copy(x0)
    end

    draws = Vector{Vector{Float64}}()
    ses = Vector{Vector{Float64}}()
    last = nothing

    for sweep in 1:n_em
        sweep_theta = Vector{Vector{Float64}}()
        for _ in 1:n_imputations
            side = impute_initiators(ordered, N, T, theta, rng)
            events = [(ordered[e][1],
                       side[e] == 0 ? ordered[e][2] : ordered[e][3],
                       side[e] == 0 ? ordered[e][3] : ordered[e][2])
                      for e in eachindex(ordered)]
            res = fit_observed_initiator(events, N, T; fit_b=fit_b, x0=theta)
            push!(sweep_theta, [res.n_node, res.gamma_n, res.kappa, res.gamma_e])
            if sweep >= n_em - 2
                push!(draws, [res.n_node, res.gamma_n, res.kappa, res.gamma_e])
                push!(ses, [res.se["n_node"], res.se["gamma_n"],
                            res.se["kappa"], res.se["gamma_e"]])
            end
            last = res
        end
        theta = [mean(getindex.(sweep_theta, k)) for k in 1:4]
    end

    D = reduce(hcat, draws)'          # n_draws x 4
    Sm = reduce(hcat, ses)'
    names = ["n_node", "gamma_n", "kappa", "gamma_e"]
    se = Dict{String,Float64}()
    for k in 1:4
        w = filter(isfinite, Sm[:, k] .^ 2)
        within = isempty(w) ? NaN : mean(w)
        between = size(D, 1) > 1 ? var(D[:, k]) : 0.0
        tot = within + (1 + 1 / size(D, 1)) * between
        se[names[k]] = (isfinite(tot) && tot > 0) ? sqrt(tot) : NaN
    end
    corr = (size(D, 1) > 2 && std(D[:, 1]) > 0 && std(D[:, 3]) > 0) ?
           cor(D[:, 1], D[:, 3]) : NaN

    return FitResult(mean(D[:, 1]), mean(D[:, 2]), mean(D[:, 3]), mean(D[:, 4]),
                     last.activities, last.attractiveness,
                     last.nll_timing, last.nll_selection, se, corr,
                     length(contacts), T,
                     "initiator latent; Monte Carlo EM with sequential imputation " *
                     "($n_em sweeps x $n_imputations draws), SEs pooled by Rubin's rules. " *
                     "corr(n_node, kappa) is the across-imputation correlation, i.e. the " *
                     "mechanism confound induced by the missing labels.")
end

"""
    profile_likelihood(events, N, T, which, grid; fit_b=false)

Profile NLL over `which in ("n_node", "kappa")`, re-optimising everything else at each
grid point. The CURVATURE of this curve, not a point estimate, is what answers the
identifiability question.
"""
function profile_likelihood(events::AbstractVector{<:Tuple{Float64,Int,Int}},
                            N::Int, T::Real, which::String, grid::AbstractVector{<:Real};
                            fit_b::Bool=false)
    T = Float64(T)
    contacts, initiator, side = split_events(events)
    g_lo, g_hi = gamma_bounds(contacts, T)
    out = zeros(length(grid))
    for (k, v) in enumerate(grid)
        f = which == "n_node" ?
            (z -> timing_nll(Float64(v), exp(z[1]), contacts, initiator, N, T)) :
            which == "kappa" ?
            (z -> selection_nll(Float64(v), exp(z[1]), contacts, side, N; fit_b=fit_b)) :
            error("`which` must be \"n_node\" or \"kappa\", got $which")
        r = optimize(f, [log(g_lo)], [log(g_hi)], [log(sqrt(g_lo * g_hi))],
                     Fminbox(LBFGS()), Optim.Options(iterations=100))
        out[k] = Optim.minimum(r)
    end
    return out
end

"""
    likelihood_ratio_test(nll_full, nll_restricted, df)

p-value of `2(nll_restricted - nll_full)` against `chi2_df`. Both nulls used in this
project (`kappa = 0`, `n_node = 0`) sit on the parameter-space boundary, where the
asymptotic null is a chi-bar-squared mixture, so this p-value is CONSERVATIVE by roughly
a factor of two for `df = 1`. Reported uncorrected rather than silently adjusted.
"""
function likelihood_ratio_test(nll_full::Float64, nll_restricted::Float64, df::Int)
    stat = 2.0 * (nll_restricted - nll_full)
    stat < 0 && return NaN
    # chi2 survival function without a Distributions.jl dependency (df = 1 or 2 only)
    if df == 1
        return erfc(sqrt(stat / 2))
    elseif df == 2
        return exp(-stat / 2)
    else
        error("only df = 1 or 2 supported without Distributions.jl, got df=$df")
    end
end

erfc(x::Float64) = begin
    # Numerical Recipes rational approximation, |error| < 1.2e-7. Adequate for reporting
    # a p-value that is already conservative for boundary nulls.
    z = abs(x); t = 1.0 / (1.0 + 0.5z)
    ans = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
          t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
          t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    x >= 0 ? ans : 2.0 - ans
end

end # module
