"""
    GenerativeModels

Initiator-resolved generative model for temporal networks with three mechanisms.
Faithful port of `models/generative_models.py`; see that file for the full derivation
and for why the previous `alpha`-mixing parametrisation was not identified.

An event is a triple `(t, i, j)` with `i` the initiator. Three mechanisms:

  A0  node baseline activity  `a_i`      memoryless activity-driven
  A1  node self-excitation    `n_node`   burstiness in *when* `i` acts
  B   tie reinforcement       `kappa`    memory in *which* tie `i` reuses

    rho_i(t)    = a_i + n_node * sum_{t_k in H_i} gamma_n exp(-gamma_n (t - t_k))
    m_ij(t)     = sum_{t_k in H_ij} exp(-gamma_e (t - t_k))
    pi_{j|i}(t) = (b_j + kappa m_ij(t)) / sum_{l != i} (b_l + kappa m_il(t))

The node kernel has unit INTEGRAL, so `n_node` is a branching ratio and stationarity is
exactly `n_node < 1`. The tie kernel has unit PEAK, so `kappa` reads as "how many times
more attractive a just-contacted partner is" against the normalisation `mean(b) = 1`.
Do not change either normalisation without changing `Inference.jl` to match; the parity
check in `scripts/check_parity.jl` exists to catch exactly that kind of drift.

NOT EXECUTED during development (no Julia available in the authoring environment).
Run `julia --project=. scripts/check_parity.jl` before trusting any result from here.
"""
module GenerativeModels

using Random

export MechanismParams, TemporalNetworkModel, ExcitationState,
       simulate, strip_initiators, summarise, node_intensities

const Event = Tuple{Float64,Int,Int}   # (time, initiator, partner), 1-based nodes

"""
    MechanismParams(n_node, gamma_n, kappa, gamma_e)

Free parameters. Construction fails loudly on the explosive regime (`n_node >= 1`),
on non-positive decay rates, and on negative `kappa`.
"""
struct MechanismParams
    n_node::Float64
    gamma_n::Float64
    kappa::Float64
    gamma_e::Float64

    function MechanismParams(n_node::Real, gamma_n::Real, kappa::Real, gamma_e::Real)
        (0.0 <= n_node < 1.0) || error(
            "n_node must be in [0, 1) for a stationary process, got $n_node; " *
            "n_node >= 1 is the explosive regime")
        (gamma_n > 0 && gamma_e > 0) || error(
            "decay rates must be strictly positive, got gamma_n=$gamma_n, gamma_e=$gamma_e")
        kappa >= 0 || error("kappa must be non-negative, got $kappa")
        new(Float64(n_node), Float64(gamma_n), Float64(kappa), Float64(gamma_e))
    end
end

"""
    TemporalNetworkModel(N, activities, params; attractiveness=ones(N))

`attractiveness` is normalised to mean 1 on construction. That normalisation is what
identifies `kappa`; removing it reintroduces the scale degeneracy this whole rewrite
exists to fix.
"""
struct TemporalNetworkModel
    N::Int
    activities::Vector{Float64}
    params::MechanismParams
    attractiveness::Vector{Float64}

    function TemporalNetworkModel(N::Int, activities::AbstractVector{<:Real},
                                  params::MechanismParams;
                                  attractiveness::Union{Nothing,AbstractVector{<:Real}}=nothing)
        N >= 2 || error("need at least 2 nodes, got N=$N")
        length(activities) == N || error(
            "activities must have length $N, got $(length(activities))")
        all(activities .>= 0) || error("activities must be non-negative")
        any(activities .> 0) || error("at least one activity must be strictly positive")

        b = attractiveness === nothing ? ones(Float64, N) : Vector{Float64}(attractiveness)
        length(b) == N || error("attractiveness must have length $N, got $(length(b))")
        all(b .>= 0) || error("attractiveness must be non-negative")
        mb = sum(b) / N
        mb > 0 || error("attractiveness must have strictly positive mean")
        new(N, Vector{Float64}(activities), params, b ./ mb)
    end
end

"""
    ExcitationState(N, gamma_n, gamma_e)

Recursive O(1) state:

    S[i]     = sum_{t_k in H_i}  gamma_n exp(-gamma_n (t - t_k))   (unit integral)
    m[i][j]  = sum_{t_k in H_ij}          exp(-gamma_e (t - t_k))  (unit peak)
    M[i]     = sum_{l != i} m[i][l]

`S` and `M` decay by a common factor when time advances; the sparse dyad memories are
decayed lazily on access. This is what makes the likelihood O(E) rather than
O(E * events-per-dyad).
"""
mutable struct ExcitationState
    N::Int
    gamma_n::Float64
    gamma_e::Float64
    t::Float64
    S::Vector{Float64}
    M::Vector{Float64}
    nbr::Vector{Dict{Int,Tuple{Float64,Float64}}}   # j => (value, t_last)
end

function ExcitationState(N::Int, gamma_n::Float64, gamma_e::Float64)
    ExcitationState(N, gamma_n, gamma_e, 0.0, zeros(N), zeros(N),
                    [Dict{Int,Tuple{Float64,Float64}}() for _ in 1:N])
end

function advance_to!(st::ExcitationState, t::Float64)
    dt = t - st.t
    dt < 0 && error("events must be time-ordered; tried to advance from $(st.t) to $t")
    dt == 0 && return nothing
    @inbounds begin
        fn = exp(-st.gamma_n * dt)
        fe = exp(-st.gamma_e * dt)
        st.S .*= fn
        st.M .*= fe
    end
    st.t = t
    return nothing
end

function dyad_memory(st::ExcitationState, i::Int, j::Int)
    e = get(st.nbr[i], j, nothing)
    e === nothing && return 0.0
    return e[1] * exp(-st.gamma_e * (st.t - e[2]))
end

function partner_weights(st::ExcitationState, i::Int, b::Vector{Float64}, kappa::Float64)
    w = copy(b)
    if kappa > 0
        for (j, (value, t_last)) in st.nbr[i]
            w[j] += kappa * value * exp(-st.gamma_e * (st.t - t_last))
        end
    end
    w[i] = 0.0
    return w
end

function register!(st::ExcitationState, t::Float64, i::Int, j::Int)
    advance_to!(st, t)
    st.S[i] += st.gamma_n           # unit-integral node kernel
    m_now = dyad_memory(st, i, j) + 1.0   # unit-peak tie kernel
    st.nbr[i][j] = (m_now, t)
    st.nbr[j][i] = (m_now, t)
    st.M[i] += 1.0
    st.M[j] += 1.0
    return nothing
end

node_intensities(model::TemporalNetworkModel, st::ExcitationState) =
    model.activities .+ model.params.n_node .* st.S

"""
    simulate(model, T_max; rng=Random.default_rng(), max_events=20_000_000)

Ogata thinning. The bound is exact: between events every excitation term decays
monotonically, so `sum_i rho_i(t)` evaluated right after the last event dominates until
the next candidate point. Exceeding `max_events` throws rather than truncating -- a
silent truncation would bias every downstream estimate.
"""
function simulate(model::TemporalNetworkModel, T_max::Real;
                  rng=Random.default_rng(), max_events::Int=20_000_000)
    T_max > 0 || error("T_max must be positive, got $T_max")
    p = model.params
    b = model.attractiveness
    st = ExcitationState(model.N, p.gamma_n, p.gamma_e)
    events = Event[]
    t = 0.0

    while t < T_max
        advance_to!(st, t)
        lam = node_intensities(model, st)
        lam_star = sum(lam)
        lam_star <= 0 && break

        t -= log(rand(rng)) / lam_star
        t >= T_max && break

        advance_to!(st, t)
        lam = node_intensities(model, st)
        lam_total = sum(lam)
        lam_total > lam_star * (1 + 1e-9) && error(
            "thinning bound violated (lam_total > lam_star); the excitation state is " *
            "not decaying monotonically, which indicates a bug")
        rand(rng) * lam_star >= lam_total && continue   # thinned

        i = _sample_index(lam, lam_total, rng)
        w = partner_weights(st, i, b, p.kappa)
        w_total = sum(w)
        w_total <= 0 && error("node $i has zero total partner weight at t=$t")
        j = _sample_index(w, w_total, rng)

        register!(st, t, i, j)
        push!(events, (t, i, j))
        length(events) > max_events && error(
            "exceeded max_events=$max_events at t=$t of T_max=$T_max; " *
            "n_node=$(p.n_node) is likely too close to 1 (explosive regime)")
    end
    return events
end

function _sample_index(w::Vector{Float64}, total::Float64, rng)
    u = rand(rng) * total
    c = 0.0
    @inbounds for k in eachindex(w)
        c += w[k]
        c >= u && return k
    end
    # Only reachable through floating-point round-off. Return the last positive-weight
    # index rather than a sentinel: the previous version of this code pushed a (0, 0)
    # event here and silently corrupted the history.
    @inbounds for k in length(w):-1:1
        w[k] > 0 && return k
    end
    error("cannot sample: all weights are zero")
end

"""
    strip_initiators(events)

Drop initiator labels, returning undirected contacts with `i < j`. This is what
empirical proximity data does implicitly; comparing inference before and after isolates
the identifiability cost of the missing label.
"""
function strip_initiators(events::AbstractVector{Event})
    [(t, min(i, j), max(i, j)) for (t, i, j) in events]
end

function summarise(events::AbstractVector{<:Tuple{Float64,Int,Int}}, N::Int, T_max::Real)
    isempty(events) && return Dict("n_events" => 0.0, "rate" => 0.0,
                                   "n_distinct_dyads" => 0.0, "repeat_fraction" => 0.0,
                                   "dyad_saturation" => 0.0)
    dyads = Set{Tuple{Int,Int}}()
    for (_, i, j) in events
        push!(dyads, (min(i, j), max(i, j)))
    end
    E = length(events)
    return Dict("n_events" => Float64(E), "rate" => E / T_max,
                "n_distinct_dyads" => Float64(length(dyads)),
                "repeat_fraction" => 1 - length(dyads) / E,
                "dyad_saturation" => length(dyads) / (0.5 * N * (N - 1)))
end

end # module
