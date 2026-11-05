#!/usr/bin/env julia
"""
Parity check: the Julia port must reproduce the Python reference to 1e-8.

Run this FIRST, before trusting any Julia result. The Julia code in this experiment was
written without a Julia runtime available, so it is unverified until this script passes.

The fixture pins an event sequence and the likelihood values on it, not the simulation:
the two languages have different RNGs, so simulated realisations will never match, but
the likelihood evaluated on a fixed sequence must match exactly.

Regenerate the fixture from Python with:

    python -c "import sys; sys.path.insert(0,'.'); \
               from tests.test_model import write_parity_fixture; write_parity_fixture()"

Then:

    julia --project=. scripts/check_parity.jl
"""

using JSON
using Printf
using Random

include(joinpath(@__DIR__, "GenerativeModels.jl"))
include(joinpath(@__DIR__, "Inference.jl"))
using .GenerativeModels
using .Inference

const TOL = 1e-8
const FIXTURE = joinpath(@__DIR__, "..", "tests", "parity_fixture.json")

function main()
    isfile(FIXTURE) || error(
        "fixture not found at $FIXTURE. Generate it from Python first; " *
        "see the docstring at the top of this file.")

    fx = JSON.parsefile(FIXTURE)
    N = Int(fx["N"])
    T = Float64(fx["T"])

    # Python is 0-based, Julia is 1-based. Shift node indices.
    events = [(Float64(e[1]), Int(e[2]) + 1, Int(e[3]) + 1) for e in fx["events"]]
    contacts, initiator, side = Inference.split_events(events)

    @printf("fixture: N=%d  T=%.1f  E=%d\n\n", N, T, length(events))

    n_fail = 0

    println("TIMING block")
    for g in fx["timing_grid"]
        n_node = Float64(g["n_node"]); gamma_n = Float64(g["gamma_n"])
        want = Float64(g["timing_nll"])
        got = timing_nll(n_node, gamma_n, contacts, initiator, N, T)
        rel = abs(got - want) / max(1.0, abs(want))
        ok = rel < TOL
        ok || (n_fail += 1)
        @printf("  n=%.2f gn=%.2f   python %.9f   julia %.9f   rel %.2e  %s\n",
                n_node, gamma_n, want, got, rel, ok ? "OK" : "MISMATCH")
    end

    println("\nSELECTION block")
    for g in fx["selection_grid"]
        kappa = Float64(g["kappa"]); gamma_e = Float64(g["gamma_e"])
        want = Float64(g["selection_nll"])
        got = selection_nll(kappa, gamma_e, contacts, side, N)
        rel = abs(got - want) / max(1.0, abs(want))
        ok = rel < TOL
        ok || (n_fail += 1)
        @printf("  k=%.2f ge=%.3f   python %.9f   julia %.9f   rel %.2e  %s\n",
                kappa, gamma_e, want, got, rel, ok ? "OK" : "MISMATCH")
    end

    println("\nInternal consistency (Julia only)")
    # The simulator's event rate must match branching theory: rate = sum(a) / (1 - n).
    # This catches a wrong node-kernel normalisation, which the fixture alone would not.
    for n_node in [0.0, 0.3, 0.6]
        Nl, Tl = 30, 4000.0
        act = fill(0.02, Nl)
        model = TemporalNetworkModel(Nl, act, MechanismParams(n_node, 0.5, 0.0, 0.1))
        ev = simulate(model, Tl; rng=MersenneTwister(11))
        observed = length(ev) / Tl
        expected = sum(act) / (1 - n_node)
        rel = abs(observed - expected) / expected
        ok = rel < 0.15
        ok || (n_fail += 1)
        @printf("  n_node=%.1f  rate observed %.4f  theory %.4f  rel %.1f%%  %s\n",
                n_node, observed, expected, 100rel, ok ? "OK" : "MISMATCH")
    end

    println()
    if n_fail == 0
        println("ALL CHECKS PASSED - the Julia port matches the Python reference.")
    else
        println("$n_fail CHECK(S) FAILED - do not use the Julia results until fixed.")
        exit(1)
    end
end

main()
