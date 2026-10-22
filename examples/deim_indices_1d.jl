using ITensorCrossSolver
using LinearAlgebra
using Random
using Printf

# CrossResult does not store DEIM pivots. Load the package's actual cross driver
# in an isolated diagnostic module and wrap only its sampling calls. The package
# methods remain unchanged; compare the result with the public API below.
module DEIMIndexTrace
using LinearAlgebra
using Random
using ITensors: Index, dim
using ITensorMPS: MPS
using ITensorCrossSolver: ITensorCrossSolver
using ITensorCrossSolver: cores_to_mps, reverse_cores, _evaluate, msb_tensor

const events = NamedTuple[]
const context = Ref{Any}(nothing)
const sweep = Ref(0)

function evaluate_fiber(f, dims, mode, left, right)
    mode == 1 && (sweep[] += 1)
    context[] = (; sweep=sweep[], mode, dims=copy(dims), left=copy(left), right=copy(right))
    return ITensorCrossSolver.evaluate_fiber(f, dims, mode, left, right)
end

function deim(U)
    pivots = ITensorCrossSolver.deim(U)
    c = context[]
    expand(p) = (c.left[cld(p, c.dims[c.mode])]..., mod1(p, c.dims[c.mode]))
    selected = expand.(pivots)
    push!(events, (; c..., matrix_size=size(U), pivots=copy(pivots), selected))
    return pivots
end

function gappy_pod(U, extra, pivots)
    samples = ITensorCrossSolver.gappy_pod(U, extra, pivots)
    events[end] = merge(events[end], (; enriched_rows=copy(samples)))
    return samples
end

include(joinpath(dirname(pathof(ITensorCrossSolver)), "cross.jl"))

function trace(f, dims; kwargs...)
    empty!(events)
    sweep[] = 0
    result = tt_cross(f, dims; kwargs...)
    return result, copy(events)
end
end

"""
Encode exp(x)-x^4 on the user's 16-point, endpoint-inclusive grid. Print the
actual DEIM rows and their binary prefix/suffix interpretation for every sweep.
Returns the normal package result, trace, and a dense reconstruction check.
"""
function deim_indices_example(; tolerance=1e-8, seed=0)
    d = 2
    N = 4
    n = d^N
    L = 1
    Δx = L/(n-1)
    x = collect(0:(n - 1)) * Δx
    y = @. exp(x) - x^4
    sites = quantics_sites(N)
    f = quantics_function(z -> exp(z)-z^4, (x,))
    kwargs = (;
        sites,
        ranks=2,
        maxdim=32,
        tolerance,
        maxsweeps=20,
        oversampling=2,
        adaptive=true,
        validation_samples=64,
    )
    result = tt_cross(f, fill(d, N); kwargs..., rng=MersenneTwister(seed))
    traced, trace = DEIMIndexTrace.trace(f, fill(d, N); kwargs..., rng=MersenneTwister(seed))
    reconstructed = dense_vector(result.mps)
    relative_error = norm(reconstructed-y)/norm(y)

    # Verify instrumentation preserves the public implementation's actual run.
    @assert result.converged && traced.converged
    @assert result.errors == traced.errors
    @assert result.evaluations == traced.evaluations
    @assert all(a == b for (a, b) in zip(mps_to_cores(result.mps), mps_to_cores(traced.mps)))
    @assert relative_error <= tolerance
    @assert all(grid_index(quantics_index(i, N)) == i for i in eachindex(y))
    from_vector = tt_cross_tensor(y, fill(d, N); kwargs..., rng=MersenneTwister(seed))
    @assert dense_vector(from_vector.mps) ≈ y rtol=tolerance

    println("Julia ", VERSION, "; tolerance=", tolerance, "; seed=", seed)
    println("Grid: n=", n, ", Δx=", Δx, "; site order: most significant bit first")
    println(
        "Ranks: ",
        bond_dimensions(result.mps),
        "; sweeps: ",
        result.sweeps,
        "; unique evaluations: ",
        result.evaluations,
    )
    println("Per-sweep errors: ", result.errors)
    @printf(
        "Dense relative L2 error: %.16e; maximum absolute error: %.16e\n",
        relative_error,
        maximum(abs, reconstructed-y)
    )
    println("DEIM rows are local, 1-based row indices; binary labels below use 0/1.")
    println("Backward selections are suffixes displayed in original site order.")
    for event in trace
        backward = iseven(event.sweep)
        selections = backward ? reverse.(event.selected) : event.selected
        labels = [join(i-1 for i in t) for t in selections]
        bond = backward ? N-event.mode : event.mode
        kind = backward ? "suffix" : "prefix"
        println(
            "sweep=",
            event.sweep,
            " bond=",
            bond,
            " ",
            kind,
            " basis_size=",
            event.matrix_size,
            " deim_rows=",
            event.pivots,
            " binary=",
            labels,
            " gappy_rows=",
            event.enriched_rows,
        )
    end

    # These are the final core's samples, not all entries visited by TT-cross.
    final_event = last(trace)
    final_indices = [
        grid_index(iseven(final_event.sweep) ? reverse((prefix..., bit)) : (prefix..., bit)) for
        prefix in final_event.selected for bit in 1:d
    ]
    println("Final-core grid indices: ", final_indices)
    println("Final-core x coordinates: ", x[final_indices])
    println("All 16 grid points were evaluated: the default validation budget is 64.")

    # Dense diagnostics use the very same MSB-first encoding as the MPS.
    tensor = msb_tensor(y, fill(d, N))
    println("SVD-DEIM on full MSB-first unfoldings (natural prefix row order):")
    for bond in 1:(N - 1)
        println("bond=", bond, " rows=", deim(svd(msb_unfolding(tensor, bond)).U))
    end

    # A single DEIM point for the whole vector is a different, rank-one problem.
    point = only(deim(reshape(y/norm(y), :, 1)))
    @printf(
        "Rank-one DEIM of the entire vector: index=%d, x=%.16f, y=%.16f\n",
        point,
        x[point],
        y[point]
    )
    return (; result, trace, x, y, reconstructed, relative_error, final_indices, point)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    deim_indices_example()
end
