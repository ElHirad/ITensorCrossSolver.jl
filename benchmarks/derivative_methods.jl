# Run from the package root:
# julia --project=. benchmarks/derivative_methods.jl
# Optional positional arguments: repetitions, comma-separated bits per axis.
# Example: julia --project=. benchmarks/derivative_methods.jl 7 4,5,6,7
using ITensorCrossSolver
using ITensorMPS
using LinearAlgebra

include(joinpath(@__DIR__, "..", "examples", "advection_diffusion_reaction_2d.jl"))

function benchmark_median(values)
    ordered = sort(collect(values))
    n = length(ordered)
    n > 0 || throw(ArgumentError("at least one observation required"))
    return isodd(n) ? ordered[(n + 1) ÷ 2] : (ordered[n ÷ 2]+ordered[n ÷ 2 + 1])/2
end

function benchmark_adr_rhs(method, sites, bits, spacings; sample_cache=true)
    if method == :pure
        return adr2d_cross_rhs(bits, spacings; cache=sample_cache)
    elseif method == :hybrid
        dx, dy = spacings
        operators = (
            x=multidim_derivative_mpo(sites, bits, 1, dx; order=1),
            y=multidim_derivative_mpo(sites, bits, 2, dy; order=1),
            xx=multidim_derivative_mpo(sites, bits, 1, dx; order=2),
            yy=multidim_derivative_mpo(sites, bits, 2, dy; order=2),
        )
        return adr2d_cross_rhs(operators; cutoff=1e-22, maxdim=64)
    end
    return throw(ArgumentError("method must be :pure or :hybrid"))
end

# Function barriers keep timing dispatch overhead independent of dictionary types.
function benchmark_solve(rhs::F, initial, final_time, dt, cross) where {F}
    return solve_pde(rhs, initial, (0.0, final_time); dt, saveevery=1, cross_kwargs=cross)
end
function benchmark_timed_solve(rhs::F, initial, final_time, dt, cross) where {F}
    @timed benchmark_solve(rhs, initial, final_time, dt, cross)
end

"""
Compare the two derivative backends on the same periodic nonlinear ADR problem.
Reports warmed wall times, total allocated bytes (not peak memory), GC time,
final bond dimensions, and independently checked DNS errors. Each case warms
both complete solves, interleaves method order across repetitions, and performs
a full GC before each timed solve. JIT warm-up, initial interpolation, operator
construction, DNS, dense reconstruction, error checks, and output are excluded
from solve timings. Warm RHS/operator setup is measured separately.

Each timed region includes all four RK stages, MPO application or sampled
neighbor evaluation, cache construction, cross compression, and snapshot saving.
Both methods use the same deterministic cross settings and exact time-step list.
Timing ratios are observations on this workload, not universal complexity claims.
"""
function benchmark_derivatives(;
    nbits_cases=(4, 5, 6),
    repetitions=5,
    final_time=0.2,
    dt=0.01,
    sample_cache=true,
    write_output=true,
    output_dir=joinpath(@__DIR__, "output"),
)
    repetitions isa Integer && repetitions >= 1 ||
        throw(ArgumentError("repetitions must be positive"))
    !isempty(nbits_cases) && all(n -> n isa Integer && 1 <= n <= 10, nbits_cases) ||
        throw(ArgumentError("benchmark grids require 1–10 bits per axis"))
    isfinite(dt) && dt > 0 && isfinite(final_time) && final_time > 0 ||
        throw(ArgumentError("invalid time interval"))
    BLAS.set_num_threads(1)
    cross = (;
        ranks=4, maxdim=32, tolerance=1e-10, maxsweeps=24, oversampling=3, validation_samples=128
    )
    rows, summaries = NamedTuple[], NamedTuple[]
    println(
        "Julia $VERSION; $(Sys.MACHINE); Julia threads=$(Threads.nthreads()); BLAS threads=$(BLAS.get_num_threads())",
    )
    println(
        "Same ADR, centered differences and RK4: dt=$dt, t_final=$final_time; repetitions=$repetitions; sample_cache=$sample_cache",
    )
    flush(stdout)
    for nbits in nbits_cases
        bits = (nbits, nbits)
        n = 1 << nbits
        dx = 2pi/n
        dt*(1.5/dx+0.3/dx^2+1) <= 2 || throw(ArgumentError("reduce dt for the $n×$n grid"))
        x = collect(0:(n - 1)) .* dx
        sites = quantics_sites(2nbits)
        initial = tt_cross(
            quantics_function(adr2d_initial, (x, x)), fill(2, 2nbits); sites, cross...
        )
        initial.converged || error("initial cross failed")
        factories = Dict(
            method => benchmark_adr_rhs(method, sites, bits, (dx, dx); sample_cache) for
            method in (:pure, :hybrid)
        )
        setup_seconds = Dict{Symbol,Float64}()
        for method in (:pure, :hybrid)
            # The factory above warms operator construction before setup timing.
            setup_samples = Float64[]
            for _ in 1:repetitions
                observation = @timed benchmark_adr_rhs(method, sites, bits, (dx, dx); sample_cache)
                push!(setup_samples, observation.time)
            end
            setup_seconds[method] = benchmark_median(setup_samples)
        end
        println("Warming both methods on $n×$n...")
        flush(stdout)
        warmed = Dict(
            method =>
                benchmark_timed_solve(factories[method], initial.mps, final_time, dt, cross).value
            for method in (:pure, :hybrid)
        )
        times = warmed[:pure].times
        times == warmed[:hybrid].times || error("time grids differ")
        dns = [adr2d_initial(xi, yj) for xi in x, yj in x]
        for h in diff(times)
            dns = adr2d_dns_step(dns, h, dx, dx)
        end
        to_field(sol) = dense_tensor(last(sol.states), (n, n))
        for method in (:pure, :hybrid)
            metrics = adr2d_l2_metrics(to_field(warmed[method]), dns, dx, dx)
            metrics.relative_l2_error < 1e-8 || error("$method fails DNS accuracy target")
        end
        pair_error = norm(to_field(warmed[:pure])-to_field(warmed[:hybrid]))/norm(dns)
        empty!(warmed)
        case_rows = NamedTuple[]
        for repetition in 1:repetitions
            # Alternate first/second to reduce order-dependent bias.
            for method in (isodd(repetition) ? (:pure, :hybrid) : (:hybrid, :pure))
                GC.gc()
                observation = benchmark_timed_solve(
                    factories[method], initial.mps, final_time, dt, cross
                )
                solution = observation.value
                solution.times == times || error("time grids differ")
                metrics = adr2d_l2_metrics(to_field(solution), dns, dx, dx)
                metrics.relative_l2_error < 1e-8 || error("$method fails DNS accuracy target")
                row = (;
                    nbits,
                    nx=n,
                    ny=n,
                    method,
                    repetition,
                    dt,
                    final_time,
                    sample_cache,
                    seconds=observation.time,
                    allocated_bytes=observation.bytes,
                    gc_seconds=observation.gctime,
                    setup_seconds=setup_seconds[method],
                    l2_error=metrics.l2_error,
                    relative_l2_error=metrics.relative_l2_error,
                    maxbond=maximum(bond_dimensions(last(solution.states))),
                )
                push!(rows, row)
                push!(case_rows, row)
                println(
                    "  $method $repetition/$repetitions: $(round(row.seconds;digits=4)) s; relative L2=$(row.relative_l2_error)",
                )
                flush(stdout)
            end
        end
        medians = Dict(
            method => benchmark_median(r.seconds for r in case_rows if r.method==method) for
            method in (:pure, :hybrid)
        )
        for method in (:pure, :hybrid)
            selected = filter(r -> r.method==method, case_rows)
            push!(
                summaries,
                (;
                    nx=n,
                    ny=n,
                    method,
                    repetitions,
                    median_seconds=medians[method],
                    min_seconds=minimum(r.seconds for r in selected),
                    max_seconds=maximum(r.seconds for r in selected),
                    median_allocated_bytes=benchmark_median(r.allocated_bytes for r in selected),
                    median_gc_seconds=benchmark_median(r.gc_seconds for r in selected),
                    setup_seconds=setup_seconds[method],
                    relative_l2_error=maximum(r.relative_l2_error for r in selected),
                    maxbond=maximum(r.maxbond for r in selected),
                    pure_over_hybrid=medians[:pure]/medians[:hybrid],
                    pair_relative_l2_error=pair_error,
                ),
            )
        end
        println(
            "$n×$n: median pure=$(medians[:pure]) s, hybrid=$(medians[:hybrid]) s, pure/hybrid=$(medians[:pure]/medians[:hybrid])",
        )
        flush(stdout)
    end
    if write_output
        mkpath(output_dir)
        for (name, table) in
            (("derivative_timings.csv", rows), ("derivative_summary.csv", summaries))
            open(joinpath(output_dir, name), "w") do io
                println(io, join(keys(first(table)), ','))
                for row in table
                    println(io, join(values(row), ','))
                end
            end
        end
        open(joinpath(output_dir, "environment.txt"), "w") do io
            println(io, "Julia $VERSION; $(Sys.MACHINE)")
            println(io, "CPU: ", first(Sys.cpu_info()).model)
            println(
                io,
                "Julia threads: ",
                Threads.nthreads(),
                "; BLAS threads: ",
                BLAS.get_num_threads(),
            )
            println(io, "Cross settings: ", cross)
            println(io, "ADR velocity=(1,-0.5), diffusivity=(0.05,0.025), reaction_rate=1")
            println(
                io,
                "dt=$dt; final_time=$final_time; sample_cache=$sample_cache; repetitions=$repetitions",
            )
            println(io, "MPO algorithm=naive; cutoff=1e-22; maxdim=64")
            println(
                io,
                "Solve timing excludes setup, JIT warm-up, DNS, dense diagnostics, and file I/O.",
            )
            return println(io, "Allocations are cumulative bytes per solve, not peak memory.")
        end
    end
    return (; rows, summaries)
end

if abspath(PROGRAM_FILE) == @__FILE__
    repetitions = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    bits = length(ARGS) < 2 ? (4, 5, 6) : Tuple(parse.(Int, split(ARGS[2], ',')))
    benchmark_derivatives(; repetitions, nbits_cases=bits)
end
