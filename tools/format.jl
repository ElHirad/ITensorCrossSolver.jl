using JuliaFormatter

root = normpath(joinpath(@__DIR__, ".."))
paths = [joinpath(root, dir) for dir in ("src", "test", "examples", "benchmarks", "tools")]
check_only = "--check" in ARGS
already_formatted = format(paths; overwrite=(!check_only))
if check_only && !already_formatted
    error("Formatting differs. Run julia --project=tools tools/format.jl")
end
println(check_only ? "Formatting check passed." : "Formatted Julia sources.")
