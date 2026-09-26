# Regenerate every FILLET benchmark and experiment in one Julia session.
#   julia experiments/regen_all.jl   (run from the repository root)
include(joinpath(@__DIR__, "..", "avalon.jl"))
cd(joinpath(@__DIR__, ".."))
t0 = time()
for (cmd, extra) in [("benchmark1", String[]), ("benchmark2", String[]), ("benchmark3", String[]),
                     ("exp1", String[]), ("exp1a", String[]), ("exp2", String[]), ("exp2a", String[]),
                     ("exp3", String[]), ("exp4", String[]),
                     ("exp3", ["base=ben1"]), ("exp4", ["base=ben1"])]
    println("\n=== $cmd $(join(extra, " "))   [t = $(round(time() - t0)) s]")
    flush(stdout)
    run_cli(cmd, extra)
end
println("\nAll done in $(round(time() - t0)) s")
