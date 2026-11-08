# gaussian_compare_run.jl — one run of the testgaussian.jl setup for a chosen
# dissipation scheme, resolution and η/s; writes slices and particle fields.
#
#   julia --project examples/gaussian_compare_run.jl <scheme> <h> <eta_by_s>
#   e.g. julia --project examples/gaussian_compare_run.jl bond 0.02 0.05
# Output: examples/gaussian_cmp/<scheme>_h<h>_eta<eta>.txt  (slices) and _particles.txt
using StaticArrays, LinearAlgebra, Printf, DelimitedFiles
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
# reuse initialize_gaussian and run_gaussian! from testgaussian.jl (everything above "# ── Main")
let src = read(joinpath(@__DIR__, "testgaussian.jl"), String)
    body = split(src, "# ── Main")[1]
    body = replace(body, r"^using PyPlot\n"m => "", r"^include\(.*\)\n"m => "")
    body = replace(body, r"# ── Plotting.*?# ── Time evolution"s => "# ── Time evolution")
    include_string(Main, body)
end

scheme = Symbol(ARGS[1]);  h = parse(Float64, ARGS[2]);  ηs = parse(Float64, ARGS[3])
L = 1.0;  t_end = 2.2
params, particles, cl, eos, cs = initialize_gaussian(h, 2.0; L, A=1.0, delta=1e-1, sigmax=0.1, sigmay=0.14,
                                                     x0=L/2, y0=L/2, n_images=1, η_by_s=ηs)
params.dissipation = scheme
t0 = time()
n_steps, s_i, s_f, H_i, H_f = run_gaussian!(particles, params, cl, eos, t_end)
@printf("RESULT scheme=%s h=%.3g eta=%.3g steps=%d  ΔS/S=%+.3e  ΔH/H=%+.3e  wall=%.0fs\n",
        scheme, h, ηs, n_steps, (s_f - s_i) / s_i, (H_f - H_i) / H_i, time() - t0)
rebuild_cell_list!(cl, particles)
xs, ex = plot_slice(particles, params, cl; field=p -> p.e, dim=:x)
ys, ey = plot_slice(particles, params, cl; field=p -> p.e, dim=:y)
tag = joinpath(@__DIR__, "gaussian_cmp", @sprintf("%s_h%g_eta%g", scheme, h, ηs))
writedlm(tag * ".txt", hcat(xs, ex, ys, ey))
writedlm(tag * "_particles.txt", hcat([p.x[1] for p in particles], [p.x[2] for p in particles], [p.e for p in particles]))
println("wrote ", tag)
