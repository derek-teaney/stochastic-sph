# number_tests_montage.jl — side-by-side figures of the deterministic suite (nersc/dtests.sh)
# for docs/number_tests.tex: one row per test, columns smoothed | energy, entropy volume |
# energy, number volume.  Login node:  julia --project examples/number_tests_montage.jl [dir]
using PyPlot
const DIR = length(ARGS) > 0 ? ARGS[1] : joinpath(ENV["SCRATCH"], "sph", "number_tests")
const OUT = joinpath(@__DIR__, "..", "docs", "figs", "number_tests")
const CFG = [("s", "smoothed"), ("k", "energy, entropy volume"), ("n", "energy, number volume")]
for name in ("testgaussian_final", "testgaussian_1d_final", "testsound_final", "testsound_viscous_final",
             "testsound_viscous_decay", "testsound_viscous_1d_final", "testdiffuse_final", "testshearwave_compare")
    fig, axs = subplots(1, 3, figsize = (18, 5.2))
    for (i, (c, lab)) in enumerate(CFG)
        f = joinpath(DIR, c, name * ".png")
        isfile(f) && axs[i].imshow(imread(f));  axs[i].axis("off");  axs[i].set_title(lab, fontsize = 13)
    end
    tight_layout();  savefig(joinpath(OUT, name * ".png"), dpi = 110);  close(fig)
end
for f in ("shearwave_decay.png", "shearwave_decay_eta0.002.png", "shearwave_scan.png")
    cp(joinpath(DIR, "sw", f), joinpath(OUT, f); force = true)
end
println("written to $OUT")
