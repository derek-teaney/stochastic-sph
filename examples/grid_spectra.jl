# grid_spectra.jl — equal-time spectra of all lattice modes of grid runs (SR_GRID=1, docs/gridstep.tex):
# C_AA(k,0) V / χ_AA for A = ε, M_x, M_y at every k of the Brillouin zone, pooled over the runs of a set,
# against the susceptibilities of the moving fluid (boost.tex).  Equipartition is 1 at every k.
#   julia --project examples/grid_spectra.jl [dir ...]     (default $SCRATCH/sph/grid_spectra/sp_v0, sp_v0.8)
# → docs/figs/grid_spectra_maps.png (the zone), grid_spectra_kh.png (against |k|h, coloured by κ(k)h, the
#   lattice derivative: the modes with small κ are the ones the deterministic step barely damps), and a table
#   of shell averages with jackknife errors over the runs.
using PyPlot, HDF5, Printf, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const ROOT = joinpath(ENV["SCRATCH"], "sph", "grid_spectra")
const DIRS = length(ARGS) > 0 ? ARGS : [joinpath(ROOT, "sp_v0"), joinpath(ROOT, "sp_v0.8")]
const FIGD = joinpath(@__DIR__, "..", "docs", "figs");  const CS2 = 1 / 3;  const H = 0.05

function Uof(θ)                     # θ = (−β⁰, β^x, β^y) → (ε, M_x, M_y), e = T⁴ (T_scale = 1)
    β0 = -θ[1];  b = SVector(θ[2], θ[3]);  T = 1 / sqrt(β0^2 - dot(b, b));  v = b / β0
    γ2 = 1 / (1 - dot(v, v));  e = T^4;  p = CS2 * e;  w = e + p
    SVector(w * γ2 - p, w * γ2 * v[1], w * γ2 * v[2])
end
function chi(Trel, v)               # susceptibilities per unit volume
    γ = 1 / sqrt(1 - v^2);  θ = SVector(-γ / Trel, γ * v / Trel, 0.0);  J = zeros(3, 3)
    for j in 1:3
        d = 1e-6 * maximum(abs, θ);  e = SVector{3}(i == j ? d : 0.0 for i in 1:3)
        J[:, j] = (Uof(θ + e) - Uof(θ - e)) / 2d
    end
    (J + J') / 2
end

# κ(k) h of the stencil at every mode (the lattice derivative), for the lattice of the runs
function kappa_map(L, kmin)
    eos = ConformalEOS(1/3, 1.0, 0.0, 0.0)
    params, ps, cl, _ = redirect_stdout(devnull) do
        setup_ic(H, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex)
    end
    st = GridStencil(ps, params, cl);  r = st.off[1]:st.off[2]-1
    [st.V * norm(sum(st.D[q] * cis(-dot(k, st.r[q])) for q in r)) * H for k in kmin]
end

sets = []
for d in DIRS
    fs = filter(f -> endswith(f, ".h5"), readdir(d; join = true));  isempty(fs) && continue
    runs = []
    for f in fs
        h5open(f) do h
            haskey(h, "grid_Ck_re") || return
            at = attributes(h);  n = read(h["grid_Cn"])[1];  n == 0 && return
            push!(runs, (C = (read(h["grid_Ck_re"]) .+ im .* read(h["grid_Ck_im"])) ./ n, n = n, V = read(h["grid_V"])[1],
                         kmin = read(h["grid_kmin"]), Trel = read(at["Trel"]), v = read(at["boost"]), L = read(at["L"])))
        end
    end
    isempty(runs) && continue
    r1 = runs[1];  χ = chi(r1.Trel, r1.v);  Nx, Ny = size(r1.C, 3), size(r1.C, 4)
    kmin = [SVector(r1.kmin[1, i, j], r1.kmin[2, i, j]) for i in 1:Nx, j in 1:Ny]
    # ratio per run: C_AA V / χ_AA
    ratio(r, a) = real.(r.C[a, a, :, :]) .* r.V ./ χ[a, a]
    R = [sum(ratio(r, a) for r in runs) / length(runs) for a in 1:3]
    push!(sets, (name = basename(d), v = r1.v, n = length(runs), kmin = kmin, R = R, runs = runs, χ = χ,
                 κh = kappa_map(r1.L[1], kmin), ratio = ratio))
end
isempty(sets) && (println("no grid spectra found in $DIRS");  exit())
names = ("ε", "M_x", "M_y")

# table: shell averages of C_AA V/χ_AA over |k|h bins, jackknife over the runs
println("C_AA(k,0) V / χ_AA, averaged over shells of |k|h (all directions); jackknife errors over the runs")
bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9)]
for s in sets
    @printf("%s (v = %.1f, %d runs, %d records each):\n", s.name, s.v, s.n, s.runs[1].n)
    kh = [norm(k) * H for k in s.kmin]
    for (lo, hi) in bins
        sel = findall(x -> lo < x <= hi, kh);  isempty(sel) && continue
        line = @sprintf("  %3.1f < kh ≤ %3.1f (%4d modes):", lo, hi, length(sel))
        for a in 1:3
            per = [sum(s.ratio(r, a)[sel]) / length(sel) for r in s.runs];  n = length(per);  m = sum(per) / n
            jk = [(sum(per) - per[i]) / (n - 1) for i in 1:n];  err = sqrt((n - 1) / n * sum(abs2, jk .- m))
            line *= @sprintf("   %s %.3f±%.3f", names[a], m, err)
        end
        println(line)
    end
    sel = findall(<(0.1), s.κh);  sel = filter(i -> norm(s.kmin[i]) > 0, sel)
    @printf("  modes with κh < 0.1 (%d, the near-null modes of the stencil): ε %.3f  M_x %.3f  M_y %.3f\n", length(sel),
            [sum(s.R[a][sel]) / length(sel) for a in 1:3]...)
end

# the zone: C_AA V/χ_AA at every k (first-zone representatives)
fig, axs = subplots(length(sets), 3, figsize = (16, 4.6 * length(sets)), squeeze = false)
for (i, s) in enumerate(sets), a in 1:3
    ax = axs[i, a];  ok = [norm(k) > 0 for k in s.kmin]
    sc = ax.scatter([k[1] * H for k in s.kmin[ok]], [k[2] * H for k in s.kmin[ok]], c = s.R[a][ok], s = 9, cmap = "RdBu_r", vmin = 0.7, vmax = 1.3)
    ax.set_aspect("equal");  colorbar(sc, ax = ax, fraction = 0.046)
    ax.set_title(@sprintf("%s: C_%s%s V/χ, %d runs", s.name, names[a], names[a], s.n), fontsize = 10)
    ax.set_xlabel("k_x h");  ax.set_ylabel("k_y h")
end
tight_layout();  savefig(joinpath(FIGD, "grid_spectra_maps.png"), dpi = 90);  close();  println("wrote grid_spectra_maps.png")

# against |k|h, coloured by κ(k)h
fig, axs = subplots(length(sets), 3, figsize = (16, 4.4 * length(sets)), squeeze = false)
for (i, s) in enumerate(sets), a in 1:3
    ax = axs[i, a];  ok = [norm(k) > 0 for k in s.kmin]
    sc = ax.scatter([norm(k) * H for k in s.kmin[ok]], s.R[a][ok], c = s.κh[ok], s = 6, cmap = "viridis")
    ax.axhline(1, color = "k", lw = 0.6);  ax.set_ylim(0.5, 1.5);  colorbar(sc, ax = ax, fraction = 0.046, label = "κ(k) h")
    ax.set_title(@sprintf("%s: C_%s%s(k,0) V/χ against |k|h", s.name, names[a], names[a]), fontsize = 10);  ax.set_xlabel("|k| h")
end
tight_layout();  savefig(joinpath(FIGD, "grid_spectra_kh.png"), dpi = 90);  close();  println("wrote grid_spectra_kh.png")
