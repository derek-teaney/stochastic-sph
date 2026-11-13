# grid_bare.jl — the bare lattice propagator of the grid scheme (docs/gridstep.tex): the one-step map M(k)
# of the deterministic scheme linearized about a uniform fluid moving with v0 (src/grid.jl,
# grid_linear_map), for all N modes of the lattice, and its modes λ(k) = e^{−iΩΔt − ΓΔt} against
# hydrodynamics at the first-zone k (ideal: Ω = ω_n, Γ = 0; η/s = 0.2: the boosted damping, boost.tex).
# The time steps are those of the stochastic runs (0.5 compute_dt(:viscous) × c_s/v_signal).
#   julia --project examples/grid_bare.jl [out]    (a few steps per case: seconds)
# → docs/figs/grid_bare_freq.png, grid_bare_damp.png, <out>/grid_bare.jls
using PyPlot, StaticArrays, LinearAlgebra, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const OUT = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "grid_bare");  mkpath(OUT)
const FIGD = joinpath(@__DIR__, "..", "docs", "figs")
const S0 = 1e6;  const H = 0.05;  const HA = 2.0;  const LBOX = 1.2;  const CS = sqrt(1/3)
quiet(f) = redirect_stdout(f, devnull)
dtfac(v0) = v0 == 0 ? 1.0 : CS * (1 + v0 * CS) / (v0 + CS)

# continuum ideal hydrodynamics of U = (ε, M_x, M_y)
function state(U, eos)
    ε = U[1];  M = SVector(U[2], U[3]);  er = rest_energy(ε, dot(M, M), eos);  p = pressure(eos, er)
    return p, M / (ε + p)
end
flux(U, j, eos) = ((p, v) = state(U, eos);  SVector(U[1 + j], U[2] * v[j] + (j == 1) * p, U[3] * v[j] + (j == 2) * p))
function Kmat(U, k, eos)
    J = zeros(3, 3)
    for i in 1:3
        dU = zeros(3);  δ = 1e-6 * U[1];  dU[i] = δ;  Up = U + SVector{3}(dU);  Um = U - SVector{3}(dU)
        J[:, i] = k[1] * (flux(Up, 1, eos) - flux(Um, 1, eos)) / 2δ + k[2] * (flux(Up, 2, eos) - flux(Um, 2, eos)) / 2δ
    end
    return J
end
function gamma_hydro(isshear, ω, k, v0, etas, Trel)
    etas == 0 && return 0.0
    ν = etas / Trel;  γ = 1 / sqrt(1 - v0^2);  Ω = γ * (ω - v0 * k[1]);  kx = γ * (k[1] - v0 * ω);  k2 = kx^2 + k[2]^2
    isshear && return ν * k2 / γ
    return 0.5 * (4 / 3) * ν * k2 / (γ * (1 + v0 * (1 / 3) * kx / Ω))
end

function bare(v0, etas)
    eos = ConformalEOS(1/3, 1.0, etas, 0.0)
    _, q, _, _ = quiet(() -> setup_ic(H, HA, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0); L = LBOX, lattice = :hex))
    w0 = q[1].e + q[1].p;  Trel = q[1].T * q[1].γ;  γ = 1 / sqrt(1 - v0^2)
    params, ps, cl, _ = quiet(() -> setup_ic(H, HA, eos, (x, y) -> γ * S0, (x, y) -> SVector(w0 * γ^2 * v0, 0.0); L = LBOX, lattice = :hex))
    st = GridStencil(ps, params, cl);  F = GridFFT(ps, params);  sc = GridScratch(st.N);  g = GridStats()
    dt = 0.5 * compute_dt(ps, params, ConformalEOS(1/3, 1.0, 0.2, 0.0); mode = :viscous) * dtfac(v0)
    step! = etas == 0 ? (p -> grid_step!(p, st, dt, eos, sc, g)) :
            (p -> (viscous_step!(p, params, cl, dt / 2, eos);  grid_step!(p, st, dt, eos, sc, g);  viscous_step!(p, params, cl, dt / 2, eos)))
    M, U0 = grid_linear_map(step!, ps, st, F, eos)
    rows = []
    for i in 1:F.Nx, j in 1:F.Ny
        k = F.kmin[i, j];  norm(k) == 0 && continue
        λ = eigvals(M[:, :, i, j]);  L = log.(λ);  Ω = -imag.(L) ./ dt;  Γ = -real.(L) ./ dt
        ωc = sort(real.(eigvals(Kmat(U0, k, eos))))
        o = sortperm(Ω)
        for (n, idx) in enumerate(o)                 # n = 1 sound −, 2 shear, 3 sound + (sorted)
            push!(rows, (kh = norm(k) * H, θ = atan(k[2], k[1]), mode = n, Ω = Ω[idx], Γ = Γ[idx], ω_hyd = ωc[n],
                         Γ_hyd = gamma_hydro(n == 2, ωc[n], k, v0, etas, Trel), kx = k[1], ky = k[2]))
        end
    end
    return (v0 = v0, etas = etas, dt = dt, rows = rows, Trel = Trel)
end

res = [bare(v0, etas) for etas in (0.0, 0.2) for v0 in (0.0, 0.3, 0.8)]
serialize(joinpath(OUT, "grid_bare.jls"), res)

# check against the wave fits (grid_waves.jl): k along x, kh = 0.52, v0 = 0.8, η/s = 0.2
for row in res[end].rows
    (abs(row.ky) < 1e-9 && abs(row.kx - 2π * 2 / LBOX) < 1 && row.kx > 0 && row.mode in (2, 3)) || continue
    @printf("check (v0 = 0.8, η/s = 0.2, k ∥ x, kh = %.2f, %s): (Ω − ω_n)/(c_s k) = %+.4f, Γ/Γ_hyd = %.3f\n", row.kh,
            row.mode == 2 ? "shear" : "sound +", (row.Ω - row.ω_hyd) / (CS * row.kh / H), row.Γ / row.Γ_hyd)
end

# figures: all modes against |k|h; colour by mode; one column per v0
names = ("sound −", "shear", "sound +");  cols = ("C2", "C0", "C3")
for (fname, ylab, yf, which) in (("grid_bare_freq.png", "(Ω − ω_hydro)/(c_s k)", rw -> (rw.Ω - rw.ω_hyd) / (CS * rw.kh / H), 0.2),
                                 ("grid_bare_damp.png", "Γ / Γ_hydro", rw -> rw.Γ / rw.Γ_hyd, 0.2))
    fig, axs = subplots(1, 3, figsize = (17, 4.8))
    for (j, v0) in enumerate((0.0, 0.3, 0.8))
        local r = res[findfirst(x -> x.v0 == v0 && x.etas == which, res)];  ax = axs[j]
        for n in 1:3
            sel = [rw for rw in r.rows if rw.mode == n]
            ax.scatter([rw.kh for rw in sel], [yf(rw) for rw in sel], s = 4, color = cols[n], label = names[n], alpha = 0.6)
        end
        ax.axhline(fname == "grid_bare_damp.png" ? 1 : 0, color = "k", lw = 0.5)
        fname == "grid_bare_damp.png" && ax.set_ylim(0, 3)
        fname == "grid_bare_freq.png" && (ax.set_xlim(0, 1.6);  ax.set_ylim(-0.25, 0.25))   # the hydrodynamic range
        ax.set_xlabel("|k| h  (first zone)");  ax.set_title(@sprintf("v0 = %.1f, η/s = %.1f: all %d modes of the lattice", v0, which, length(r.rows) ÷ 3), fontsize = 10)
        j == 1 && (ax.set_ylabel(ylab);  ax.legend(fontsize = 8, markerscale = 3))
    end
    tight_layout();  savefig(joinpath(FIGD, fname), dpi = 100);  close();  println("wrote $fname")
end
# the ideal step: the numerical damping of every mode (should be ≈ 0; < 0 is growth)
fig, axs = subplots(1, 3, figsize = (17, 4.8))
for (j, v0) in enumerate((0.0, 0.3, 0.8))
    local r = res[findfirst(x -> x.v0 == v0 && x.etas == 0.0, res)];  ax = axs[j]
    for n in 1:3
        sel = [rw for rw in r.rows if rw.mode == n]
        ax.scatter([rw.kh for rw in sel], [rw.Γ / (CS * rw.kh / H) for rw in sel], s = 4, color = cols[n], label = names[n], alpha = 0.6)
    end
    ax.axhline(0, color = "k", lw = 0.5);  ax.set_xlabel("|k| h  (first zone)")
    ax.set_title(@sprintf("ideal step, v0 = %.1f: Γ/(c_s k) of every mode", v0), fontsize = 10);  j == 1 && (ax.set_ylabel("Γ/(c_s k)");  ax.legend(fontsize = 8, markerscale = 3))
end
tight_layout();  savefig(joinpath(FIGD, "grid_bare_ideal.png"), dpi = 100);  close();  println("wrote grid_bare_ideal.png")
