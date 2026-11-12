# multistep_rates.jl — the rates of docs/multistep.tex on a perfect hexagonal lattice,
# from closed-form lattice sums, versus h/a at fixed h (conformal EOS, fluid at rest).
#
#   Ideal step (second variation of U = Σ_a s ε(S_a), ε = e/S, S_a = s Σ_b W(x_a − x_b)),
#   for a displacement u_b = U ê cos(k·x_b):
#
#     ω²(k, ê) = −(1/6) (g·ê)²/Σ₀² + (1/(2Σ₀)) Σ_b (1 − cos k·r_b) êᵀ ∇∇W(r_b) ê,
#     g(k) = Σ_b sin(k·r_b) ∇W(r_b),  Σ₀ = Σ_b W(r_b) (self term included),
#
#   using M = sT, ε'/T = 1/(4S), ε''/T = −1/(6S²) (conformal).  At small k this gives
#   ω_L → c_s k and ω_T² → (1/(4Σ₀)) Σ_b x_b² ∂_y²W k², which vanishes in the continuum.
#
#   Viscous step (bond form, docs/stochrep.tex): Γ(k) = 2K(k)/(VwT), split into the star
#   (resolved, K_res) and residual (K − K_res) parts; in units of ν = η/w these are geometric.
#
# Run:  julia --project examples/multistep_rates.jl  [--check]
using StaticArrays, LinearAlgebra, Printf

const C2 = 7 / (4π)                                    # Wendland C2 2D normalization, h = 1
W(r)  = (q = r; q >= 2 ? 0.0 : C2 * (1 - q / 2)^4 * (1 + 2q))
Wp_r(r)  = r >= 2 ? 0.0 : -5C2 * (1 - r / 2)^3          # W'(r)/r
Wpp(r)   = r >= 2 ? 0.0 : -5C2 * (1 - r / 2)^2 * (1 - 2r)
gradW(r::SVector{2}) = (ρ = norm(r); Wp_r(ρ) * r)
function hessW(r::SVector{2})
    ρ = norm(r);  n = r / ρ
    return Wpp(ρ) * (n * n') + Wp_r(ρ) * (I - n * n')
end
ψ(ρ, hψ) = (q = ρ / hψ; q < 2 ? (1 - q / 2)^3 / hψ^2 : 0.0)

"lattice vectors of the hexagonal lattice (rows along x) with 0 < |r| < 2 (h = 1)"
function hexstar(a)
    rs = SVector{2,Float64}[];  ny = ceil(Int, 2 / (a * sqrt(3) / 2)) + 1;  nx = ceil(Int, 2 / a) + 2
    for j in -ny:ny, i in -nx:nx
        r = SVector(a * (i + 0.5 * isodd(j)), a * j * sqrt(3) / 2)
        0 < norm(r) < 2 && push!(rs, r)
    end
    return rs
end

struct Lat
    a::Float64;  rs::Vector{SVector{2,Float64}};  Σ0::Float64
    w::Vector{Float64}                                 # bond weights, Σ w r rᵀ = 𝟙
end
function Lat(ha; hψ = 1.0)
    a = 1 / ha;  rs = hexstar(a)
    Σ0 = W(0.0) + sum(W(norm(r)) for r in rs)
    p = [ψ(norm(r), hψ) for r in rs];  n = sum(p[j] * dot(rs[j], rs[j]) for j in eachindex(rs)) / 2
    return Lat(a, rs, Σ0, p ./ n)
end

function ω2(L::Lat, k, ê)
    g = sum(sin(dot(k, r)) * gradW(r) for r in L.rs)
    t = sum((1 - cos(dot(k, r))) * dot(ê, hessW(r) * ê) for r in L.rs)
    return -dot(g, ê)^2 / (6L.Σ0^2) + t / (2L.Σ0)
end

"bond-form dissipation per unit β amplitude (V = T = 1, κs = η = 1, κb = η/3): returns Γ/ν, Γ_res/ν"
function Γrates(L::Lat, k, ê; κs = 1.0, κb = 1 / 3)
    Δ = [ê * (cis(dot(k, r)) - 1) for r in L.rs]
    G = sum(L.w[j] * (Δ[j] * transpose(L.rs[j])) for j in eachindex(L.rs))
    θ = tr(G);  Ω = (G - transpose(G)) / 2;  Sg = (G + transpose(G)) / 2;  S0 = Sg - θ / 2 * I
    Kb = sum(L.w[j] * sum(abs2, Δ[j] - (Ω + θ / 2 * I) * L.rs[j]) for j in eachindex(L.rs))
    Ktot = κs * Kb + 0.5 * κb * abs2(θ)
    Kres = 0.5 * (2κs * sum(abs2, S0) + κb * abs2(θ))
    return 2Ktot, 2Kres
end

if "--check" in ARGS
    # Check against the code: lattice sums vs smooth_to_particles!, ω² vs hamiltonian_force!,
    # Γ vs Kmode on a code star.  h/a = 2, box 0.6, h = 0.05.
    include(joinpath(@__DIR__, "..", "src", "sph.jl"));  include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
    include(joinpath(@__DIR__, "..", "src", "setup.jl"));  include(joinpath(@__DIR__, "..", "src", "rng.jl"))
    include(joinpath(@__DIR__, "..", "src", "stochrep.jl"));  include(joinpath(@__DIR__, "stochrep_thermalize_common.jl"))
    h = 0.05;  eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
    for ha in (2.0, 3.0)
        params, ps, cl, _ = setup_ic(h, ha, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 0.6, lattice = :hex)
        Lt = Lat(ha);  a = params.a
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
        @printf("h/a=%.0f  Σ0 h² : lattice %.8f  code %.8f   (a_y/a = %.6f vs √3/2 = %.6f)\n", ha, Lt.Σ0,
                ps[1].S_smooth / ps[1].s * h^2, (params.L[2] / round(Int, params.N / round(Int, params.L[1] / a))) / a, sqrt(3) / 2)
        T = ps[1].T;  s = ps[1].s;  x0 = [p.x for p in ps];  dotm = zeros(SVector{2,Float64}, length(ps))
        for nk in (1, 4, 8, round(Int, params.L[1] / a / 2)), (pol, ê) in ((:L, SVector(1.0, 0)), (:T, SVector(0, 1.0)))
            k = SVector(2π * nk / params.L[1], 0.0);  ε = 1e-5 * a
            for (i, p) in enumerate(ps); p.x = x0[i] + ε * ê * cos(dot(k, x0[i])); end
            rebuild_cell_list!(cl, ps);  smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
            hamiltonian_force!(dotm, ps, params, cl)
            Fk = sum(dot(dotm[i], ê) * cos(dot(k, x0[i])) for i in eachindex(ps)) / (length(ps) / 2)
            ωcode = -Fk / (ε * s * T) * h^2                      # ω² h², M = sT
            @printf("   kh=%.3f %s  ω²h²: analytic %+.6e  code %+.6e\n", norm(k) * h, pol, ω2(Lt, k * h, ê), ωcode)
        end
        for (i, p) in enumerate(ps); p.x = x0[i]; end
        rebuild_cell_list!(cl, ps)
        st = stochrep_frozen(ps, params, cl, eos).stars[1];  w = ps[1].e + ps[1].p;  ν = st.κs / w
        for kh in (0.3, 1.5, 4.0), ê in (SVector(1.0, 0), SVector(0, 1.0))
            K = Kmode(st, SVector(kh / h, 0.0), ê, false)
            Γc = 2 .* K ./ (st.V * w * st.T) ./ ν .* h^2;  Γa = Γrates(Lt, SVector(kh, 0.0), ê)
            @printf("   kh=%.1f ê=%s  Γh²/ν code (%.5f, %.5f)  analytic (%.5f, %.5f)\n", kh, ê == SVector(1.0, 0) ? "L" : "T", Γc..., Γa...)
        end
    end
    exit()
end

"dynamical matrix ω²_ij(k) (h = 1): eigenvalues are ω² of the two polarizations"
function D2(L::Lat, k)
    g = sum(sin(dot(k, r)) * gradW(r) for r in L.rs)
    t = sum((1 - cos(dot(k, r))) * hessW(r) for r in L.rs)
    return Symmetric(-(g * g') / (6L.Σ0^2) + t / (2L.Σ0))
end

# ── Equipartition (λ = 1): ⟨|u_k|²⟩ = 1/(s_a ω²) per polarization, on the k-grid of a run box.
# Returns rms nearest-neighbour distance change and rms displacement (COM removed) times √s_a,
# in units of a; divide by √s_a.
function disorder(ha; Lbox = 12.0)                     # Lbox in units of h (0.6 / 0.05)
    Lt = Lat(ha);  a = Lt.a
    Nx = round(Int, Lbox / a);  Ny = round(Int, Lbox / (a * sqrt(3) / 2));  Ny += isodd(Ny)
    Lx = Nx * a;  Ly = Ny * a * sqrt(3) / 2
    nn = [SVector(a, 0.0), SVector(a / 2, a * sqrt(3) / 2), SVector(-a / 2, a * sqrt(3) / 2)]
    dd = 0.0;  uu = 0.0;  nk = 0;  neg = 0
    for i in 0:Nx-1, j in 0:Ny-1                       # reciprocal lattice of the box (all N modes;
        (i == 0 && j == 0) && continue                 #  wavevectors folded into the zone are equivalent)
        k = SVector(2π * i / Lx, 2π * j / Ly - 2π * i / Lx * 0)   # rows offset: use the box periodicity
        F = eigen(D2(Lt, k))
        for p in 1:2
            ω2v = F.values[p];  ê = F.vectors[:, p]
            ω2v <= 0 && (neg += 1; continue)
            uu += 1 / ω2v
            dd += sum((dot(r / a, ê))^2 * 2 * (1 - cos(dot(k, r))) for r in nn) / 3 / ω2v
        end
        nk += 1
    end
    N = Nx * Ny
    return (dnn = sqrt(dd / N) / a, du = sqrt(uu / N) / a, N = N, neg = neg)   # × 1/√s_a; ω in units c=1/h → a in h
end

if isdefined(Main, :LINDEMANN)
    for ha in (2.0, 3.0)
        D = disorder(ha);  a = 0.05 / ha;  V0 = sqrt(3) / 2 * a^2
        @printf("h/a=%.0f N=%d (negative modes %d): nn change = %.3f a/√s_a, displacement = %.3f a/√s_a (h = 1 units)\n",
                ha, D.N, D.neg, D.dnn, D.du)
        for S0 in (1e6, 3e5, 2.25e5, 1e5)
            sa = S0 * V0
            @printf("   S0=%.2e  s_a=%6.1f  S0·πh²=%6.0f  δv=%.3f  δv_h=%.4f :  nn change %.2f a,  displacement %.2f a\n",
                    S0, sa, S0 * π * 0.05^2, 1 / sqrt(sa), 1 / sqrt(S0 * 3.627 * 0.05^2), D.dnn / sqrt(sa), D.du / sqrt(sa))
        end
    end
    exit()
end

# ── The scan: k along x (to the K point 4π/3a) and along y (to the M point 2π/(√3 a)) ──
println("Units: h = 1.  ω in c_s/h-free units (ω h; c_s h⁻¹ = 0.577), Γ in ν/h².")
println(" h/a  bonds | c_t²/c_s²   min ω²h² (where)    max ω_T h  max ω_L h | max Γ_res/ν  max Γ_resid/ν  kh(R=1/2)")
rows = []
for ha in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
    Lt = Lat(ha);  a = Lt.a
    # small-k transverse coefficient (exact lattice sum): ω_T² = c_t² k²
    ct2 = sum(r[1]^2 * hessW(r)[2, 2] for r in Lt.rs) / (4Lt.Σ0)
    worst = (Inf, 0.0, :x);  ωTmax = 0.0;  ωLmax = 0.0;  Γrmax = 0.0;  Γdmax = 0.0;  khR = NaN
    for (dir, kmax) in ((SVector(1.0, 0), 4π / (3a)), (SVector(0, 1.0), 2π / (sqrt(3) * a)))
        e⊥ = SVector(-dir[2], dir[1])
        for kk in range(1e-3, kmax, length = 400)
            k = kk * dir
            wL = ω2(Lt, k, dir);  wT = ω2(Lt, k, e⊥)
            min(wL, wT) < worst[1] && (worst = (min(wL, wT), kk, wL < wT ? :L : :T))
            ωLmax = max(ωLmax, sqrt(max(wL, 0)));  ωTmax = max(ωTmax, sqrt(max(wT, 0)))
            for ê in (dir, e⊥)
                Γ, Γr = Γrates(Lt, k, ê)
                Γrmax = max(Γrmax, Γr);  Γdmax = max(Γdmax, Γ - Γr)
                dir == SVector(1.0, 0) && ê == e⊥ && isnan(khR) && Γr < 0.5Γ && (khR = kk)
            end
        end
    end
    @printf("%4.0f %6d | %+.3e  %+.3e (kh=%.2f %s)  %8.3f  %8.3f | %10.2f  %12.2f   %6.2f\n",
            ha, length(Lt.rs), ct2 / (1/3), worst[1], worst[2], worst[3], ωTmax, ωLmax, Γrmax, Γdmax, khR)
    push!(rows, (ha = ha, ct2 = ct2, worst = worst, ωTmax = ωTmax, ωLmax = ωLmax, Γrmax = Γrmax, Γdmax = Γdmax))
end
