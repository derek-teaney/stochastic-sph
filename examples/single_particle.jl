# single_particle.jl — the exact single-particle distribution at fixed volume (docs/statics.tex, "One particle at fixed volume"
# and Step 0 of the plan): for each HMC run given, at its bath temperature T0 and mean volume V = L²/N, the quadrature of
#   P(e, m | V) ∝ exp[vote_entropy(e, m, V) − e/T0]       (flat in (e, m); vote_entropy = S − ln T for the Liouville measure)
# in (e, |m|) with the polar Jacobian 2π|m|.  Writes per run: ln Z at V(1 + δ), δ ∈ {−2,…,2}·dV (for f₁, f₂), and at V the moments
# ⟨s⟩, Var s, ⟨m_x²⟩, ⟨|m|⁴⟩, ⟨τ_kin²⟩ (one shear component, τ = ½|v||m| sin 2φ: ⟨τ²⟩ = ⟨|v|²|m|²⟩/8), ⟨e⟩, and ⟨s⟩ at V(1 ± dV).
#   julia --project examples/single_particle.jl data/volfluct/vf_ha2_s*.h5   →  data/volfluct/single_particle.csv
using StaticArrays, LinearAlgebra, Printf, HDF5
include(joinpath(@__DIR__, "..", "src", "load.jl"))

const NE = 2400;  const NX = 2400;  const DV = 0.01
eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)

# quadrature at volume V, bath temperature T; returns ln Z and the moments
function quad(V, T, liouville)
    s0 = (4 / 3) * T^3;  ebar = T^4 * V;  sde = T * sqrt(4T^3 * V)       # saddle: e = T⁴V, Var e = T² C_V V, C_V = 4T³
    es = range(max(1e-9ebar, ebar - 14sde), ebar + 40sde, length = NE)
    xs = range(0, 1, length = NX + 1)[1:end-1] .+ 0.5 / NX                # |m| = e x, midpoints
    lw = fill(-Inf, NE, NX);  S = zeros(NE, NX);  M2 = zeros(NE, NX);  VM = zeros(NE, NX)
    Threads.@threads for i in 1:NE
        e = es[i]
        for j in 1:NX
            m = e * xs[j]
            Sd, v, Tb = eos_from_energy(e / V, SVector(m / V, 0.0), eos)
            (isfinite(Sd) && dot(v, v) <= V2MAX && Tb > 0) || continue
            s = V * Sd
            lw[i, j] = (liouville ? s - log(Tb) : s) - e / T + log(2π * m * e)     # Jacobian 2π m dm, dm = e dx
            S[i, j] = s;  M2[i, j] = m^2;  VM[i, j] = norm(v) * m
        end
    end
    mx = maximum(lw);  w = exp.(lw .- mx);  Z = sum(w)
    lnZ = mx + log(Z * step(es) * step(xs))
    av(f) = sum(w .* f) / Z
    ms = av(S)
    return (lnZ = lnZ, s = ms, vars = av((S .- ms) .^ 2), mx2 = av(M2) / 2, m4 = av(M2 .^ 2), tau2 = av(VM .^ 2) / 8,
            e = av(repeat(collect(es), 1, NX)), s0V = s0 * V)
end

# grid mode: julia examples/single_particle.jl grid  → the same quantities against s(T)V = 4σ on a log grid (T = 1; everything
# depends on s(T)V only), Liouville measure, data/volfluct/single_particle_grid.csv, for the theory curves of the figures
const GRID = !isempty(ARGS) && ARGS[1] == "grid"
const MEASURES = GRID ? (true,) : (true, false)
out = open(joinpath(@__DIR__, "..", "data", "volfluct", GRID ? "single_particle_grid.csv" : "single_particle.csv"), "w")
println(out, "file,measure,T0,V,sa_label,sa_T0,sigma_T0,f1,f2,s_mean,var_s,CV_V,mx2,Tw_V,m4,tau2,T2,e_mean,eV,dsdV_V,s_minus")
cases = GRID ? [(1.0, sa / (4 / 3), NaN, "grid") for sa in exp.(range(log(4.0), log(8000.0), length = 24))] :
        [h5open(f) do h
             a = attrs(h);  (a["T0"], prod(a["L"]) / a["N"], a["s_a"], f)
         end for f in ARGS]
for (T0, V, sal, f) in cases
    for liou in MEASURES
        q = [quad(V * (1 + k * DV), T0, liou) for k in -2:2]
        l = [x.lnZ for x in q];  c = q[3]
        f1 = (l[4] - l[2]) / (2DV)                                # V d lnZ/dV  (d ln V = DV)
        f2 = (l[4] - 2l[3] + l[2]) / DV^2                         # V² d²lnZ/dV²  (steps of V·DV in V)
        dsdV = (q[4].s - q[2].s) / (2DV)                          # V d⟨s⟩/dV
        sT = (4 / 3) * T0^3;  w = T0 * sT;  CV = 4T0^3
        row = (basename(f), liou ? "liouville" : "einstein", T0, V, sal, sT * V, sT * V / 4, f1, f2, c.s, c.vars, CV * V,
               c.mx2, T0 * w * V, c.m4, c.tau2, T0^2, c.e, T0^4 * V, dsdV, c.s - dsdV)
        println(out, join(row, ","))
        @printf("%-16s %-9s s_a(T0) %7.2f  σ = p/Tn %6.2f  σ_eff = f1+f2 %8.3f  α_eff = −f2 %.3f (α = 1.5)  r_s %.3f  r_m %.3f  r_τ %.3f  ⟨s⟩/s(T0)V %.4f  V d⟨s⟩/dV / s(T0)V %.4f\n",
                basename(f), row[2], sT * V, sT * V / 4, f1 + f2, -f2, c.vars / (CV * V), c.mx2 / (T0 * w * V),
                c.tau2 / T0^2, c.s / (sT * V), dsdV / (sT * V))
    end
end
close(out)
