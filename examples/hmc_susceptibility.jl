# hmc_susceptibility.jl — the equal-time correlators of the conserved densities in the HMC equilibrium against the thermodynamic
# susceptibilities χ (docs/equilibrium_positions.tex): C_AA(k) = ⟨|A_k|²⟩/(N V), A_k = Σ_a a_a e^{−ik·x_a}, a = particle energy, M_x, M_y.
#   (1) the initial states of the HMC-start runs of hmc_liou2 (first record, t ≈ 0.03, after the HMC start): E, M_x, M_y, along x and y;
#   (2) the HMC chains of hmc_eq (after a burn-in of 100 updates): M_L, M_T in bands of kh (energies not recorded there).
include(joinpath(@__DIR__, "..", "src", "load.jl"));  using Statistics, Printf, StaticArrays, LinearAlgebra
const CS2 = 1 / 3                                    # campaigns hmc_liou2, hmc_eq, before dimfix
pre_dimfix(cs) = abs(cs - 1 / 3) < 1e-12 || error("this script is written for the runs before dimfix (cs² = 1/3, e = T⁴; docs/dimension.tex); the run has cs² = $cs")
function Uof(θ)
    β0 = -θ[1];  b = SVector(θ[2], θ[3]);  T = 1 / sqrt(β0^2 - dot(b, b));  v = b / β0
    γ2 = 1 / (1 - dot(v, v));  e = T^4;  p = CS2 * e;  w = e + p
    SVector(w * γ2 - p, w * γ2 * v[1], w * γ2 * v[2])
end
function chi(Trel, v)
    γ = 1 / sqrt(1 - v^2);  θ = SVector(-γ / Trel, γ * v / Trel, 0.0);  J = zeros(3, 3)
    for j in 1:3
        d = 1e-6 * maximum(abs, θ);  e = SVector{3}(i == j ? d : 0.0 for i in 1:3);  J[:, j] = (Uof(θ + e) - Uof(θ - e)) / 2d
    end
    (J + J') / 2
end
ms(v) = @sprintf("%.3f(%d)", mean(v), round(Int, 1000 * std(v) / sqrt(length(v))))
D = ENV["SCRATCH"] * "/sph"
println("(1) initial states of the HMC-start runs (hmc_liou2), C_AA(k)/χ_AA along x (k ∥ x) and y (k ∥ y), mean(error) over runs")
for v in ("0", "0.8")
    rs = []
    for c in ("glf", "dkd")
        d = "$D/hmc_liou2/hl2_$(c)_hmc_v$v"
        for f in readdir(d; join = true)
            endswith(f, ".h5") || continue
            r = read_equilibrium(f);  rec = r.rec;  m = r.meta;  pre_dimfix(get(m, :cs2, 1 / 3))
            push!(rs, (E = [x for x in rec.E[1]], ML = [x for x in rec.ML[1]], MT = [x for x in rec.MT[1]], ks = rec.ks, NV = m.N * m.V, Trel = m.Trel, v = m.boost, h = m.h))
            r = nothing
        end
    end
    r1 = rs[1];  χ = chi(r1.Trel, r1.v);  ks = r1.ks
    @printf("  v = %s (%d starts): χ_EE %.4g  χ_MxMx %.4g  χ_MyMy %.4g\n", v, length(rs), χ[1,1], χ[2,2], χ[3,3])
    for (dir, sel) in (("k∥x", i -> abs(ks[i][2]) < 1e-9 && ks[i][1] > 0), ("k∥y", i -> abs(ks[i][1]) < 1e-9 && ks[i][2] > 0))
        js = sort(filter(sel, eachindex(ks)), by = i -> norm(ks[i]))[1:3]
        for j in js
            kh = norm(ks[j]) * r1.h;  along = dir == "k∥x"
            E = [abs2(r.E[j]) / r.NV / χ[1, 1] for r in rs]
            Mx = [abs2(along ? r.ML[j] : r.MT[j]) / r.NV / χ[2, 2] for r in rs]      # k ∥ x: M_x = M_L; k ∥ y: M_x = M_T (up to sign)
            My = [abs2(along ? r.MT[j] : r.ML[j]) / r.NV / χ[3, 3] for r in rs]
            @printf("    %s kh=%.2f:  E %s   M_x %s   M_y %s\n", dir, kh, ms(E), ms(Mx), ms(My))
        end
    end
end
println("(2) HMC chains (hmc_eq, burn-in 100 updates): C_MM(k)/χ_MM in bands of kh, longitudinal and transverse, mean(error) over chains")
for set in ("hq12_s54", "hq12_s135", "hq12_s541", "hq24_s541")
    d = "$D/hmc_eq/$set";  isdir(d) || continue
    per = []
    for f in readdir(d; join = true)
        endswith(f, ".h5") || continue
        r = read_equilibrium(f);  rec = r.rec;  m = r.meta;  pre_dimfix(get(m, :cs2, 1 / 3));  V = m.s_a / (srel0_of(m) * (m.boost == 0 ? 1 : 1 / sqrt(1 - m.boost^2)))
        n = length(rec.t);  it = 101:n;  χ = chi(m.T0, m.boost)
        kh = [norm(k) * m.h for k in rec.ks]
        push!(per, (kh = kh, L = [mean(abs2(rec.ML[s][i]) for s in it) / (m.N * V) / χ[2, 2] for i in eachindex(rec.ks)],
                             T = [mean(abs2(rec.MT[s][i]) for s in it) / (m.N * V) / χ[3, 3] for i in eachindex(rec.ks)]))
        r = nothing
    end
    line = @sprintf("  %-10s (%2d chains):", set, length(per))
    for kc in (0.1, 0.3, 0.5, 1.0, 1.5)
        js = findall(x -> abs(x - kc) < 0.05, per[1].kh);  isempty(js) && continue
        line *= @sprintf("  kh %.1f: L %s T %s", kc, ms([mean(p.L[js]) for p in per]), ms([mean(p.T[js]) for p in per]))
    end
    println(line)
end
