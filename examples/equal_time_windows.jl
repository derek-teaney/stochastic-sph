# equal_time_windows.jl — equal-time C_AA(k,0)/χ_AA along the flow (k ∥ x) in time windows, to see whether
# a state started in exact equilibrium drifts away from it (docs/commonkernel.tex: is the C_EE deficit made by
# the ideal step?).  A_k = Σ_a a_a e^{−ik·x_a} for a = e (particle energy), m_x, m_y; ⟨|A_k|²⟩ = N V χ_AA in equilibrium.
#   julia --project examples/equal_time_windows.jl dir1 [dir2 ...]
using HDF5, Printf, StaticArrays, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "src", "load.jl"))
const WINDOWS = [(0.0, 5.0), (5.0, 20.0), (20.0, 50.0), (50.0, 100.0), (100.0, 300.0)]
const CS2 = 1 / 3

function Uof(θ)
    β0 = -θ[1];  b = SVector(θ[2], θ[3]);  T = 1 / sqrt(β0^2 - dot(b, b));  v = b / β0
    γ2 = 1 / (1 - dot(v, v));  e = T^4;  p = CS2 * e;  w = e + p
    SVector(w * γ2 - p, w * γ2 * v[1], w * γ2 * v[2])
end
function chi(Trel, v)
    γ = 1 / sqrt(1 - v^2);  θ = SVector(-γ / Trel, γ * v / Trel, 0.0);  J = zeros(3, 3)
    for j in 1:3
        d = 1e-6 * maximum(abs, θ);  e = SVector{3}(i == j ? d : 0.0 for i in 1:3)
        J[:, j] = (Uof(θ + e) - Uof(θ - e)) / 2d
    end
    (J + J') / 2
end

for d in ARGS
    fs = filter(f -> endswith(f, ".h5"), readdir(d; join = true));  isempty(fs) && continue
    runs = [read_equilibrium(f) for f in fs]
    r1 = runs[1];  m = r1.meta;  χ = chi(m.Trel, m.boost);  NV = m.N * m.V
    ks = r1.rec.ks
    js = sort([j for j in eachindex(ks) if abs(ks[j][2]) < 1e-9 && ks[j][1] > 0], by = j -> ks[j][1])[1:min(3, end)]
    @printf("%s  (v = %.1f, %d runs)\n", basename(d), m.boost, length(runs))
    for (t0, t1) in WINDOWS
        any(r -> any(t -> t0 <= t < t1, r.rec.t), runs) || continue
        line = @sprintf("  t ∈ [%3.0f, %3.0f):", t0, t1)
        for j in js
            per = zeros(length(runs), 3)
            for (i, r) in enumerate(runs)
                it = findall(t -> t0 <= t < t1, r.rec.t)
                isempty(r.rec.E) || (per[i, 1] = mean(abs2(r.rec.E[n][j]) for n in it) / (NV * χ[1, 1]))
                per[i, 2] = mean(abs2(r.rec.ML[n][j]) for n in it) / (NV * χ[2, 2])     # k ∥ x: M_L = M_x
                per[i, 3] = mean(abs2(r.rec.MT[n][j]) for n in it) / (NV * χ[3, 3])     #         M_T = M_y
            end
            n = length(runs);  mu = vec(mean(per, dims = 1));  er = vec(std(per, dims = 1)) ./ sqrt(n)
            line *= @sprintf("   kh=%.2f: E %.3f±%.3f  Mx %.3f±%.3f  My %.3f±%.3f", norm(ks[j]) * m.h, mu[1], er[1], mu[2], er[2], mu[3], er[3])
        end
        println(line)
    end
end
