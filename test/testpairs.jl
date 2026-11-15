# testpairs.jl — the pair observables of ParticleRecorder (docs/equilibrium_positions.tex, stage 1):
#   1. the hexagonal lattice: g(r) is zero between the shells and its integral counts the shells (6 at a,
#      6 at √3 a, 6 at 2a); ψ6 = 1 (global and local); the harmonics g₂, g₄ vanish on the first shell and g₆ = ±g on the
#      first two (bonds at multiples of 60° and at 30° + 60°k);
#   2. Poisson positions: g(r) → 1 and S(k) = |N_k|²/N → 1 within the noise; ψ6 global ~ N^{-1/2}; the harmonics → 0;
#   3. the sum rule: n0 ∫ g 2πr dr up to R equals the mean number of neighbours within R, counted directly.
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
quiet(f) = redirect_stdout(f, devnull)
ok = true
check(c, msg) = (println(c ? "  ✓ " : "  ✗ FAIL: ", msg); global ok &= c)

eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)
params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 1.2, lattice = :hex))
N = params.N;  a = params.a;  A = params.L[1] * params.L[2];  n0 = N / A
ks = box_wavevectors(params, 1.0)

# n0 ∫_0^R g 2πr dr from the binned g (bins entirely inside R)
function count_within(p, R)
    m = floor(Int, R / p.dr + 1e-9)
    return n0 * sum(p.g[k] * π * ((k * p.dr)^2 - ((k - 1) * p.dr)^2) for k in 1:m)
end
direct(ps, R) = sum(count(b -> b != i && norm(minimum_image(ps[b].x - ps[i].x, params.L)) < R, eachindex(ps)) for i in eachindex(ps)) / length(ps)

println("1. hexagonal lattice, N = $N, a = $(round(a, digits = 4)), h/a = 2")
pr = ParticleRecorder(ps, params; ks, cl)
p = pr.pair0
r = (collect(1:pr.nbin) .- 0.5) .* pr.dr
shells = sqrt.((1, 3, 4, 7, 9, 12, 13, 16))           # |n1 a1 + n2 a2|/a of the hexagonal lattice, up to 4a
between = [k for k in 1:pr.nbin if all(abs(r[k] / a - s) > 0.06 for s in shells)]
check(all(p.g[between] .== 0), "g = 0 between the shells")
for (R, n) in ((1.2a, 6), (1.9a, 12), (2.2a, 18))
    c = count_within((g = p.g, dr = pr.dr), R)
    check(abs(c - n) < 1e-9, @sprintf("neighbours within %.1f a: %.6f (expected %d)", R / a, c, n))
end
check(abs(p.psi6 - 1) < 1e-12 && abs(p.psi6loc - 1) < 1e-12, @sprintf("ψ6 = %.12f, local %.12f", p.psi6, p.psi6loc))
gpar = count_within((g = p.gpar, dr = pr.dr), 1.2a);  gperp = count_within((g = p.gperp, dr = pr.dr), 1.2a)
check(abs(gpar + gperp - 12) < 1e-9, @sprintf("g_par, g_perp at the first shell: %.3f + %.3f (each normalized by half the shell; sum 12)", gpar, gperp))
cnt(G, R1, R2) = n0 * sum(G[k] * π * ((k * pr.dr)^2 - ((k - 1) * pr.dr)^2) for k in floor(Int, R1 / pr.dr + 1e-9)+1:floor(Int, R2 / pr.dr + 1e-9))
c2 = cnt(p.gn[:, 1], 0, 1.2a);  c4 = cnt(p.gn[:, 2], 0, 1.2a);  c6 = cnt(p.gn[:, 3], 0, 1.2a);  c6b = cnt(p.gn[:, 3], 1.5a, 1.9a)
check(abs(c2) < 1e-9 && abs(c4) < 1e-9 && abs(c6 - 6) < 1e-9 && abs(c6b + 6) < 1e-9,
      @sprintf("harmonics, first shell: n0∫g₂ = %.1e, n0∫g₄ = %.1e, n0∫g₆ = %.6f (6); √3 a shell n0∫g₆ = %.6f (−6)", c2, c4, c6, c6b))

println("2. Poisson positions")
rng = Xoshiro(7)
for q in ps;  q.x = SVector(rand(rng) * params.L[1], rand(rng) * params.L[2]);  end
rebuild_cell_list!(cl, ps)
po = ParticleRecorder(ps, params; ks, cl).pair0
far = [k for k in 1:pr.nbin if r[k] > a]
gm = sum(po.g[far]) / length(far);  gs = sqrt(sum(abs2, po.g[far] .- 1) / length(far))
check(abs(gm - 1) < 0.03, @sprintf("mean g(r > a) = %.4f (rms deviation per bin %.3f)", gm, gs))
Sk = abs2.(po.Nk) ./ N;  Sm = sum(Sk) / length(Sk)
check(abs(Sm - 1) < 3 / sqrt(length(Sk)), @sprintf("mean S(k) over %d wave vectors = %.3f", length(Sk), Sm))
check(po.psi6 < 5 / sqrt(N), @sprintf("ψ6 global = %.4f (N^{-1/2} = %.4f), local %.3f", po.psi6, 1 / sqrt(N), po.psi6loc))
hm = [sum(po.gn[far, j]) / length(far) for j in 1:3]
check(all(abs.(hm) .< 0.03), @sprintf("harmonics of Poisson positions, mean over r > a: g₂ %.4f, g₄ %.4f, g₆ %.4f", hm...))

println("3. sum rule (Poisson positions)")
for R in (1.0a, 2.0a, 3.5a)
    c = count_within((g = po.g, dr = pr.dr), R);  d = direct(ps, R)
    check(abs(c - d) < 1e-9 * max(1, d), @sprintf("R = %.1f a: n0 ∫ g = %.6f, direct %.6f", R / a, c, d))
end

println("4. the angle convention: a rotated lattice and aligned dimers")
params, ps, cl, _ = quiet(() -> setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 1.2, lattice = :hex))
# the hexagonal lattice turned by 30° about one site, in a box large enough that the first two shells are unaffected far from its
# edges: compare only particles away from the box boundary, through the counts of a single central particle's neighbours
c0 = params.L / 2;  R = SMatrix{2,2}(cosd(30), sind(30), -sind(30), cosd(30))
x0 = [q.x for q in ps];  i0 = argmin([norm(x - c0) for x in x0])
nb = [R * minimum_image(x0[j] - x0[i0], params.L) for j in eachindex(x0) if j != i0]
first = [r for r in nb if norm(r) < 1.2a];  second = [r for r in nb if 1.5a < norm(r) < 1.9a]
s6a = sum(cos(6atan(r[2], r[1])) for r in first);  s6b = sum(cos(6atan(r[2], r[1])) for r in second)
check(abs(s6a + 6) < 1e-9 && abs(s6b - 6) < 1e-9, @sprintf("turned by 30°: Σ cos 6φ over the first shell %.6f (−6), second %.6f (+6)", s6a, s6b))
# dimers: Poisson centres, each with a partner at distance 0.5a along x (then along y).  Within 0.45a-0.55a every particle has its
# partner (cos 2φ = +1 along x, −1 along y, cos 4φ = +1 either way) plus about n0·π(0.55²−0.45²)a² ≈ 0.36 random neighbours
rng = Xoshiro(11);  half = N ÷ 2
cen = [SVector(rand(rng) * params.L[1], rand(rng) * params.L[2]) for _ in 1:half]
for (dir, s2, lab) in ((SVector(1.0, 0.0), 1.0, "along x"), (SVector(0.0, 1.0), -1.0, "along y"))
    for i in 1:half
        ps[2i-1].x = cen[i];  ps[2i].x = cen[i] + 0.5a * dir
    end
    wrap_positions!(ps, params.L);  rebuild_cell_list!(cl, ps)
    pd = ParticleRecorder(ps, params; cl).pair0
    sh(G) = cnt(G, 0.45a, 0.55a)
    local c, c2, c4 = sh(pd.g), sh(pd.gn[:, 1]), sh(pd.gn[:, 2])
    check(abs(c - 1.36) < 0.15 && abs(c2 - s2) < 0.15 && abs(c4 - 1) < 0.15,
          @sprintf("dimers %s, shell 0.45–0.55 a: n0∫g = %.3f (≈ 1.36), n0∫g₂ = %+.3f (%+.0f), n0∫g₄ = %+.3f (+1)", lab, c, c2, s2, c4))
end

println(ok ? "testpairs: PASS" : "testpairs: FAIL");  ok || exit(1)
