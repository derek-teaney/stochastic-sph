using PyPlot
using Random
using Printf
include(joinpath(@__DIR__, "..", "src", "load.jl"))
include(joinpath(@__DIR__, "..", "src", "legacy", "load.jl"))   # two_level, particle_sweep (superseded)

# ── Shared resolution ─────────────────────────────────────────────────────────
const h_test  = 0.05
const ha_test = 8.0
const L_test  = 1.0
const cs2v    = 1.0 / 3.0
const s0      = 1.0     # entropy density (per unit area)

# Build a shear-wave IC (k=4π/L) with given M_amp for a dissipative test.
# scheme: discretization of the dissipative function (params.dissipation).
function make_test_ic(M_amp; η_by_s=0.0, jitter=0.0, scheme=:two_level)
    params, particles, cl, eos, _, _ = setup_shear_wave(h_test, ha_test;
        L=L_test, M_amp, s0, cs2v, η_by_s, jitter)
    params.dissipation = scheme
    return params, particles, cl, eos
end

println("=" ^ 60)
println("testdissipative.jl — GENERIC dissipative step tests")
println("=" ^ 60)

# Tests 1–3 run for both discretizations of the dissipative function.
for scheme in (:two_level, :bond)
println("\n" * "─" ^ 60)
println("dissipation = :$scheme")
println("─" ^ 60)

# ── Test 1: Zero viscosity → no force, entropy conserved ─────────────────────
println("\nTest 1: zero viscosity (η=ζ=0) → zero force everywhere")
let
    params, particles, cl, eos = make_test_ic(0.1; η_by_s=0.0, scheme)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    max_force = maximum(norm(f) for f in dotm)
    @assert max_force < 1e-14 "zero-viscosity force = $max_force"

    s_before = sum(pa.s for pa in particles)
    viscous_step!(particles, params, cl, 0.01, eos)
    s_after = sum(pa.s for pa in particles)
    Δs = s_after - s_before
    @assert abs(Δs) < 1e-14 "entropy changed by $Δs with zero viscosity"
    println("  max force = $max_force  ✓")
    println("  Δs_total  = $Δs  ✓")
end

# ── Test 2: Entropy production ≥ 0 for every particle ────────────────────────
println("\nTest 2: entropy production ≥ 0 for all particles (nonzero η, v≠0)")
let
    η_by_s = 0.3
    params, particles, cl, eos = make_test_ic(0.2; η_by_s, scheme)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    dt        = 0.001
    neg_count = 0
    for a in eachindex(particles)
        Δm   = dt .* dotm[a]
        Δs_a = -dot(particles[a].dotx / particles[a].Teff, Δm)
        Δs_a < -1e-14 && (neg_count += 1)
    end
    @assert neg_count == 0 "$neg_count particles with Δs < 0"
    println("  neg-entropy particles = $neg_count / $N  ✓")
end

# ── Test 3: Momentum conservation (Newton's 3rd law) ─────────────────────────
println("\nTest 3: total momentum conserved (Σ dotm = 0)")
let
    η_by_s = 0.2
    params, particles, cl, eos = make_test_ic(0.15; η_by_s, scheme)
    N    = params.N
    dotm = zeros(SVector{2,Float64}, N)
    dissipative_force!(dotm, particles, params, cl, eos)

    total     = sum(dotm)
    err       = sqrt(total[1]^2 + total[2]^2)
    sum_norms = sum(norm(f) for f in dotm)
    rel_err   = err / sum_norms
    @assert rel_err < 1e-14 "total viscous force rel. error = $rel_err (should be < 1e-14)"
    println("  |Σ dotm|        = $err")
    println("  Σ |dotm|        = $sum_norms")
    println("  rel. error           = $rel_err  ✓")
end
end   # for scheme

# ── Test 4: κ tensor spot-check on a few particles ───────────────────────────
println("\nTest 4: κ^{ijlm} tensor components at two sample particles")
let
    η_by_s = 0.2
    params, particles, cl, eos = make_test_ic(0.15; η_by_s, jitter=0.6)

    for idx in [1, params.N ÷ 2]
        pa    = particles[idx]
        K     = kappa_tensor(pa, eos, Kappa3D())
        vx    = pa.v_over_S[1] * pa.S_smooth
        vy    = pa.v_over_S[2] * pa.S_smooth
        v2    = vx^2 + vy^2
        T_rel = pa.T * pa.γ
        sr    = srel(eos, pa.e)
        # For shear wave (vx=0): κ[x,y,x,y] = T_rel·srel·η/s·(1−v²) exactly.
        expected = T_rel * sr * η_by_s * (1 - v2)
        xyxy_err = abs(K[1,2,1,2] - expected) / abs(expected)
        println("  particle $idx:  x=$(round.(pa.x, sigdigits=4))" *
                "  v=($(round(vx,sigdigits=4)), $(round(vy,sigdigits=4)))" *
                "  T_rel=$(round(T_rel, sigdigits=5))  srel=$(round(sr, sigdigits=5))")
        println("    κ[x,x,x,x] = $(round(K[1,1,1,1], sigdigits=6))")
        println("    κ[x,x,y,y] = $(round(K[1,1,2,2], sigdigits=6))")
        println("    κ[x,y,x,y] = $(round(K[1,2,1,2], sigdigits=6))" *
                "  expected T_rel·srel·η/s·(1−v²) = $(round(expected, sigdigits=6))" *
                "  rel_err=$(round(xyxy_err, sigdigits=2))")
        println("    κ[y,y,y,y] = $(round(K[2,2,2,2], sigdigits=6))")
        @assert xyxy_err < 1e-12 "κ[x,y,x,y] mismatch at particle $idx: rel_err=$xyxy_err"
    end
end

# ── Test 5: bond form — force = −∂K/∂b, 2K identity, rotation/dilation ──────
println("\nTest 5: dissipation = :bond — f = −∂K/∂b, −Σb·f = 2K, rotation & dilation give no force")
let
    η_by_s = 0.2
    params, particles, cl, eos = make_test_ic(0.15; η_by_s, jitter=0.3, scheme=:bond)
    N   = params.N
    rng = Xoshiro(5)
    bv  = [SVector(randn(rng), randn(rng)) for _ in 1:N]
    f   = zeros(SVector{2,Float64}, N)
    K0  = bond_dissipative_force!(f, particles, params, cl, eos, bv)

    # (a) entropy production identity for a quadratic K
    ent = -sum(dot(bv[i], f[i]) for i in 1:N)
    @assert K0 > 0 "K = $K0 not positive"
    @assert abs(ent - 2K0) / (2K0) < 1e-12 "−Σ b·f = $ent ≠ 2K = $(2K0)"
    @printf("  K = %.6e   −Σ b·f = %.6e   2K = %.6e  ✓\n", K0, ent, 2K0)

    # (b) finite differences of K vs −f
    maxerr = 0.0
    for (i, c) in ((1, 1), (17, 2), (N ÷ 3, 1), (N ÷ 2, 2), (N - 5, 1))
        δ  = 0.1    # K is exactly quadratic in b ⇒ central differences are exact for any δ;
                    # a large δ keeps round-off in the ~10⁶-term sum K out of the check
        e  = c == 1 ? SVector(δ, 0.0) : SVector(0.0, δ)
        bp = copy(bv); bp[i] += e;  bm = copy(bv); bm[i] -= e
        fd = (bond_dissipation(particles, params, cl, eos, bp) -
              bond_dissipation(particles, params, cl, eos, bm)) / (2δ)
        maxerr = max(maxerr, abs(fd + f[i][c]) / max(abs(fd), 1e-300))
    end
    @assert maxerr < 1e-5 "finite-difference mismatch $maxerr"
    @printf("  finite differences: max rel. error = %.1e  ✓\n", maxerr)

    # (c) rigid rotation and uniform dilation give no force (unjittered lattice:
    #     the SPH gradient is exact there for linear fields); a shear does not vanish.
    params, particles, cl, eos = make_test_ic(0.0; η_by_s, scheme=:bond)
    N   = params.N;  f = zeros(SVector{2,Float64}, N)
    xc  = params.L / 2
    inner = [i for i in 1:N if norm(minimum_image(particles[i].x - xc, params.L)) <
                                minimum(params.L) / 2 - 4params.h]
    function fmax(Gm)
        bvl = [Gm * minimum_image(p.x - xc, params.L) for p in particles]
        bond_dissipative_force!(f, particles, params, cl, eos, bvl)
        return maximum(norm(f[i]) for i in inner)
    end
    # a non-linear (sinusoidal) shear sets the scale of a genuine force
    bsin = [SVector(0.0, 1e-3 * sin(2π * p.x[1] / params.L[1])) for p in particles]
    bond_dissipative_force!(f, particles, params, cl, eos, bsin)
    scale = maximum(norm, f)
    rot = fmax(SMatrix{2,2}(0.0, 1e-3, -1e-3, 0.0))
    dil = fmax(SMatrix{2,2}(1e-3, 0.0, 0.0, 1e-3))
    @printf("  %d interior particles:  max|f| rotation %.1e  dilation %.1e   (sinusoidal shear: %.1e)\n",
            length(inner), rot, dil, scale)
    @assert rot < 1e-10 * scale "rigid rotation produces a force"
    @assert dil < 1e-10 * scale "uniform dilation produces a force"
    println("  rotation and dilation are force-free  ✓")
end

# ── Test 6: bond form on a moving background reproduces κ^{ijmn}(v) ──────────
# Uniform background velocity v, linear b-field b = Gᵀx with random G.  For each
# interior particle its share of K must equal ½ V T κ(v):SS (S = sym G), with κ from
# kappa_tensor — the comoving metric ℓ and the scalar projector P are what make this
# work.  Then f = −∂K/∂b by finite differences at v ≠ 0.
println("\nTest 6: dissipation = :bond on moving backgrounds — K = ½ V T κ(v):SS, f = −∂K/∂b")
let
    # the mean field is the exact bond fit ⇒ linear fields are reproduced to round-off
    for (cs2b, db, ζb, M0, tol) in ((0.5, 2, 0.0, SVector(0.40, 0.23), 1e-12),
                                    (0.5, 2, 0.0, SVector(0.52, 0.30), 1e-12),
                                    (1/3, 3, 0.0, SVector(0.40, 0.23), 1e-12),
                                    (1/3, 3, 0.5, SVector(0.52, 0.30), 1e-12))
        eosb = ConformalEOS(cs2b, T_scale_2d_lq1, 1.0, ζb)
        pb, ps, clb, eosb = setup_ic(15.0, 2.55, eosb, (x, y) -> 1.0, (x, y) -> M0;
                                     L=150.0, lattice=:hex, d=db)
        pb.dissipation = :bond
        Nb = pb.N;  xc = pb.L / 2
        vb = ps[1].v_over_S * ps[1].S_smooth
        @assert maximum(norm(p.v_over_S * p.S_smooth - vb) for p in ps) < 1e-10 "background not uniform"
        inner = [i for i in 1:Nb if norm(minimum_image(ps[i].x - xc, pb.L)) < minimum(pb.L) / 2 - 4pb.h]
        km  = db == 2 ? Kappa2D() : Kappa3D()
        rng = Xoshiro(11);  worst = 0.0;  Kv = zeros(Nb)
        for trial in 1:4
            G  = @SMatrix randn(rng, 2, 2)                     # G[p,q] = ∂_p b_q
            bv = [G' * minimum_image(p.x - xc, pb.L) for p in ps]
            bond_dissipation(ps, pb, clb, eosb, bv; Kvec=Kv)
            Sm = (G + G') / 2
            for i in inner
                κ  = kappa_tensor(ps[i], eosb, km)
                Ke = 0.5 * (ps[i].s / ps[i].S_smooth) *
                     sum(κ[a,b,c,e] * Sm[a,b] * Sm[c,e] for a in 1:2, b in 1:2, c in 1:2, e in 1:2)
                worst = max(worst, abs(Kv[i] - Ke) / Ke)
            end
        end
        # finite differences at v ≠ 0 (random b; K is quadratic ⇒ central differences exact)
        bv = [SVector(randn(rng), randn(rng)) for _ in 1:Nb]
        f  = zeros(SVector{2,Float64}, Nb)
        bond_dissipative_force!(f, ps, pb, clb, eosb, bv)
        fderr = 0.0
        for (i, c) in ((3, 1), (Nb ÷ 2, 2), (Nb - 7, 1))
            e  = c == 1 ? SVector(0.1, 0.0) : SVector(0.0, 0.1)
            bp = copy(bv); bp[i] += e;  bm = copy(bv); bm[i] -= e
            fd = (bond_dissipation(ps, pb, clb, eosb, bp) - bond_dissipation(ps, pb, clb, eosb, bm)) / 0.2
            fderr = max(fderr, abs(fd + f[i][c]) / abs(fd))
        end
        @printf("  d=%d cs²=%.3g ζ/s=%.1f |v|=%.3f:  max rel. error of K vs ½VTκ(v):SS = %.1e   FD error = %.1e\n",
                db, cs2b, ζb, norm(vb), worst, fderr)
        @assert worst < tol "bond K does not reproduce κ(v): $worst"
        @assert fderr < 1e-6 "finite-difference mismatch at v≠0: $fderr"
    end
    println("  κ^{ijmn}(v) reproduced on moving backgrounds  ✓")
end

# ── Test 7: exact-fit properties on a JITTERED, MOVING background ────────────
# The mean field is the bond-weighted least-squares fit (docs/onelevel1d.tex, App. A).
# On any arrangement and at any velocity:
#  (a) a linear b-field gives each particle exactly ½ V T κ(v_a):SS
#      (except particles where the Λ correction fell back to Λ = 0);
#  (b) K is blind to an added rigid rotation for an ARBITRARY b-field — the fitted
#      rotation minimizes K, so the bond forces are torque-free (angular momentum);
#  (c) f = −∂K/∂b by finite differences.
println("\nTest 7: dissipation = :bond, jittered + moving — exact fit: linear fields, rotation blindness, FD")
let
    eosb = ConformalEOS(1/3, T_scale_2d_lq1, 1.0, 0.3)
    pb, ps, clb, eosb = setup_ic(5.0, 2.55, eosb, (x, y) -> 1.0, (x, y) -> SVector(0.40, 0.23);
                                 L=150.0, lattice=:hex, d=3, jitter=0.3)
    pb.dissipation = :bond
    Nb = pb.N;  xc = pb.L / 2
    inner = [i for i in 1:Nb if norm(minimum_image(ps[i].x - xc, pb.L)) < minimum(pb.L) / 2 - 4pb.h]
    rng = Xoshiro(21);  Kv = zeros(Nb);  Kw = zeros(Nb)
    vmin, vmax = extrema(norm(p.v_over_S * p.S_smooth) for p in ps)

    # (a) linear field
    G  = @SMatrix randn(rng, 2, 2)
    bv = [G' * minimum_image(p.x - xc, pb.L) for p in ps]
    bond_dissipation(ps, pb, clb, eosb, bv; Kvec=Kv)
    Sm = (G + G') / 2
    errs = Float64[]
    for i in inner
        # the bonds use the comoving metric at ẋ_a (pa.dotx); evaluate κ(v) there too
        q = deepcopy(ps[i]);  q.v_over_S = ps[i].dotx / ps[i].S_smooth
        κ  = kappa_tensor(q, eosb, Kappa3D())
        Ke = 0.5 * (ps[i].s / ps[i].S_smooth) *
             sum(κ[a,b,c,e] * Sm[a,b] * Sm[c,e] for a in 1:2, b in 1:2, c in 1:2, e in 1:2)
        push!(errs, abs(Kv[i] - Ke) / Ke)
    end
    nbad = count(>(1e-10), errs)
    @printf("  |v| in [%.3f, %.3f], jitter 0.3a:  linear field  K_a = ½VTκ(v):SS  to %.1e for %d of %d particles\n",
            vmin, vmax, maximum(e for e in errs if e <= 1e-10), length(errs) - nbad, length(errs))
    @assert nbad <= 0.01 * length(errs) "too many particles miss ½VTκ(v):SS: $nbad"

    # (b) rotation blindness for an arbitrary field
    br  = [SVector(randn(rng), randn(rng)) for _ in 1:Nb]
    Ω   = SMatrix{2,2}(0.0, -0.7, 0.7, 0.0)
    bR  = [br[i] + Ω * minimum_image(ps[i].x - xc, pb.L) for i in 1:Nb]
    bond_dissipation(ps, pb, clb, eosb, br; Kvec=Kv)
    bond_dissipation(ps, pb, clb, eosb, bR; Kvec=Kw)
    rot = maximum(abs(Kw[i] - Kv[i]) / Kv[i] for i in inner)
    @printf("  arbitrary b + rigid rotation:  max rel. change of K_a = %.1e\n", rot)
    @assert rot < 1e-10 "K not blind to a rigid rotation: $rot"

    # (c) finite differences
    f = zeros(SVector{2,Float64}, Nb)
    bond_dissipative_force!(f, ps, pb, clb, eosb, br)
    fderr = 0.0
    for (i, c) in ((5, 1), (Nb ÷ 3, 2), (Nb - 11, 1))
        e  = c == 1 ? SVector(0.1, 0.0) : SVector(0.0, 0.1)
        bp = copy(br); bp[i] += e;  bm = copy(br); bm[i] -= e
        fd = (bond_dissipation(ps, pb, clb, eosb, bp) - bond_dissipation(ps, pb, clb, eosb, bm)) / 0.2
        fderr = max(fderr, abs(fd + f[i][c]) / abs(fd))
    end
    @printf("  finite differences: max rel. error = %.1e\n", fderr)
    @assert fderr < 1e-6 "finite-difference mismatch: $fderr"
    println("  exact-fit properties hold  ✓")
end

println("\n" * "=" ^ 60)
println("All testdissipative.jl tests passed.")
println("=" ^ 60)
