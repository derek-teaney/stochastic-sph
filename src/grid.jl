# grid.jl — the ideal step on a fixed lattice: kick, drift and return (docs/gridstep.tex).
#
# The particles sit on the sites R_a of a perfect hexagonal lattice and never move.  Each
# site holds an energy e_a, a momentum m_a (density frame, m_a/V = T^{0i}) and an entropy s_a,
# related by the equation of state at the common volume V = 1/n_0, n_0 = Σ_b W(r_ab).  One
# ideal step is
#     kick(Δt/2) → transport(Δt) → kick(Δt/2),
# the kick at fixed s_a (ṁ_a = −V² Σ_b (p_a + p_b) D_ab), the transport (the drift with the
# return to the sites folded in) on (e_a, m_a), eq. (transport) of the note; each substep is
# integrated with one predictor and one corrector (Heun).  D_ab = ∂W(R_a − R_b)/∂R_a.
#
# The state lives in the Particle vector (x fixed on the sites, s and m updated), so the
# stochastic step at fixed positions, the recorders and the run records work unchanged.
# Public:  GridStencil(particles, params, cl), grid_checks(st, eos, dt), grid_step!(ps, st, dt, eos, gs)
#          GridScratch(N), GridStats() (per-step diagnostics, summed over a run).

# ── The stencil ──────────────────────────────────────────────────────────────
struct GridStencil
    N      :: Int
    V      :: Float64                      # 1/n_0
    n0     :: Float64
    a      :: Float64
    off    :: Vector{Int}                  # neighbours of a: nb[off[a]:off[a+1]-1]  (b ≠ a)
    nb     :: Vector{Int}
    D      :: Vector{SVector{2,Float64}}   # D_ab = ∂W(R_a − R_b)/∂R_a
    r      :: Vector{SVector{2,Float64}}   # r_ab = R_a − R_b (minimum image)
    nspread :: Float64                     # max |n_a/n_0 − 1| over the sites (0 on a perfect lattice)
end

function GridStencil(particles::Vector{Particle}, params::SimParams, cl::CellList2D)
    nl = neighbor_list!(cl, particles, params.h)
    N = length(particles);  h = params.h;  K = params.kernel
    off = ones(Int, N + 1);  nb = Int[];  D = SVector{2,Float64}[];  r = SVector{2,Float64}[]
    n = zeros(N)
    for a in 1:N
        for b in neighbors(nl, a)
            rv = minimum_image(particles[a].x - particles[b].x, params.L);  rm = norm(rv)
            n[a] += kernel_W(rm, h, K)
            (b == a || rm >= 2h) && continue
            push!(nb, b);  push!(D, kernel_gradW(rv, rm, h, K));  push!(r, rv)
        end
        off[a + 1] = length(nb) + 1
    end
    n0 = sum(n) / N
    return GridStencil(N, 1 / n0, n0, params.a, off, nb, D, r, maximum(abs.(n ./ n0 .- 1)))
end

# The checks of gridstep.tex, "Once, at set-up": Σ_b D_ab, n_0 and V (N V against the box),
# the first moment c_1^{ij} = −V Σ_b r^i D^j (should be δ^{ij}), κ_max over the box modes
# and c_s κ_max Δt (< 2 for the leapfrog of sound).
function grid_checks(st::GridStencil, params::SimParams, eos::AbstractEOS, dt::Float64)
    sumD = maximum(norm(sum(st.D[st.off[a]:st.off[a+1]-1])) for a in 1:st.N) * st.V * st.a
    c1 = zeros(2, 2);  rng = st.off[1]:st.off[2]-1
    for j in rng, i1 in 1:2, i2 in 1:2;  c1[i1, i2] -= st.V * st.r[j][i1] * st.D[j][i2];  end
    c1max = maximum(begin
        c = zeros(2, 2)
        for j in st.off[a]:st.off[a+1]-1, i1 in 1:2, i2 in 1:2;  c[i1, i2] -= st.V * st.r[j][i1] * st.D[j][i2];  end
        maximum(abs.(c .- c1))
    end for a in 1:st.N)
    # κ(k) = |V Σ_b D_ab e^{−ik·r_ab}| over the box modes (all classes of the periodic lattice)
    Nx = round(Int, params.L[1] / params.a);  Ny = round(Int, st.N / Nx)
    κmax = 0.0;  kmax = SVector(0.0, 0.0)
    for i in -Nx:Nx, j in -Ny:Ny
        k = SVector(2π * i / params.L[1], 2π * j / params.L[2])
        s = zero(SVector{2,ComplexF64})
        for q in rng;  s += st.D[q] * cis(-dot(k, st.r[q]));  end
        κ = st.V * norm(s);  κ > κmax && (κmax = κ;  kmax = k)
    end
    cs = sqrt(cs2(eos))
    return (sumD_rel = sumD, n0 = st.n0, V = st.V, NV_over_area = st.N * st.V / (params.L[1] * params.L[2]),
            nspread = st.nspread, c1xx = c1[1, 1] - 1, c1yy = c1[2, 2] - 1, c1xy = c1[1, 2], c1_site_spread = c1max,
            kappa_max = κmax, k_at_kappa_max_h = norm(kmax) * params.h, cs_kappa_dt = cs * κmax * dt,
            neighbours = length(rng))
end

function print_grid_checks(c)
    @printf("grid: n0 = %.6g  V = %.6g  N V/area − 1 = %+.2e  max|n_a/n0 − 1| = %.1e  neighbours %d\n",
            c.n0, c.V, c.NV_over_area - 1, c.nspread, c.neighbours)
    @printf("grid: max|Σ_b D_ab| V a = %.1e   c1 − 1 = (%+.4f, %+.4f), c1xy = %+.1e (site spread %.1e)\n",
            c.sumD_rel, c.c1xx, c.c1yy, c.c1xy, c.c1_site_spread)
    @printf("grid: κ_max = %.4g (at kh = %.3g)   c_s κ_max Δt = %.4f  (sound stable below 2)\n",
            c.kappa_max, c.k_at_kappa_max_h, c.cs_kappa_dt)
end

# ── The equation of state at the fixed volume V ─────────────────────────────
# forward: (s, m) → (e, p, v, e_rest); e = V(wγ² − p) the energy of the site (warm start er0)
@inline function grid_forward(s::Float64, m::SVector{2,Float64}, V::Float64, er0::Float64, eos::AbstractEOS)
    S = s / V;  M = m / V;  Mm = norm(M)
    er = eos_newton(S, Mm, er0, eos)
    p = pressure(eos, er);  w = er + p
    G, _ = eos_wgamma(w, Mm);  γ = G / w
    return V * (w * γ^2 - p), p, M * (w / G^2), er
end
# inverse: (e, m) → (s, p, v, e_rest, ok); ok = false when e ≤ |m| (no state)
@inline function grid_inverse(e::Float64, m::SVector{2,Float64}, V::Float64, eos::AbstractEOS)
    ε = e / V;  M = m / V;  M2 = dot(M, M)
    (isfinite(ε) && ε > 0 && M2 < ε^2) || return (NaN, NaN, SVector(NaN, NaN), NaN, false)
    er = rest_energy(ε, M2, eos);  p = pressure(eos, er);  w = er + p
    v = M / (ε + p);  v2 = dot(v, v)
    v2 < 1 || return (NaN, NaN, SVector(NaN, NaN), NaN, false)
    S = w / sqrt(1 - v2) / Trel(eos, er)
    return V * S, p, v, er, true
end

# ── Rates ────────────────────────────────────────────────────────────────────
function grid_force!(F, p, st::GridStencil)
    V2 = st.V^2
    Threads.@threads for a in 1:st.N
        acc = zero(SVector{2,Float64});  pa = p[a]
        @inbounds for q in st.off[a]:st.off[a+1]-1
            acc -= (pa + p[st.nb[q]]) * st.D[q]
        end
        F[a] = V2 * acc
    end
end

# transport rates, eq. (transport) of gridstep.tex
function grid_transport!(de, dm, e, m, v, p, st::GridStencil)
    V = st.V
    Threads.@threads for a in 1:st.N
        ea = e[a];  ma = m[a];  va = v[a]
        fe = 0.0;  dv = 0.0;  fm = zero(SVector{2,Float64})
        @inbounds for q in st.off[a]:st.off[a+1]-1
            b = st.nb[q];  D = st.D[q];  vb = v[b]
            fe += dot(ea * va + e[b] * vb, D)
            dv += dot(vb - va, D)
            fm += ma * dot(va, D) + m[b] * dot(vb, D)
        end
        de[a] = -V * fe - p[a] * V^2 * dv
        dm[a] = -V * fm
    end
end

# (∇·v)_a = V Σ_b (v_b − v_a)·D_ab
function grid_divergence(v, st::GridStencil, a::Int)
    d = 0.0
    @inbounds for q in st.off[a]:st.off[a+1]-1;  d += dot(v[st.nb[q]] - v[a], st.D[q]);  end
    return st.V * d
end

# ── Scratch and diagnostics ──────────────────────────────────────────────────
struct GridScratch
    s  :: Vector{Float64};  m  :: Vector{SVector{2,Float64}};  e :: Vector{Float64}
    p  :: Vector{Float64};  v  :: Vector{SVector{2,Float64}};  er :: Vector{Float64}
    F  :: Vector{SVector{2,Float64}};  F2 :: Vector{SVector{2,Float64}}
    de :: Vector{Float64};  dm :: Vector{SVector{2,Float64}}
    de2 :: Vector{Float64};  dm2 :: Vector{SVector{2,Float64}}
    m0 :: Vector{SVector{2,Float64}};  e0 :: Vector{Float64};  ok :: Vector{Bool}
end
GridScratch(N) = GridScratch(zeros(N), zeros(SVector{2,Float64}, N), zeros(N), zeros(N), zeros(SVector{2,Float64}, N),
                             zeros(N), zeros(SVector{2,Float64}, N), zeros(SVector{2,Float64}, N), zeros(N),
                             zeros(SVector{2,Float64}, N), zeros(N), zeros(SVector{2,Float64}, N),
                             zeros(SVector{2,Float64}, N), zeros(N), fill(true, N))

# Per step (last) and summed over the run (sums of the substep changes).
mutable struct GridStats
    steps :: Int
    dH_K1 :: Float64;  dH_T :: Float64;  dH_K2 :: Float64      # last step, relative to H
    dS_T  :: Float64                                           # last step, relative to S
    sum_dH :: Float64;  sum_dS_T :: Float64;  sum_absdS_T :: Float64
    dP_max :: Float64                                          # max over substeps of |ΔΣm| / Σ|m|
    roundtrip_max :: Float64                                   # max |s(e(s)) − s|/s after the kicks
    disp_max :: Float64                                        # max over the run of max_a |v_a| Δt / a
    div_max :: Float64                                         # max over the run of max_a |∇·v| Δt
    eos_fail :: Int                                            # sites with no state (transport predictor/corrector)
    first_fail :: String
end
GridStats() = GridStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "")

grid_summary(g::GridStats) = (steps = g.steps, sum_dH_over_H = g.sum_dH, sum_dS_T_over_S = g.sum_dS_T,
    sum_absdS_T_over_S = g.sum_absdS_T, dP_max = g.dP_max, roundtrip_max = g.roundtrip_max,
    disp_max = g.disp_max, div_max = g.div_max, eos_fail = g.eos_fail, first_fail = g.first_fail)

# ── The step ─────────────────────────────────────────────────────────────────
# kick(τ) at fixed s: m ← m + τ/2 (F(p(s,m)) + F(p(s, m + τF))); then e from the forward EOS.
function grid_kick!(sc::GridScratch, st::GridStencil, τ::Float64, eos::AbstractEOS)
    (; s, m, e, p, v, er, F, F2, m0) = sc;  V = st.V;  N = st.N
    Threads.@threads for a in 1:N
        _, p[a], _, er[a] = grid_forward(s[a], m[a], V, er[a], eos);  m0[a] = m[a]
    end
    grid_force!(F, p, st)
    Threads.@threads for a in 1:N
        m[a] = m0[a] + τ * F[a]
        _, p[a], _, er[a] = grid_forward(s[a], m[a], V, er[a], eos)
    end
    grid_force!(F2, p, st)
    Threads.@threads for a in 1:N
        m[a] = m0[a] + 0.5τ * (F[a] + F2[a])
        e[a], p[a], v[a], er[a] = grid_forward(s[a], m[a], V, er[a], eos)
    end
end

# transport(Δt) on (e, m): Heun with the rates eq. (transport); then s from the inverse EOS.
# Returns the number of sites with no state (predictor + corrector); on failure the state is
# left at the start of the substep with s unchanged.
function grid_transport_step!(sc::GridScratch, st::GridStencil, dt::Float64, eos::AbstractEOS, g::GridStats, n::Int)
    (; s, m, e, p, v, er, de, dm, de2, dm2, m0, e0, ok) = sc;  V = st.V;  N = st.N
    Threads.@threads for a in 1:N
        sa, p[a], v[a], er[a], ok[a] = grid_inverse(e[a], m[a], V, eos)
    end
    nfail = count(!, ok)
    nfail > 0 && (record_fail!(g, n, "transport start", ok);  return nfail)
    g.disp_max = max(g.disp_max, maximum(norm, v) * dt / st.a)
    g.div_max = max(g.div_max, maximum(abs(grid_divergence(v, st, a)) for a in 1:N) * dt)
    grid_transport!(de, dm, e, m, v, p, st)
    Threads.@threads for a in 1:N
        e0[a] = e[a];  m0[a] = m[a]
        e[a] += dt * de[a];  m[a] += dt * dm[a]
        _, p[a], v[a], er[a], ok[a] = grid_inverse(e[a], m[a], V, eos)
    end
    nfail = count(!, ok)
    if nfail > 0
        record_fail!(g, n, "transport predictor", ok)
        e .= e0;  m .= m0;  return nfail
    end
    grid_transport!(de2, dm2, e, m, v, p, st)
    Threads.@threads for a in 1:N
        e[a] = e0[a] + 0.5dt * (de[a] + de2[a]);  m[a] = m0[a] + 0.5dt * (dm[a] + dm2[a])
        sn, p[a], v[a], er[a], ok[a] = grid_inverse(e[a], m[a], V, eos)
        ok[a] && (s[a] = sn)
    end
    nfail = count(!, ok)
    if nfail > 0
        record_fail!(g, n, "transport corrector", ok)
        e .= e0;  m .= m0
    end
    return nfail
end

function record_fail!(g::GridStats, n, where, ok)
    g.eos_fail += count(!, ok)
    isempty(g.first_fail) && (g.first_fail = "step $n, $where, site $(findfirst(!, ok))")
end

# One ideal step: kick(Δt/2), transport(Δt), kick(Δt/2), on particles[a].s, .m (positions fixed).
# Fills the diagnostics g; returns the number of sites with no state (0 normally).
function grid_step!(particles::Vector{Particle}, st::GridStencil, dt::Float64, eos::AbstractEOS,
                    sc::GridScratch, g::GridStats; n::Int = g.steps + 1)
    (; s, m, e) = sc;  N = st.N
    for a in 1:N;  s[a] = particles[a].s;  m[a] = particles[a].m;  end
    Pabs = sum(norm, m) + eps()
    Threads.@threads for a in 1:N
        e[a], _, _, sc.er[a] = grid_forward(s[a], m[a], st.V, sc.er[a], eos)
    end
    H0 = sum(e);  S0 = sum(s);  P0 = sum(m)
    grid_kick!(sc, st, dt / 2, eos)
    H1 = sum(e);  P1 = sum(m)
    rt = roundtrip(sc, st, eos)
    nfail = grid_transport_step!(sc, st, dt, eos, g, n)
    H2 = sum(e);  S2 = sum(s);  P2 = sum(m)
    grid_kick!(sc, st, dt / 2, eos)
    H3 = sum(e);  P3 = sum(m)
    rt = max(rt, roundtrip(sc, st, eos))
    for a in 1:N;  particles[a].s = s[a];  particles[a].m = m[a];  end
    g.steps += 1
    g.dH_K1 = (H1 - H0) / H0;  g.dH_T = (H2 - H1) / H0;  g.dH_K2 = (H3 - H2) / H0
    g.dS_T = (S2 - S0) / S0
    g.sum_dH += (H3 - H0) / H0;  g.sum_dS_T += g.dS_T;  g.sum_absdS_T += abs(g.dS_T)
    g.dP_max = max(g.dP_max, norm(P1 - P0) / Pabs, norm(P2 - P1) / Pabs, norm(P3 - P2) / Pabs)
    g.roundtrip_max = max(g.roundtrip_max, rt)
    return nfail
end

# max_a |s(e_a, m_a) − s_a| / s_a after a kick: the round trip s → e → s of the EOS
function roundtrip(sc::GridScratch, st::GridStencil, eos::AbstractEOS)
    r = 0.0
    for a in 1:st.N
        s2, _, _, _, ok = grid_inverse(sc.e[a], sc.m[a], st.V, eos)
        r = max(r, ok ? abs(s2 / sc.s[a] - 1) : Inf)
    end
    return r
end

# Fraction of the fluctuation power Σ_k |f_k|² at kh > khc (f = m − m̄ or e − ē on the sites):
# 1 − (power in the box modes with 0 < kh ≤ khc) / (N Σ_a |f_a|²)  (Parseval on the lattice).
function grid_highk_fraction(particles::Vector{Particle}, params::SimParams, f::Vector; khc = 1.0)
    N = length(particles);  fb = sum(f) / N;  tot = N * sum(x -> sum(abs2, x - fb), f)
    tot == 0 && return 0.0
    kc = khc / params.h;  lo = 0.0
    imax = floor(Int, kc * params.L[1] / 2π);  jmax = floor(Int, kc * params.L[2] / 2π)
    for i in -imax:imax, j in -jmax:jmax
        (i == 0 && j == 0) && continue
        k = SVector(2π * i / params.L[1], 2π * j / params.L[2]);  norm(k) <= kc || continue
        c = sum((f[a] - fb) * cis(-dot(k, particles[a].x)) for a in 1:N)
        lo += sum(abs2, c)
    end
    return 1 - lo / tot
end
