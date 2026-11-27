# stochastic.jl — the stochastic viscous step on the particle variables (docs/stochastic.tex,
# docs/inverse_moment.tex).  Other forms of the step: tag pre-prune.
#
# During the step the positions, the particle volumes V_a and the particle energies e_a
# are fixed; only the momenta move.  The total particle entropy
#     𝕊[m] = Σ_a S_a(e_a, m_a),      ∂S_a/∂m_a = −β_a ,
# is a sum of single-particle terms.
#
# Two sweeps, each over a checkerboard of cells:
#   stress sweep    particle a draws a random symmetric stress ξ_a with
#                   ⟨ξξ⟩ = 2Δt T_a κ_a / V_a and gives the traction ξ_a A_ab to each
#                   bond b.  Mean: the resolved part of the bond force.
#   residual sweep  particle a draws a random vector y_ab on each bond, ⟨y yᵀ⟩ = 2Δt c_ab,
#                   conditioned on zero virial Σ_b y_ab r_abᵀ.
#                   Mean: the residual part of the bond force.
#
# Each move is accepted with one vote of the star, min(1, exp ΔS), ΔS = Σ_c ΔS_c, all or
# nothing; with the Hastings correction, min(1, exp(ΔS + ln q′/q)).
#
# The geometry of the stars (w_ab, faces) is frozen at the start of the step, at the
# particle velocities of that moment, in the comoving forms of docs/stochastic.tex
# ("A moving fluid").  Without the Hastings correction the coefficients (T_rel, η, ζ, c_ab,
# ℓ, P) are frozen too; with it they are recomputed at each proposal.
#
# Moves in cells of the same color touch disjoint sets of particles and run in parallel;
# within a cell the moves are sequential, each seeing the current momenta.
#
# The random numbers of the move of particle a come from rngs[a]; the sweep orders from the
# master rng.
#
# Loaded by load.jl (uses pointwise.jl and rng.jl).

# ℓ^{1/2} for ℓ = 𝟙 − v vᵀ (|v| < 1)
function sqrt_comoving(v::SVector{2,Float64})
    v = cap_velocity(v)                                 # |v|² ≤ 1 − 1/γ_max² (sph.jl)
    v2 = dot(v, v)
    v2 == 0 && return SMatrix{2,2,Float64}(I)
    return SMatrix{2,2,Float64}(I) - (1 - sqrt(1 - v2)) * (v * v') / v2
end

# ── Frozen star data ─────────────────────────────────────────────────────────
struct Star
    nb :: Vector{Int}                   # bonds b with w_ab > 0
    r  :: Vector{SVector{2,Float64}}    # r_ab = x_b − x_a (minimum image)
    w  :: Vector{Float64}               # w_ab = V_b ψ_ab / n_a;  Σ_b w_ab r_ab r_abᵀ has trace 2
    c  :: Vector{Float64}               # residual coefficient: K_a ⊃ ½ Σ_b c_ab |δ_ab|²
    V  :: Float64                       # V_a
    T  :: Float64                       # T_rel,a  (rest-frame temperature, not pa.T = T_rel/γ)
    κs :: Float64                       # η_a
    κb :: Float64                       # ζ_eff,a
    A  :: Vector{SVector{2,Float64}}    # face A_ab = V_a w_ab 𝓜_a⁻¹ r_ab (moment_faces!)
    B  :: NTuple{3,SMatrix{2,2,Float64,4}}  # stress basis: ξ = √(2Δt T_rel/V) Σ_k z_k B_k
    ℓ  :: SMatrix{2,2,Float64,4}        # comoving metric ℓ_a = 𝟙 − v vᵀ
    ℓh :: SMatrix{2,2,Float64,4}        # ℓ_a^{1/2}
    ℓi :: SMatrix{2,2,Float64,4}        # ℓ_a⁻¹
    U  :: Vector{SVector{2,Float64}}    # the conditioning vectors u_ab = c_ab R_a⁻¹ r_ab,
                                        # R_a = Σ_b c_ab r rᵀ (inverse_moment.tex, eq. condition)
end

# Storage kept between steps so that the per-particle arrays are reused.
# Pass the same StochCache to every stochastic_step! of one simulation; one cache per simulation.
mutable struct StochCache
    stars :: Vector{Star}
end
StochCache() = StochCache(Star[])

# With `cache`, the stars are written into cache.stars, reusing the arrays of the
# previous step's stars.
function build_stars(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                     eos::AbstractEOS, V::Vector{Float64}; cache = nothing)
    N    = length(particles)
    h    = params.h;  L = params.L;  d = params.phys_dim
    check_stochastic_dims(params)
    e1 = SMatrix{2,2,Float64}(1, 0, 0, -1) / sqrt(2);  e2 = SMatrix{2,2,Float64}(0, 1, 1, 0) / sqrt(2)
    stars = cache === nothing ? Vector{Star}(undef, N) : resize!(cache.stars, N)
    nl = neighbor_list!(cl, particles, h)
    threaded_chunks(N) do range
    nb = Tuple{Int,SVector{2,Float64},Float64}[];  ψ0b = Float64[];  Vbb = Float64[]
    for a in range
        pa = particles[a]
        empty!(nb)
        begin                                            # neighbours in cell-scan order
            for b in neighbors(nl, a)
                b == a && continue
                r = minimum_image(particles[b].x - pa.x, L);  rm = norm(r)
                rm < 2h && push!(nb, (b, r, rm))
            end
        end
        # comoving forms at the particle's velocity (docs/stochastic.tex, "A moving fluid")
        va = SSCHEME ? sscheme_target_velocity(pa, eos) : pa.dotx   # ẋ_a (sscheme: the target's velocity, as star_coef)
        ℓ, ℓi, P, Pℓ = comoving_structures(va, cs2(eos), 2)
        # the bond weights of the SPH kernel at the lab distance (inverse_moment.tex)
        ψs, n_a = bond_weights_lab(nb, particles, common_bond_weight(params); ψ0 = ψ0b, Vb = Vbb)
        Ta = target_trel(pa, eos);  sr = srel(eos, pa.e)   # T_rel = γ T, pa.T = ∂H/∂s (sscheme: the target's T_rel)
        κs, κb = eta_by_s(eos) * sr, zeta_eff_by_s(eos, d) * sr   # ζ_eff with the floor ε η (sph.jl, dimension.tex)
        ℓh = sqrt_comoving(va)
        # E_k = ℓ^{1/2} e_k ℓ^{1/2} − (P:ℓ^{1/2}e_kℓ^{1/2}/⟨Pℓ⟩) ℓ;  bulk ℓ/⟨Pℓ⟩
        E(ek) = (X = ℓh * ek * ℓh;  X - (sum(P .* X) / Pℓ) * ℓ)
        B = (sqrt(2κs) * E(e1), sqrt(2κs) * E(e2), sqrt(κb) * ℓ / Pℓ)
        nk = n_a > 0 ? count(>(0), ψs) : 0                 # arrays at their exact size
        if cache !== nothing && isassigned(stars, a)
            o = stars[a]
            bs = resize!(o.nb, nk);  rs = resize!(o.r, nk);  ws = resize!(o.w, nk)
            cs = resize!(o.c, nk);  As = resize!(o.A, nk);  Us = resize!(o.U, nk)
        else
            bs = Vector{Int}(undef, nk);  rs = Vector{SVector{2,Float64}}(undef, nk)
            ws = Vector{Float64}(undef, nk);  cs = Vector{Float64}(undef, nk)
            As = Vector{SVector{2,Float64}}(undef, nk);  Us = Vector{SVector{2,Float64}}(undef, nk)
        end
        j = 0
        for (k, (b, r, rr)) in enumerate(nb)
            (ψs[k] > 0 && n_a > 0) || continue
            wab = V[b] * ψs[k] / n_a;  j += 1
            bs[j] = b;  rs[j] = r;  ws[j] = wab
            cs[j] = 2κs * Ta * V[a] * wab * resid_factor(params, ℓ, r)
            As[j] = V[a] * wab * r
        end
        # faces A_ab = V_a w_ab 𝓜_a⁻¹ r_ab (Σ_b A_ab r_abᵀ = V_a 𝟙);  u_ab = c_ab R_a⁻¹ r_ab
        moment_faces!(As, Us, ws, cs, rs)
        stars[a] = Star(bs, rs, ws, cs, V[a], Ta, κs, κb, As, B, ℓ, ℓh, ℓi, Us)
    end
    end
    return stars
end

@noinline function moment_faces!(As, Us, ws, cs, rs)
    Mm = zero(SMatrix{2,2,Float64});  R = zero(SMatrix{2,2,Float64})
    for k in eachindex(rs)
        Mm += ws[k] * (rs[k] * rs[k]');  R += cs[k] * (rs[k] * rs[k]')
    end
    Mi = inv(Mm);  Ri = inv(R)
    for k in eachindex(rs)
        As[k] = Mi * As[k];  Us[k] = cs[k] * (Ri * rs[k])
    end
end

# ── Checkerboard ─────────────────────────────────────────────────────────────
# Colors along one direction of n cells: cells of the same color are at least two
# cells (≥ 4h) apart, including across the periodic wrap.
function cell_colors_1d(n::Int)
    n < 3 && return collect(0:n-1)
    m = 3 * (n ÷ 3)
    return [i <= m ? (i - 1) % 3 : 3 + (i - m - 1) for i in 1:n]
end

# Members of each cell after a shift of the grid by u (positions are fixed; only the
# partition into cells moves).  Same cell sizes, so cells stay ≥ 2h and tile the box.
function shifted_cells(particles::Vector{Particle}, cl::CellList2D, u::SVector{2,Float64})
    cells = [Int[] for _ in 1:cl.ncx, _ in 1:cl.ncy]
    for (a, pa) in enumerate(particles)
        ix, iy = cell_index(mod.(pa.x + u, cl.L), cl)
        push!(cells[ix, iy], a)
    end
    return cells
end

# Groups of cells (ix, iy) by color.
function color_groups(cl::CellList2D)
    cx = cell_colors_1d(cl.ncx);  cy = cell_colors_1d(cl.ncy)
    nxc = maximum(cx) + 1
    groups = Dict{Int,Vector{Tuple{Int,Int}}}()
    for iy in 1:cl.ncy, ix in 1:cl.ncx
        push!(get!(groups, cx[ix] + nxc * cy[iy], Tuple{Int,Int}[]), (ix, iy))
    end
    return [groups[k] for k in sort(collect(keys(groups)))]
end

# ── Moves ────────────────────────────────────────────────────────────────────
# Proposed kicks of the move of particle a: δ[k] to bond k (a gives up their sum).
function propose_stress!(δ, st::Star, dt, rng)
    amp = sqrt(2dt * st.T / st.V)
    ξ = amp * (randn(rng) * st.B[1] + randn(rng) * st.B[2] + randn(rng) * st.B[3])
    for k in eachindex(st.nb)
        δ[k] = ξ * st.A[k]                          # t^i_ab = ξ^{ik} A_ab,k
    end
end

# The residual move (inverse_moment.tex): the draws conditioned on zero virial,
#   y_ab − Y_a u_ab,   Y_a = Σ_b y_ab r_abᵀ,   u_ab = c_ab R_a⁻¹ r_ab.
#
# A separate @noinline function for bit-identity: the compiler's fusion of muladd depends on
# the surrounding code.
@noinline function propose_resid_moment!(δ, st::Star, dt, rng)
    Y = zero(SMatrix{2,2,Float64})
    for k in eachindex(st.nb)
        y = sqrt(2dt * st.c[k]) * (st.ℓh * SVector(randn(rng), randn(rng)))   # ⟨yyᵀ⟩ ∝ ℓ
        δ[k] = y;  Y += y * st.r[k]'
    end
    for k in eachindex(st.nb)
        δ[k] -= Y * st.U[k]
    end
end

propose_resid!(δ, st::Star, dt, rng) = propose_resid_moment!(δ, st, dt, rng)

# Particle entropy S_a(e, m) at volume V (the inverse map, pointwise.jl).
#
# −Inf for a state no fluid can have: the inversion fails, or γ > γ_max, i.e.
# |v|² > 1 − 1/γ_max² (sph.jl, SPH_GAMMA_MAX).  Such states have zero weight under e^𝕊, so a
# move that reaches one is rejected (docs/plan.tex, "Guarding against superluminal proposals").
particle_entropy(e, m, V, eos; guess = nothing) = particle_inverse(e, m, V, eos; guess)[1]

# The entropy that weights a state in the votes (the Liouville measure,
# equilibrium_positions.tex, Appendix A).  The measure is flat in s_a; at fixed e_a and V_a it
# induces e^{S_a}/T_a in m_a, so the votes use S_a − ln T_a.
#
# T_a = ∂e_a/∂s_a is the bare temperature (the third return of eos_from_energy).
# −Inf as particle_entropy.
function particle_entropy_lnT(e, m, V, eos; guess = nothing)
    s, _, T = particle_inverse(e, m, V, eos; guess)
    return (isfinite(s) && T > 0) ? s - log(T) : -Inf
end
# SPH_RENORM is recorded here.  sph.jl reads it: split (the default), sscheme, or 0 (the tree level).
(SPLIT || SSCHEME) ? setting("SPH_RENORM", :split) : setting("SPH_RENORM", false)
# The vote entropy is chosen at load time, so that at the tree level the code is unchanged
# (bit-identical regression).
if SSCHEME
    vote_entropy(e, m, V, eos; guess = nothing) = sscheme_vote_entropy(e, m, V, eos)
elseif SPLIT
    vote_entropy(e, m, V, eos; guess = nothing) = split_vote_entropy(e, m, V, eos, true; guess)
else
    vote_entropy(e, m, V, eos; guess = nothing) = particle_entropy_lnT(e, m, V, eos; guess)
end
# SPH_RENORM=split: the state of each particle at the start of a stochastic step (its label and
# the target's rest-frame energy density), the starting point of the inverse solves of its votes
# and coefficients.  A guess changes the Newton iterations, not the result.
#
# vg(a) is nothing otherwise, and the target's inverse ignores it.
const VOTE_GUESS = Tuple{Float64,SVector{2,Float64}}[]
@inline vg(a) = SPLIT ? @inbounds(VOTE_GUESS[a]) : nothing

# Events guarded against, summed over the run (reset with reset_guards!()).
const GUARD_REJECTED = Threads.Atomic{Int}(0)    # moves rejected for an unphysical state
const GUARD_LEAPFROG = Threads.Atomic{Int}(0)    # failed leapfrog steps retried as two half steps (hmc.jl)
reset_guards!() = (GUARD_REJECTED[] = 0;  GUARD_LEAPFROG[] = 0)
guards() = (rejected = GUARD_REJECTED[], leapfrog_retries = GUARD_LEAPFROG[])
# Restore the counters from a checkpoint (the NamedTuple returned by guards()).
# Older checkpoints have no leapfrog_retries.
restore_guards!(g) = (GUARD_REJECTED[] = g.rejected;  GUARD_LEAPFROG[] = get(g, :leapfrog_retries, 0);  g)

# The single vote of the star: accept the move with min(1, exp ΔS), ΔS = Σ_c ΔS_c over the
# particle and its bonds, all or nothing.
function ask_star!(m, Sc, e, V, a, st::Star, δ, eos, rng, Snew)
    n  = length(st.nb)
    δa = -sum(@view δ[1:n])
    ma′ = m[a] + δa
    Sa = vote_entropy(e[a], ma′, V[a], eos; guess = vg(a))
    if !isfinite(Sa)
        Threads.atomic_add!(GUARD_REJECTED, 1);  return false
    end
    ΔStot = Sa - Sc[a]
    for k in 1:n
        b = st.nb[k]
        Snew[k] = vote_entropy(e[b], m[b] + δ[k], V[b], eos; guess = vg(b))
        if !isfinite(Snew[k])
            Threads.atomic_add!(GUARD_REJECTED, 1);  return false
        end
        ΔStot += Snew[k] - Sc[b]
    end
    logP = min(0.0, ΔStot)
    if logP >= 0 || log(rand(rng)) < logP
        for k in 1:n
            b = st.nb[k];  m[b] += δ[k];  Sc[b] = Snew[k]
        end
        m[a] = ma′;  Sc[a] = Sa
        return true
    end
    return false
end

# One checkerboard sweep of the given kind (:stress or :resid).  Returns the number accepted.
function color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng)
    nacc = Threads.Atomic{Int}(0)
    for g in shuffle(rng, collect(eachindex(groups)))
        cg = groups[g]
        order = [shuffle(rng, cells[ix, iy]) for (ix, iy) in cg]
        Threads.@threads for ci in eachindex(cg)
            δ = sizehint!(SVector{2,Float64}[], 128);  Snew = sizehint!(Float64[], 128)
            for a in order[ci]
                st = stars[a];  n = length(st.nb);  n == 0 && continue
                resize!(δ, n);  resize!(Snew, n)
                if kind === :stress
                    propose_stress!(δ, st, dt, rngs[a])
                else
                    propose_resid!(δ, st, dt, rngs[a])
                end
                ok = ask_star!(m, Sc, e, V, a, st, δ, eos, rngs[a], Snew)
                ok && Threads.atomic_add!(nacc, 1)
            end
        end
    end
    return nacc[]
end

# ── The Hastings correction (SPH_HASTINGS=1; inverse_moment.tex, stage 2) ──────────────
# The coefficients of star a are computed from the present state (e_a, m_a, V_a) of particle a,
# at each proposal and again at the proposed state.  They are those build_stars takes from the
# particle fields after solve_eos! (the particle's own state; ẋ_a = v_a).
#
# The geometry of the star (bonds, w_ab, faces) stays that of build_stars.
struct StarCoef
    T  :: Float64                                    # T_rel,a
    κs :: Float64                                    # η_a
    κb :: Float64                                    # ζ_eff,a
    ℓ  :: SMatrix{2,2,Float64,4}
    ℓh :: SMatrix{2,2,Float64,4}
    ℓi :: SMatrix{2,2,Float64,4}
    B  :: NTuple{3,SMatrix{2,2,Float64,4}}
end

# nothing for a state no fluid can have
function star_coef(e, m, V, eos, d; guess = nothing)
    SSCHEME && return sscheme_star_coef(e, m, V, eos, d)
    s, vi, Ti = particle_inverse(e, m, V, eos; guess);  isfinite(s) || return nothing
    S_in = s / V;  M = m / V
    st = particle_forward(s, S_in, M, e / V, eos; M_mag = norm(M), guess = (Ti, vi))   # the state at (s, m, V)
    return star_coef_at(st.e, st.v, st.T * st.γ, eos, d)
end
# The coefficients at the rest-frame energy density ε, the velocity v and the T_rel of the
# star's particle.
@inline function star_coef_at(ε, v, Ta, eos, d)
    sr = srel(eos, ε)
    ℓ, ℓi, P, Pℓ = comoving_structures(v, cs2(eos), 2)
    κs, κb = eta_by_s(eos) * sr, zeta_eff_by_s(eos, d) * sr   # ζ_eff with the floor ε η
    ℓh = sqrt_comoving(v)
    e1 = SMatrix{2,2,Float64}(1, 0, 0, -1) / sqrt(2);  e2 = SMatrix{2,2,Float64}(0, 1, 1, 0) / sqrt(2)
    E(ek) = (X = ℓh * ek * ℓh;  X - (sum(P .* X) / Pℓ) * ℓ)
    return StarCoef(Ta, κs, κb, ℓ, ℓh, ℓi, (sqrt(2κs) * E(e1), sqrt(2κs) * E(e2), sqrt(κb) * ℓ / Pℓ))
end

# SPH_RENORM=sscheme: the coefficients at the target state of (e, m, V), as build_stars
# (sscheme_plan.tex, item C: the target's ε, v and T_rel).
function sscheme_star_coef(e, m, V, eos, d)
    c = SSchemeConsts(eos);  mm = norm(m)
    Tr, uγ = sscheme_target_energy(e, mm, V, c)
    st = isfinite(Tr) ? sscheme_state(Tr, uγ, V, c) : SS_NOSTATE
    st.ok || return nothing
    v = mm > 0 ? (st.v / mm) * m : zero(m)
    return star_coef_at(st.erest, v, st.Trel, eos, d)
end

symvec(X) = SVector(X[1, 1], X[1, 2], X[2, 2])
stress_J(c::StarCoef, V, dt) = (amp = sqrt(2dt * c.T / V);
    hcat(symvec(amp * c.B[1]), symvec(amp * c.B[2]), symvec(amp * c.B[3])))
resid_c(c::StarCoef, st::Star, k, V, params) = 2c.κs * c.T * V * st.w[k] * resid_factor(params, c.ℓ, st.r[k])

# The stress move: δ[k] = ξ A_ab with ξ = Σ_k z_k J_k.
# Returns ln q(m′→m)/q(m→m′) (inverse_moment.tex, eq. qstress2).
function propose_stress_hastings!(δ, st::Star, c::StarCoef, e, ma, V, dt, eos, d, rng; guess = nothing)
    z = SVector(randn(rng), randn(rng), randn(rng))
    J = stress_J(c, V, dt);  xv = J * z
    ξ = SMatrix{2,2,Float64}(xv[1], xv[2], xv[2], xv[3])
    for k in eachindex(st.nb);  δ[k] = ξ * st.A[k];  end
    c′ = star_coef(e, ma - sum(@view δ[1:length(st.nb)]), V, eos, d; guess);  c′ === nothing && return -Inf
    J′ = stress_J(c′, V, dt);  z′ = J′ \ (-xv)
    return -0.5 * dot(z′, z′) + 0.5 * dot(z, z) + log(abs(det(J))) - log(abs(det(J′)))
end

# The residual move, conditioned on zero virial with the coefficients of the present state.
# Returns the log ratio (inverse_moment.tex, eq. qresid).
function propose_resid_hastings!(δ, st::Star, c::StarCoef, e, ma, V, dt, eos, d, params, rng; guess = nothing)
    n = length(st.nb)
    Y = zero(SMatrix{2,2,Float64});  R = zero(SMatrix{2,2,Float64})
    for k in 1:n
        ck = resid_c(c, st, k, V, params)
        y = sqrt(2dt * ck) * (c.ℓh * SVector(randn(rng), randn(rng)))
        δ[k] = y;  Y += y * st.r[k]';  R += ck * (st.r[k] * st.r[k]')
    end
    Ri = inv(R)
    for k in 1:n
        δ[k] -= resid_c(c, st, k, V, params) * (Y * (Ri * st.r[k]))
    end
    c′ = star_coef(e, ma - sum(@view δ[1:n]), V, eos, d; guess);  c′ === nothing && return -Inf
    R′ = zero(SMatrix{2,2,Float64});  lq = 0.0;  quad = 0.0
    for k in 1:n
        ck = resid_c(c, st, k, V, params);  ck′ = resid_c(c′, st, k, V, params)
        R′ += ck′ * (st.r[k] * st.r[k]')
        lq += log(ck / ck′)
        quad += dot(δ[k], (c′.ℓi / ck′ - c.ℓi / ck) * δ[k])
    end
    return lq + (n / 2 - 1) * log(det(c.ℓ) / det(c′.ℓ)) + log(det(R′) / det(R)) - quad / (4dt)
end

# The single vote with the Hastings factor: accept with min(1, e^{ΔS + lq}).
function ask_star_hastings!(m, Sc, e, V, a, st::Star, δ, lq, eos, rng, Snew)
    n  = length(st.nb)
    isfinite(lq) || (Threads.atomic_add!(GUARD_REJECTED, 1);  return false)
    ma′ = m[a] - sum(@view δ[1:n])
    Sa = vote_entropy(e[a], ma′, V[a], eos; guess = vg(a))
    isfinite(Sa) || (Threads.atomic_add!(GUARD_REJECTED, 1);  return false)
    ΔStot = Sa - Sc[a]
    for k in 1:n
        b = st.nb[k]
        Snew[k] = vote_entropy(e[b], m[b] + δ[k], V[b], eos; guess = vg(b))
        isfinite(Snew[k]) || (Threads.atomic_add!(GUARD_REJECTED, 1);  return false)
        ΔStot += Snew[k] - Sc[b]
    end
    logP = min(0.0, ΔStot + lq)
    if logP >= 0 || log(rand(rng)) < logP
        for k in 1:n
            b = st.nb[k];  m[b] += δ[k];  Sc[b] = Snew[k]
        end
        m[a] = ma′;  Sc[a] = Sa
        return true
    end
    return false
end

# One checkerboard sweep with the Hastings correction (as color_sweep!).
function color_sweep_hastings!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng, params)
    nacc = Threads.Atomic{Int}(0);  d = params.phys_dim
    lsum = Threads.Atomic{Float64}(0.0);  lsq = Threads.Atomic{Float64}(0.0);  lnum = Threads.Atomic{Int}(0)   # ln q′/q stats
    for g in shuffle(rng, collect(eachindex(groups)))
        cg = groups[g]
        order = [shuffle(rng, cells[ix, iy]) for (ix, iy) in cg]
        Threads.@threads for ci in eachindex(cg)
            δ = sizehint!(SVector{2,Float64}[], 128);  Snew = sizehint!(Float64[], 128)
            for a in order[ci]
                st = stars[a];  n = length(st.nb);  n == 0 && continue
                resize!(δ, n);  resize!(Snew, n)
                c = star_coef(e[a], m[a], V[a], eos, d; guess = vg(a));  c === nothing && continue
                lq = kind === :stress ? propose_stress_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, rngs[a]; guess = vg(a)) :
                                        propose_resid_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, params, rngs[a]; guess = vg(a))
                if isfinite(lq)
                    Threads.atomic_add!(lsum, lq);  Threads.atomic_add!(lsq, lq^2);  Threads.atomic_add!(lnum, 1)
                end
                ask_star_hastings!(m, Sc, e, V, a, st, δ, lq, eos, rngs[a], Snew) && Threads.atomic_add!(nacc, 1)
            end
        end
    end
    return nacc[], lsum[], lsq[], lnum[]
end

# ── The step ─────────────────────────────────────────────────────────────────
# Frozen data of a step: particle volumes, particle energies, stars.
function stochastic_frozen(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                         eos::AbstractEOS; cache = nothing)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)        # ẋ_a, the velocity of the comoving metric
    N = length(particles);  V = Vector{Float64}(undef, N);  e = Vector{Float64}(undef, N)
    Threads.@threads for a in 1:N
        pa = particles[a];  V[a] = pa.V;  e[a] = particle_energy(pa, eos)
    end
    return (V = V, e = e, stars = build_stars(particles, params, cl, eos, V; cache))
end

# The settings of the stochastic step:
#   hastings  the Hastings correction (inverse_moment.tex, stage 2): the coefficients of each
#             star from the present state of its particle at every proposal, and the ratio of
#             the reverse and forward proposal densities in the vote.
#             Default from SPH_HASTINGS (on); SPH_HASTINGS=0 turns it off.
struct StochOptions
    hastings :: Bool
end
StochOptions(params::SimParams; hastings::Bool = setting("SPH_HASTINGS", true)) = StochOptions(hastings)

# The step with the settings as keywords (tests, examples): builds the StochOptions.
stochastic_step!(particles::Vector{Particle}, params::SimParams, cl::CellList2D, dt::Float64, eos::AbstractEOS; kw...) =
    stochastic_step!(particles, params, cl, dt, eos, StochOptions(params); kw...)

# One stochastic viscous step at fixed positions: a stress sweep and a residual sweep, in an
# order that alternates with `flip`; each move is accepted by the single vote of its star.
# The checkerboard grid is shifted at random for each sweep.
#
# Pass `frozen = stochastic_frozen(...)` to keep V_a, e_a and the stars fixed across steps
# (a frozen lattice); otherwise they are recomputed from the current state.
#
# Pass the same `cache = StochCache()` to every step of a simulation to reuse its arrays
# (faster, same results).
#
# Returns the acceptance fractions and the guard and Hastings statistics (below).
function stochastic_step!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                          dt::Float64, eos::AbstractEOS, opts::StochOptions;
                          flip::Bool = false, rng::AbstractRNG = Random.default_rng(),
                          rngs::Vector{Xoshiro} = PHYS_RNG, groups = color_groups(cl),
                          frozen = nothing, cache = nothing)
    (; hastings) = opts
    @assert length(rngs) == length(particles) "rng vector length ≠ particle count — call seed_phys!(N, seed)"
    check_stochastic_dims(params)
    fz = frozen === nothing ? stochastic_frozen(particles, params, cl, eos; cache) : frozen
    V, e, stars = fz.V, fz.e, fz.stars
    N  = length(particles)
    m  = [pa.m for pa in particles]
    # Sc holds the vote entropy (S_a − ln T_a at the tree level) during the sweeps;
    # the particle entropies are the true S_a, recomputed at the end.
    #
    # SPH_RENORM=split: the guess is the label and the target's rest-frame energy density (renorm_split.jl).
    if SPLIT
        resize!(VOTE_GUESS, N)
        for a in 1:N;  pa = particles[a];  VOTE_GUESS[a] = (pa.s, SVector(pa.e, 0.0));  end
    end
    Sc = Vector{Float64}(undef, N)
    Threads.@threads for a in 1:N
        Sc[a] = vote_entropy(e[a], m[a], V[a], eos; guess = vg(a))
    end
    kinds = flip ? (:resid, :stress) : (:stress, :resid)
    acc = Dict(:stress => NaN, :resid => NaN)
    grj = Dict(:stress => 0.0, :resid => 0.0)          # fraction of proposals rejected by the guard
    lqs = Dict(:stress => (NaN, NaN), :resid => (NaN, NaN))   # with hastings: mean and rms of ln q′/q
    for kind in kinds
        # random shift of the grid for each sweep
        u = SVector(rand(rng) * cl.cell_size[1], rand(rng) * cl.cell_size[2])
        cells = shifted_cells(particles, cl, u)
        g0 = GUARD_REJECTED[]
        if hastings
            nh, ls, lq2, ln = color_sweep_hastings!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng, params)
            acc[kind] = nh / N
            ln > 0 && (lqs[kind] = (ls / ln, sqrt(lq2 / ln)))
        else
            acc[kind] = color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng) / N
        end
        grj[kind] = (GUARD_REJECTED[] - g0) / N
    end
    for (a, pa) in enumerate(particles)
        pa.m = m[a];  pa.s = particle_entropy(e[a], m[a], V[a], eos; guess = vg(a))
    end
    # For the stress and the residual sweep: the acceptances, and the fractions of proposals
    # rejected by the guard (a state no fluid can have; for the conformal EOS exactly |v| ≥ 1).
    #
    # With hastings also the mean and rms of the log ratio ln q′/q of the proposals of each sweep
    # (NaN without).
    return (stress = acc[:stress], resid = acc[:resid], gstress = grj[:stress], gresid = grj[:resid],
            lq_stress = lqs[:stress], lq_resid = lqs[:resid])
end
