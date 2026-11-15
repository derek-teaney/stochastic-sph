# stochastic.jl — stochastic viscous step in the particle representation (docs/stochastic.tex).
#
# During the step the positions, the particle volumes V_a = s_a/S_a and the particle
# energies e_a are fixed; only the momenta move.  The total particle entropy
#     𝕊[m] = Σ_a S_a(e_a, m_a),      ∂S_a/∂m_a = −β_a ,
# is a sum of single-particle terms.  With `gradient`, the gradient energy g_a of changerep.tex is
# frozen during the sweeps and refit in the back map (the gradient-corrected change of rep).
#
# Two sweeps, each over a checkerboard of cells:
#   stress sweep    particle a draws a random symmetric stress ξ_a with
#                   ⟨ξξ⟩ = 2Δt T_a κ_a / V_a and gives the traction ξ_a A_ab to each
#                   bond b, A_ab = V_a w_ab r_ab.  Mean: the resolved part of the bond force.
#   residual sweep  particle a draws a random vector y_ab on each bond,
#                   ⟨y yᵀ⟩ = 2Δt c_ab / λ, and gives y_ab − (Y_a/V_a) A_ab,
#                   Y_a = Σ_b y_ab r_abᵀ.  Mean: the residual part of the bond force.
#                   λ = 1 is detailed balance, λ > 1 a cold residual, λ = Inf the
#                   deterministic residual step.
# Each move is accepted with Π_b min(1, exp(λ ΔS_ab)), ΔS_ab = ΔS_b + ΔS_a/n_a (λ = 1 for
# the stress move): all or nothing.
#
# The coefficients (T_rel, η, ζ, w_ab, c_ab, ℓ, P) are frozen at the start of the step, at
# the particle velocities of that moment; in 2D they take the comoving forms of
# docs/stochastic.tex ("A moving fluid"), in 1D the rest-frame forms.  Moves in cells of the same color
# touch disjoint sets of particles and run in parallel; within a cell the moves are
# sequential, each seeing the current momenta.  The random numbers of the move of
# particle a come from rngs[a]; the sweep orders from the master rng.
#
# Loaded by load.jl (uses pointwise.jl and rng.jl).

# ℓ^{1/2} for ℓ = 𝟙 − v vᵀ (|v| < 1)
function sqrt_comoving(v::SVector{2,Float64})
    v2 = dot(v, v)
    v2 == 0 && return SMatrix{2,2,Float64}(I)
    return SMatrix{2,2,Float64}(I) - (1 - sqrt(1 - v2)) * (v * v') / v2
end

# ── Frozen star data ─────────────────────────────────────────────────────────
struct Star
    nb :: Vector{Int}                   # bonds b with w_ab > 0
    r  :: Vector{SVector{2,Float64}}    # r_ab = x_b − x_a (minimum image)
    w  :: Vector{Float64}               # w_ab = V_b ψ_ab / n_a,  Σ_b w_ab r_ab r_abᵀ = 𝟙 (2D)
    c  :: Vector{Float64}               # residual coefficient: K_a ⊃ ½ Σ_b c_ab |δ_ab|²
    V  :: Float64                       # V_a
    T  :: Float64                       # T_rel,a  (rest-frame temperature, not pa.T = T_rel/γ)
    κs :: Float64                       # 2D: η_a;            1D: κ_a
    κb :: Float64                       # 2D: ζ_eff,a;        1D: unused
    A  :: Vector{SVector{2,Float64}}    # face covector A_ab = V_a w_ab ℓ_a⁻¹ r_ab
    B  :: NTuple{3,SMatrix{2,2,Float64,4}}  # stress basis: ξ = √(2Δt T_rel/V) Σ_k z_k B_k
    ℓ  :: SMatrix{2,2,Float64,4}        # comoving metric ℓ_a = 𝟙 − v vᵀ
    ℓh :: SMatrix{2,2,Float64,4}        # ℓ_a^{1/2}
    ℓi :: SMatrix{2,2,Float64,4}        # ℓ_a⁻¹
    U  :: Vector{SVector{2,Float64}}    # :moment only (else empty): the conditioning vectors u_ab = c_ab R_a⁻¹ r_ab,
                                        # R_a = Σ_b c_ab r rᵀ (inverse_moment.tex, eq. condition)
end

# Storage kept between steps so that the per-particle arrays are reused (pass the same
# StochCache to every stochastic_step! of one simulation; one cache per simulation).
mutable struct StochCache
    stars :: Vector{Star}
    kst   :: Vector{Vector{Tuple{Int,SVector{2,Float64},Float64}}}
end
StochCache() = StochCache(Star[], Vector{Tuple{Int,SVector{2,Float64},Float64}}[])

# With `cache`, the stars are written into cache.stars, reusing the arrays of the
# previous step's stars.
function build_stars(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                     eos::AbstractEOS, V::Vector{Float64}; cache = nothing)
    N    = length(particles)
    h    = params.h;  L = params.L;  hψ = params.bond_h;  d = params.d
    oneD = params.kernel isa WendlandC2_1D
    v0   = zero(SVector{2,Float64})
    ℓ0, ℓi0, P0, Pℓ0 = comoving_structures(v0, cs2(eos), oneD ? d : 2)   # rest frame
    e1 = SMatrix{2,2,Float64}(1, 0, 0, -1) / sqrt(2);  e2 = SMatrix{2,2,Float64}(0, 1, 1, 0) / sqrt(2)
    stars = cache === nothing ? Vector{Star}(undef, N) : resize!(cache.stars, N)
    nl = neighbor_list!(cl, particles, h)
    threaded_chunks(N) do range
    nb = Tuple{Int,SVector{2,Float64},Float64}[];  ψ0b = Float64[];  Vbb = Float64[];  ψb = Float64[]
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
        # 2D: comoving forms at the particle's velocity (docs/stochastic.tex, "A moving
        # fluid"); 1D: rest frame.
        va = oneD ? v0 : pa.dotx                         # coarse-grained ẋ_a
        ℓ, ℓi, P, Pℓ = oneD ? (ℓ0, ℓi0, P0, Pℓ0) : comoving_structures(va, cs2(eos), 2)
        common = lab_weights(params)                     # one kernel (commonkernel.tex): lab weights
        moment = is_moment(params)                       # no Λ_a, inverse-moment faces (inverse_moment.tex)
        ψs, n_a = oneD ? bond_weights_1d(nb, particles, ℓi, hψ) :
                  moment ? bond_weights_lab(nb, particles, common_bond_weight(params); ψ0 = ψ0b, Vb = Vbb) :
                  common ? bond_weights_comoving(nb, particles, FLAT2, FLAT2, hψ; ψ0 = ψ0b, Vb = Vbb, ψ = ψb,
                                                 ψfun = common_bond_weight(params)) :
                         bond_weights_comoving(nb, particles, ℓ, ℓi, hψ; ψ0 = ψ0b, Vb = Vbb, ψ = ψb)
        Ta = pa.T * pa.γ;  sr = srel(eos, pa.e)          # T_rel = γ (e+p)/S;  pa.T = T_rel/γ = ∂H/∂s
        η  = eta_by_s(eos) * sr;  ζ = zeta_by_s(eos) * sr
        if oneD
            θx = ℓ[1,1] / Pℓ
            Sx = SMatrix{2,2,Float64}(1.0, 0.0, 0.0, 0.0) - θx * P
            κ1 = 2η * (tr(ℓ * Sx * ℓ * Sx) + (d - 2) * θx^2 / d^2) + ζ * θx^2
            κs, κb = κ1, 0.0
        else
            κs, κb = η, ζ + 2η * (1 / 2 - 1 / d)
        end
        ℓh = sqrt_comoving(va)
        if oneD
            B = (sqrt(κs) * SMatrix{2,2,Float64}(1, 0, 0, 0), zero(SMatrix{2,2,Float64}),
                 zero(SMatrix{2,2,Float64}))
        else
            # E_k = ℓ^{1/2} e_k ℓ^{1/2} − (P:ℓ^{1/2}e_kℓ^{1/2}/⟨Pℓ⟩) ℓ;  bulk ℓ/⟨Pℓ⟩
            E(ek) = (X = ℓh * ek * ℓh;  X - (sum(P .* X) / Pℓ) * ℓ)
            B = (sqrt(2κs) * E(e1), sqrt(2κs) * E(e2), sqrt(κb) * ℓ / Pℓ)
        end
        nk = n_a > 0 ? count(>(0), ψs) : 0                 # arrays at their exact size
        if cache !== nothing && isassigned(stars, a)
            o = stars[a]
            bs = resize!(o.nb, nk);  rs = resize!(o.r, nk);  ws = resize!(o.w, nk)
            cs = resize!(o.c, nk);  As = resize!(o.A, nk);  Us = resize!(o.U, moment ? nk : 0)
        else
            bs = Vector{Int}(undef, nk);  rs = Vector{SVector{2,Float64}}(undef, nk)
            ws = Vector{Float64}(undef, nk);  cs = Vector{Float64}(undef, nk)
            As = Vector{SVector{2,Float64}}(undef, nk);  Us = Vector{SVector{2,Float64}}(undef, moment ? nk : 0)
        end
        j = 0
        for (k, (b, r, rr)) in enumerate(nb)
            (ψs[k] > 0 && n_a > 0) || continue
            wab = V[b] * ψs[k] / n_a;  j += 1
            bs[j] = b;  rs[j] = r;  ws[j] = wab
            cs[j] = (oneD ? κs : 2κs) * Ta * V[a] * wab * (common ? resid_factor(params, ℓ, r) : 1.0)
            As[j] = common ? V[a] * wab * r : V[a] * wab * (ℓi * r)     # faces: Σ_b A_ab r_abᵀ = V_a 𝟙 either way
        end
        moment && moment_faces!(As, Us, ws, cs, rs)      # faces V_a w_ab 𝓜_a⁻¹ r_ab;  u_ab = c_ab R_a⁻¹ r_ab
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
function propose_stress!(δ, st::Star, dt, oneD, rng)
    amp = sqrt(2dt * st.T / st.V)
    ξ = oneD ? amp * randn(rng) * st.B[1] :
               amp * (randn(rng) * st.B[1] + randn(rng) * st.B[2] + randn(rng) * st.B[3])
    for k in eachindex(st.nb)
        δ[k] = ξ * st.A[k]                          # t^i_ab = ξ^{ik} A_ab,k
    end
end

# :moment (inverse_moment.tex): the draws conditioned on zero virial, y − Y_a u_ab, u_ab = c_ab R_a⁻¹ r_ab.  A separate function, so
# that the code of the other schemes is unchanged (bit-identical: the compiler's fusion of muladd depends on the surrounding code).
@noinline function propose_resid_moment!(δ, st::Star, dt, λ, rng)
    Y = zero(SMatrix{2,2,Float64})
    for k in eachindex(st.nb)
        y = sqrt(2dt * st.c[k] / λ) * (st.ℓh * SVector(randn(rng), randn(rng)))   # ⟨yyᵀ⟩ ∝ ℓ
        δ[k] = y;  Y += y * st.r[k]'
    end
    for k in eachindex(st.nb)
        δ[k] -= Y * st.U[k]
    end
end

function propose_resid!(δ, st::Star, dt, λ, oneD, rng)
    isempty(st.U) || return propose_resid_moment!(δ, st, dt, λ, rng)
    if oneD
        Y = 0.0
        for k in eachindex(st.nb)
            y = sqrt(2dt * st.c[k] / λ) * randn(rng)
            δ[k] = SVector(y, 0.0);  Y += y * st.r[k][1]
        end
        for k in eachindex(st.nb)
            δ[k] -= SVector(Y * st.w[k] * st.r[k][1], 0.0)
        end
    else
        Y = zero(SMatrix{2,2,Float64})
        for k in eachindex(st.nb)
            y = sqrt(2dt * st.c[k] / λ) * (st.ℓh * SVector(randn(rng), randn(rng)))   # ⟨yyᵀ⟩ ∝ ℓ
            δ[k] = y;  Y += y * st.r[k]'
        end
        for k in eachindex(st.nb)
            δ[k] -= (Y * st.A[k]) / st.V              # y − (Y/V_a) A_ab
        end
    end
end

# Particle entropy S_a(e, m) at volume V; −Inf for a state no fluid can have (the
# inversion fails, or |v| ≥ 1).  Such states have zero weight under e^𝕊: a move that
# reaches one is rejected (docs/plan.tex, "Guarding against superluminal proposals").
function particle_entropy(e, m, V, eos)
    ε = e / V;  M = m / V
    (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return -Inf      # |M| < ε is necessary (p ≤ e)
    r = try
        eos_from_energy(ε, M, eos)
    catch
        return -Inf
    end
    S, v, _ = r
    return (isfinite(S) && dot(v, v) < 1) ? V * S : -Inf
end

# Conformal EOS: the closed-form inversion cannot throw once |M| < ε (all square roots
# have positive arguments), so the try block, about 17% of a call, is not needed.
# Same arithmetic as the general method: bit-identical results.
function particle_entropy(e, m, V, eos::ConformalEOS)
    ε = e / V;  M = m / V
    (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return -Inf
    S, v, _ = eos_from_energy(ε, M, eos)
    return (isfinite(S) && dot(v, v) < 1) ? V * S : -Inf
end

# The entropy that weights a state in the votes.  measure = :einstein: S_a itself (flat in e_a, m_a).  measure = :liouville
# (equilibrium_positions.tex, Appendix A): the measure flat in s_a, which at fixed e_a and V_a induces e^{S_a}/T_a in m_a, so the
# votes use S_a − ln T_a, T_a = ∂e_a/∂s_a the bare temperature (the third return of eos_from_energy).  −Inf as particle_entropy.
function particle_entropy_lnT(e, m, V, eos)
    ε = e / V;  M = m / V
    (isfinite(ε) && ε > 0 && dot(M, M) < ε^2) || return -Inf
    r = try
        eos_from_energy(ε, M, eos)
    catch
        return -Inf
    end
    S, v, T = r
    return (isfinite(S) && dot(v, v) < 1 && T > 0) ? V * S - log(T) : -Inf
end
vote_entropy(e, m, V, eos, liouville::Bool) = liouville ? particle_entropy_lnT(e, m, V, eos) : particle_entropy(e, m, V, eos)

# Events guarded against, summed over the run (reset with reset_guards!()).
const GUARD_REJECTED = Threads.Atomic{Int}(0)    # moves rejected for an unphysical state
const GUARD_EVENSPLIT = Threads.Atomic{Int}(0)   # bond split fell back to the even split
const GUARD_BACKMAP = Threads.Atomic{Int}(0)     # gradient back map fell back to the plain map
const GUARD_SUBDIV = Threads.Atomic{Int}(0)      # deterministic residual step subdivided
reset_guards!() = foreach(c -> (c[] = 0), (GUARD_REJECTED, GUARD_EVENSPLIT, GUARD_BACKMAP, GUARD_SUBDIV))
guards() = (rejected = GUARD_REJECTED[], evensplit = GUARD_EVENSPLIT[], backmap = GUARD_BACKMAP[],
            subdiv = GUARD_SUBDIV[])
# restore the counters from a checkpoint (the NamedTuple returned by guards())
restore_guards!(g) = (GUARD_REJECTED[] = g.rejected;  GUARD_EVENSPLIT[] = g.evensplit;
                      GUARD_BACKMAP[] = g.backmap;  GUARD_SUBDIV[] = g.subdiv;  g)

# Ask the bonds: accept with Π_b min(1, exp(λ ΔS_ab)), all or nothing.  The bond
# factor is ΔS_ab = ΔS_b + σ_ab + R/n_a, with σ_ab the share of ΔS_a due to the
# traction t_ab = δ[k] (vote = :bonds, docs/stochastic.tex):
#     σ_ab = ½[S_a(m_a − t_ab) − S_a(m_a)] − ½[S_a(m_a′ + t_ab) − S_a(m_a′)],
# m_a′ = m_a + δm_a, and R = ΔS_a − Σ_b σ_ab.  Each σ_ab flips sign exactly under the
# reverse move, and σ_ab ≈ β_a·t_ab, so ΔS_ab ≈ −Δβ_ab·t_ab is the bond's own entropy
# production (invariant under a uniform shift of β).  vote = :even uses σ_ab = 0
# (ΔS_a split evenly); vote = :star a single vote, min(1, exp(λ Σ_c ΔS_c)).  All exact.
function ask_star!(m, Sc, e, V, a, st::Star, δ, λ, eos, rng, Snew, vote::Symbol = :star, liouville::Bool = false)
    n  = length(st.nb)
    δa = -sum(@view δ[1:n])
    ma′ = m[a] + δa
    Sa = vote_entropy(e[a], ma′, V[a], eos, liouville)
    if !isfinite(Sa)
        Threads.atomic_add!(GUARD_REJECTED, 1);  return false
    end
    ΔSa = Sa - Sc[a]
    ΔStot = ΔSa
    for k in 1:n
        b = st.nb[k]
        Snew[k] = vote_entropy(e[b], m[b] + δ[k], V[b], eos, liouville)
        if !isfinite(Snew[k])
            Threads.atomic_add!(GUARD_REJECTED, 1);  return false
        end
        ΔStot += Snew[k] - Sc[b]
    end
    logP = 0.0
    if vote === :star
        logP = min(0.0, λ * ΔStot)
    elseif vote === :even
        for k in 1:n
            logP += min(0.0, λ * (Snew[k] - Sc[st.nb[k]] + ΔSa / n))
        end
    else
        σ = zeros(n)
        for k in 1:n
            σ[k] = 0.5 * (vote_entropy(e[a], m[a] - δ[k], V[a], eos, liouville) - Sc[a]) -
                   0.5 * (vote_entropy(e[a], ma′ + δ[k], V[a], eos, liouville) - Sa)
        end
        # an unphysical intermediate state: use the even split for this move (the same
        # points are evaluated for the reverse move, so detailed balance is kept)
        if !all(isfinite, σ)
            Threads.atomic_add!(GUARD_EVENSPLIT, 1);  σ .= 0.0
        end
        R = ΔSa - sum(σ)
        for k in 1:n
            logP += min(0.0, λ * (Snew[k] - Sc[st.nb[k]] + σ[k] + R / n))
        end
    end
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
function color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, λ, oneD, eos, rngs, rng, vote, liouville::Bool = false)
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
                    propose_stress!(δ, st, dt, oneD, rngs[a])
                    ok = ask_star!(m, Sc, e, V, a, st, δ, 1.0, eos, rngs[a], Snew, vote, liouville)
                else
                    propose_resid!(δ, st, dt, λ, oneD, rngs[a])
                    ok = ask_star!(m, Sc, e, V, a, st, δ, λ, eos, rngs[a], Snew, vote, liouville)
                end
                ok && Threads.atomic_add!(nacc, 1)
            end
        end
    end
    return nacc[]
end

# λ = Inf: the deterministic residual step, Δm from the residual tractions
# −Δt c_ab δ_ab, δ_ab = Δβ_ab − G_a r_ab, at fixed e_a; exact back map.
function resid_deterministic!(m, Sc, e, V, stars, dt, oneD, eos)
    N = length(m)
    β = map(1:N) do a
        S, v, T = eos_from_energy(e[a] / V[a], m[a] / V[a], eos);  v / T
    end
    Δm = zeros(SVector{2,Float64}, N)
    for a in 1:N
        st = stars[a]
        G = zero(SMatrix{2,2,Float64})
        for k in eachindex(st.nb)
            G += st.w[k] * ((β[st.nb[k]] - β[a]) * st.r[k]')
        end
        G = G * st.ℓi                                  # G_a = Σ w Δβ rᵀ ℓ⁻¹
        isempty(st.U) || (G = moment_fit(β, a, st))    # :moment: the fit X_a with the weights c_ab

        for k in eachindex(st.nb)
            δab = β[st.nb[k]] - β[a] - G * st.r[k]
            oneD && (δab = SVector(δab[1], 0.0))
            t = -dt * st.c[k] * (st.ℓ * δab)
            Δm[st.nb[k]] += t;  Δm[a] -= t
        end
    end
    # guard: if any particle would become unphysical, do the step as two half steps
    Snew = [particle_entropy(e[a], m[a] + Δm[a], V[a], eos) for a in 1:N]
    if !all(isfinite, Snew)
        Threads.atomic_add!(GUARD_SUBDIV, 1)
        dt < 1e-6 && error("resid_deterministic!: unphysical state even for dt = $dt")
        resid_deterministic!(m, Sc, e, V, stars, dt / 2, oneD, eos)
        resid_deterministic!(m, Sc, e, V, stars, dt / 2, oneD, eos)
        return
    end
    for a in 1:N
        m[a] += Δm[a];  Sc[a] = Snew[a]
    end
end

@noinline moment_fit(β, a, st::Star) =
    sum((β[st.nb[k]] - β[a]) * st.U[k]' for k in eachindex(st.nb); init = zero(SMatrix{2,2,Float64}))

# ── The Hastings correction (dissipation = :moment, SPH_HASTINGS=1; inverse_moment.tex, stage 2) ──────────────
# The coefficients of star a are computed from the present state (e_a, m_a, V_a) of particle a, as build_stars does from the
# particle fields after solve_eos! (energy Hamiltonian, number volume: the particle's own state; ẋ_a = v_a), at each proposal
# and again at the proposed state.  The geometry of the star (bonds, w_ab, faces) stays that of build_stars.
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
function star_coef(e, m, V, eos, d)
    s = particle_entropy(e, m, V, eos);  isfinite(s) || return nothing
    S_in = s / V;  M = m / V;  Mm = norm(M)
    ε = eos_newton(S_in, Mm, e / V, eos)
    w = ε + pressure(eos, ε);  G, _ = eos_wgamma(w, Mm);  γ = G / w;  v = M * (w / G^2)
    Ta = (w / S_in) * γ;  sr = srel(eos, ε);  η = eta_by_s(eos) * sr;  ζ = zeta_by_s(eos) * sr
    ℓ, ℓi, P, Pℓ = comoving_structures(v, cs2(eos), 2)
    κs, κb = η, ζ + 2η * (1 / 2 - 1 / d)
    ℓh = sqrt_comoving(v)
    e1 = SMatrix{2,2,Float64}(1, 0, 0, -1) / sqrt(2);  e2 = SMatrix{2,2,Float64}(0, 1, 1, 0) / sqrt(2)
    E(ek) = (X = ℓh * ek * ℓh;  X - (sum(P .* X) / Pℓ) * ℓ)
    return StarCoef(Ta, κs, κb, ℓ, ℓh, ℓi, (sqrt(2κs) * E(e1), sqrt(2κs) * E(e2), sqrt(κb) * ℓ / Pℓ))
end

symvec(X) = SVector(X[1, 1], X[1, 2], X[2, 2])
stress_J(c::StarCoef, V, dt) = (amp = sqrt(2dt * c.T / V);
    hcat(symvec(amp * c.B[1]), symvec(amp * c.B[2]), symvec(amp * c.B[3])))
resid_c(c::StarCoef, st::Star, k, V, params) = 2c.κs * c.T * V * st.w[k] * resid_factor(params, c.ℓ, st.r[k])

# The stress move: δ[k] = ξ A_ab with ξ = Σ_k z_k J_k; returns ln q(m′→m)/q(m→m′) (inverse_moment.tex, eq. qstress2).
function propose_stress_hastings!(δ, st::Star, c::StarCoef, e, ma, V, dt, eos, d, rng)
    z = SVector(randn(rng), randn(rng), randn(rng))
    J = stress_J(c, V, dt);  xv = J * z
    ξ = SMatrix{2,2,Float64}(xv[1], xv[2], xv[2], xv[3])
    for k in eachindex(st.nb);  δ[k] = ξ * st.A[k];  end
    c′ = star_coef(e, ma - sum(@view δ[1:length(st.nb)]), V, eos, d);  c′ === nothing && return -Inf
    J′ = stress_J(c′, V, dt);  z′ = J′ \ (-xv)
    return -0.5 * dot(z′, z′) + 0.5 * dot(z, z) + log(abs(det(J))) - log(abs(det(J′)))
end

# The residual move, conditioned on zero virial with the coefficients of the present state; returns the log ratio
# (inverse_moment.tex, eq. qresid).
function propose_resid_hastings!(δ, st::Star, c::StarCoef, e, ma, V, dt, eos, d, params, rng)
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
    c′ = star_coef(e, ma - sum(@view δ[1:n]), V, eos, d);  c′ === nothing && return -Inf
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
function ask_star_hastings!(m, Sc, e, V, a, st::Star, δ, lq, eos, rng, Snew, liouville::Bool)
    n  = length(st.nb)
    isfinite(lq) || (Threads.atomic_add!(GUARD_REJECTED, 1);  return false)
    ma′ = m[a] - sum(@view δ[1:n])
    Sa = vote_entropy(e[a], ma′, V[a], eos, liouville)
    isfinite(Sa) || (Threads.atomic_add!(GUARD_REJECTED, 1);  return false)
    ΔStot = Sa - Sc[a]
    for k in 1:n
        b = st.nb[k]
        Snew[k] = vote_entropy(e[b], m[b] + δ[k], V[b], eos, liouville)
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
function color_sweep_hastings!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng, liouville::Bool, params)
    nacc = Threads.Atomic{Int}(0);  d = params.d
    for g in shuffle(rng, collect(eachindex(groups)))
        cg = groups[g]
        order = [shuffle(rng, cells[ix, iy]) for (ix, iy) in cg]
        Threads.@threads for ci in eachindex(cg)
            δ = sizehint!(SVector{2,Float64}[], 128);  Snew = sizehint!(Float64[], 128)
            for a in order[ci]
                st = stars[a];  n = length(st.nb);  n == 0 && continue
                resize!(δ, n);  resize!(Snew, n)
                c = star_coef(e[a], m[a], V[a], eos, d);  c === nothing && continue
                lq = kind === :stress ? propose_stress_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, rngs[a]) :
                                        propose_resid_hastings!(δ, st, c, e[a], m[a], V[a], dt, eos, d, params, rngs[a])
                ask_star_hastings!(m, Sc, e, V, a, st, δ, lq, eos, rngs[a], Snew, liouville) && Threads.atomic_add!(nacc, 1)
            end
        end
    end
    return nacc[]
end

# ── Gradient-corrected change of representation ─────────────────────────────
# (changerep.tex, Appendix A): the gradient energy g_a of the long-wavelength momentum
# profile is frozen during the sweeps, so e_a = e′_a + g_a is fixed there; the back map
# refits it, s′ = S_a(e′_a + g′_a, m′_a).  Separate functions, so that the threaded loops
# see concretely typed arrays.
function gradient_freeze(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                         eos::AbstractEOS, cache)
    N   = length(particles)
    kst = cache === nothing ? Vector{Vector{Tuple{Int,SVector{2,Float64},Float64}}}(undef, N) :
                              resize!(cache.kst, N)
    Cs  = Vector{SMatrix{2,2,Float64,4}}(undef, N)
    s0  = [pa.s for pa in particles]
    μ0  = [pa.m / pa.s for pa in particles]
    g0  = Vector{Float64}(undef, N)
    nl  = neighbor_list!(cl, particles, params.h)
    reuse = cache !== nothing
    threaded_chunks(N) do range
        buf = Tuple{Int,SVector{2,Float64},Float64}[]
        for a in range
            kst[a] = reuse && isassigned(kst, a) ? kernel_star!(kst[a], a, particles, params, cl; nl) :
                                                   copy(kernel_star!(buf, a, particles, params, cl; nl))
            Cs[a]  = inverse_susceptibility(particles[a], eos)
        end
    end
    Threads.@threads for a in 1:N
        g0[a] = gradient_energy(s0[a], Cs[a], star_fit(kst[a], μ0)...)
    end
    return (kst = kst, Cs = Cs, s0 = s0, g0 = g0)
end

function gradient_backmap!(Sc::Vector{Float64}, gr, m::Vector{SVector{2,Float64}},
                           e::Vector{Float64}, V::Vector{Float64}, eos::AbstractEOS)
    kst, Cs, s0, g0 = gr.kst, gr.Cs, gr.s0, gr.g0
    N  = length(Sc)
    μ1 = [m[a] / s0[a] for a in 1:N]                       # s still old, as in the deterministic step
    Threads.@threads for a in 1:N
        g1 = gradient_energy(s0[a], Cs[a], star_fit(kst[a], μ1)...)
        Sg = particle_entropy(e[a] - g0[a] + g1, m[a], V[a], eos)
        if isfinite(Sg)
            Sc[a] = Sg
        else
            Threads.atomic_add!(GUARD_BACKMAP, 1)          # keep the plain map
        end
    end
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
        pa = particles[a];  V[a] = pa.s / pa.S_smooth;  e[a] = particle_energy(pa, eos)
    end
    return (V = V, e = e, stars = build_stars(particles, params, cl, eos, V; cache))
end

# Defaults: λ = 1 (detailed balance), the single vote (vote = :star: one factor
# min(1, e^{λ ΣΔS}) per move; :bonds, the per-bond product, rejects more and heats faster
# in coupled runs, docs/fullruns.tex), the gradient-corrected change of representation.
# One stochastic viscous step at fixed positions: a stress sweep and a residual sweep,
# in an order that alternates with `flip`.  With `gradient` (default, unless `frozen` is
# given) the change of representation uses the gradient-corrected map.  Returns acceptance fractions.  Pass
# `frozen = stochastic_frozen(...)` to keep V_a, e_a and the stars fixed across steps
# (a frozen lattice); otherwise they are recomputed from the current state.  Pass the same
# `cache = StochCache()` to every step of a simulation to reuse its arrays (faster, same results).
# The settings of the stochastic step, with their defaults in one place:
#   λ         residual move: 1 (detailed balance) or Inf (deterministic)
#   residual  do the residual move (else the stress move only)
#   vote      :star (one consensus vote per star) or :bonds
#   shift     random shift of the checkerboard grid for each sweep
#   gradient  gradient-corrected back map; default on, except with hamiltonian = :energy
#             (H = Σ e_a exactly, no gradient energy to carry; energystep.tex) and on a
#             frozen lattice
#   hastings  the Hastings correction (inverse_moment.tex, stage 2): the coefficients of each star from the present state of its
#             particle at every proposal, and the ratio of the reverse and forward proposal densities in the vote.  Needs
#             dissipation = :moment, λ = 1, the single vote, 2D.  Default from SPH_HASTINGS, else false.
# The default of `gradient` is resolved here, so the struct holds the value used.
struct StochOptions
    λ        :: Float64
    residual :: Bool
    vote     :: Symbol
    shift    :: Bool
    gradient :: Bool
    hastings :: Bool
end
function StochOptions(params::SimParams; λ = 1.0, residual::Bool = true, vote::Symbol = :star,
                      shift::Bool = true, frozen::Bool = false,
                      gradient::Bool = !frozen && params.hamiltonian !== :energy,
                      hastings::Bool = setting("SPH_HASTINGS", false))
    gradient && params.hamiltonian === :energy &&
        error("stochastic_step!: gradient = true is not consistent with hamiltonian = :energy")
    hastings && !(is_moment(params) && λ == 1 && vote === :star && !(params.kernel isa WendlandC2_1D)) &&
        error("SPH_HASTINGS=1 needs dissipation = :moment, λ = 1, the single vote and 2D (inverse_moment.tex, stage 2)")
    return StochOptions(Float64(λ), residual, vote, shift, gradient, hastings)
end

# The step with the settings as keywords (tests, examples): builds the StochOptions.
function stochastic_step!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                          dt::Float64, eos::AbstractEOS;
                          λ::Float64 = 1.0, residual::Bool = true, vote::Symbol = :star,
                          shift::Bool = true, frozen = nothing,
                          gradient::Bool = frozen === nothing && params.hamiltonian !== :energy,
                          kw...)
    opts = StochOptions(params; λ, residual, vote, shift, frozen = frozen !== nothing, gradient)
    return stochastic_step!(particles, params, cl, dt, eos, opts; frozen, kw...)
end

function stochastic_step!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                          dt::Float64, eos::AbstractEOS, opts::StochOptions;
                          flip::Bool = false, rng::AbstractRNG = Random.default_rng(),
                          rngs::Vector{Xoshiro} = PHYS_RNG, groups = color_groups(cl),
                          frozen = nothing, cache = nothing)
    (; λ, residual, vote, shift, gradient, hastings) = opts
    @assert length(rngs) == length(particles) "rng vector length ≠ particle count — call seed_phys!(N, seed)"
    oneD = params.kernel isa WendlandC2_1D
    fz = frozen === nothing ? stochastic_frozen(particles, params, cl, eos; cache) : frozen
    V, e, stars = fz.V, fz.e, fz.stars
    N  = length(particles)
    m  = [pa.m for pa in particles]
    # measure = :liouville: Sc holds the vote entropy S_a − ln T_a during the sweeps; the particle entropies are the true S_a,
    # recomputed at the end.  Not with the deterministic residual (λ = ∞) or the gradient back map, which write true S_a into Sc.
    liouville = params.measure === :liouville
    liouville && (isinf(λ) || gradient) && error("measure = :liouville is not available with λ = ∞ or the gradient back map")
    Sc = Vector{Float64}(undef, N)
    Threads.@threads for a in 1:N
        Sc[a] = vote_entropy(e[a], m[a], V[a], eos, liouville)
    end
    # Gradient-corrected change of representation (changerep.tex, Appendix A): the gradient
    # energy g_a of the long-wavelength momentum profile is frozen during the sweeps, so
    # e_a = e′_a + g_a is fixed there; the back map refits it, s′ = S_a(e′_a + g′_a, m′_a).
    gr = gradient ? gradient_freeze(particles, params, cl, eos, cache) : nothing
    kinds = residual ? (flip ? (:resid, :stress) : (:stress, :resid)) : (:stress,)
    acc = Dict(:stress => NaN, :resid => NaN)
    grj = Dict(:stress => 0.0, :resid => 0.0)          # fraction of proposals rejected by the guard
    for kind in kinds
        if kind === :resid && isinf(λ)
            resid_deterministic!(m, Sc, e, V, stars, dt, oneD, eos)
        else
            # random shift of the grid for each sweep (1D: along x only)
            u = shift ? SVector(rand(rng) * cl.cell_size[1], oneD ? 0.0 : rand(rng) * cl.cell_size[2]) :
                        zero(SVector{2,Float64})
            cells = shifted_cells(particles, cl, u)
            g0 = GUARD_REJECTED[]
            acc[kind] = hastings ?
                color_sweep_hastings!(m, Sc, e, V, stars, groups, cells, kind, dt, eos, rngs, rng, liouville, params) / N :
                color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, λ, oneD,
                             eos, rngs, rng, vote, liouville) / N
            grj[kind] = (GUARD_REJECTED[] - g0) / N
        end
    end
    gradient && gradient_backmap!(Sc, gr, m, e, V, eos)
    for (a, pa) in enumerate(particles)
        pa.m = m[a];  pa.s = liouville ? particle_entropy(e[a], m[a], V[a], eos) : Sc[a]
    end
    # acceptances, and the fractions of proposals rejected by the guard (a state no fluid can
    # have; for the conformal EOS exactly |v| ≥ 1), for the stress and the residual sweep
    return (stress = acc[:stress], resid = acc[:resid], gstress = grj[:stress], gresid = grj[:resid])
end
