# stochrep.jl — stochastic viscous step in the particle representation (docs/stochrep.tex).
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
# docs/stochrep.tex ("A moving fluid"), in 1D the rest-frame forms.  Moves in cells of the same color
# touch disjoint sets of particles and run in parallel; within a cell the moves are
# sequential, each seeing the current momenta.  The random numbers of the move of
# particle a come from rngs[a]; the sweep orders from the master rng.
#
# Include after celllist.jl (uses changerep.jl) and rng.jl.

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
end

function build_stars(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                     eos::AbstractEOS, V::Vector{Float64})
    N    = length(particles)
    h    = params.h;  L = params.L;  hψ = params.bond_h;  d = params.d
    oneD = params.kernel isa WendlandC2_1D
    v0   = zero(SVector{2,Float64})
    ℓ0, ℓi0, P0, Pℓ0 = comoving_structures(v0, cs2(eos), oneD ? d : 2)   # rest frame
    e1 = SMatrix{2,2,Float64}(1, 0, 0, -1) / sqrt(2);  e2 = SMatrix{2,2,Float64}(0, 1, 1, 0) / sqrt(2)
    stars = Vector{Star}(undef, N)
    Threads.@threads for a in 1:N
        pa = particles[a]
        nb = Tuple{Int,SVector{2,Float64},Float64}[]
        ix, iy = cell_index(pa.x, cl)
        for dx in neighbor_range(cl.ncx), dy in neighbor_range(cl.ncy)
            jx = mod(ix + dx - 1, cl.ncx) + 1;  jy = mod(iy + dy - 1, cl.ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b == a && continue
                r = minimum_image(particles[b].x - pa.x, L);  rm = norm(r)
                rm < 2h && push!(nb, (b, r, rm))
            end
        end
        # 2D: comoving forms at the particle's velocity (docs/stochrep.tex, "A moving
        # fluid"); 1D: rest frame.
        va = oneD ? v0 : pa.dotx                         # coarse-grained ẋ_a
        ℓ, ℓi, P, Pℓ = oneD ? (ℓ0, ℓi0, P0, Pℓ0) : comoving_structures(va, cs2(eos), 2)
        ψs, n_a = oneD ? bond_weights_1d(nb, particles, ℓi, hψ) :
                         bond_weights_comoving(nb, particles, ℓ, ℓi, hψ)
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
        bs = Int[];  rs = SVector{2,Float64}[];  ws = Float64[];  cs = Float64[];  As = SVector{2,Float64}[]
        for (k, (b, r, rr)) in enumerate(nb)
            (ψs[k] > 0 && n_a > 0) || continue
            wab = V[b] * ψs[k] / n_a
            push!(bs, b);  push!(rs, r);  push!(ws, wab)
            push!(cs, (oneD ? κs : 2κs) * Ta * V[a] * wab)
            push!(As, V[a] * wab * (ℓi * r))
        end
        stars[a] = Star(bs, rs, ws, cs, V[a], Ta, κs, κb, As, B, ℓ, ℓh, ℓi)
    end
    return stars
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

function propose_resid!(δ, st::Star, dt, λ, oneD, rng)
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

# Events guarded against, summed over the run (reset with reset_guards!()).
const GUARD_REJECTED = Threads.Atomic{Int}(0)    # moves rejected for an unphysical state
const GUARD_EVENSPLIT = Threads.Atomic{Int}(0)   # bond split fell back to the even split
const GUARD_BACKMAP = Threads.Atomic{Int}(0)     # gradient back map fell back to the plain map
const GUARD_SUBDIV = Threads.Atomic{Int}(0)      # deterministic residual step subdivided
reset_guards!() = foreach(c -> (c[] = 0), (GUARD_REJECTED, GUARD_EVENSPLIT, GUARD_BACKMAP, GUARD_SUBDIV))
guards() = (rejected = GUARD_REJECTED[], evensplit = GUARD_EVENSPLIT[], backmap = GUARD_BACKMAP[],
            subdiv = GUARD_SUBDIV[])

# Ask the bonds: accept with Π_b min(1, exp(λ ΔS_ab)), all or nothing.  The bond
# factor is ΔS_ab = ΔS_b + σ_ab + R/n_a, with σ_ab the share of ΔS_a due to the
# traction t_ab = δ[k] (vote = :bonds, docs/stochrep.tex):
#     σ_ab = ½[S_a(m_a − t_ab) − S_a(m_a)] − ½[S_a(m_a′ + t_ab) − S_a(m_a′)],
# m_a′ = m_a + δm_a, and R = ΔS_a − Σ_b σ_ab.  Each σ_ab flips sign exactly under the
# reverse move, and σ_ab ≈ β_a·t_ab, so ΔS_ab ≈ −Δβ_ab·t_ab is the bond's own entropy
# production (invariant under a uniform shift of β).  vote = :even uses σ_ab = 0
# (ΔS_a split evenly); vote = :star a single vote, min(1, exp(λ Σ_c ΔS_c)).  All exact.
function ask_star!(m, Sc, e, V, a, st::Star, δ, λ, eos, rng, Snew, vote::Symbol = :bonds)
    n  = length(st.nb)
    δa = -sum(@view δ[1:n])
    ma′ = m[a] + δa
    Sa = particle_entropy(e[a], ma′, V[a], eos)
    if !isfinite(Sa)
        Threads.atomic_add!(GUARD_REJECTED, 1);  return false
    end
    ΔSa = Sa - Sc[a]
    ΔStot = ΔSa
    for k in 1:n
        b = st.nb[k]
        Snew[k] = particle_entropy(e[b], m[b] + δ[k], V[b], eos)
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
            σ[k] = 0.5 * (particle_entropy(e[a], m[a] - δ[k], V[a], eos) - Sc[a]) -
                   0.5 * (particle_entropy(e[a], ma′ + δ[k], V[a], eos) - Sa)
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
function color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, λ, oneD, eos, rngs, rng, vote)
    nacc = Threads.Atomic{Int}(0)
    for g in shuffle(rng, collect(eachindex(groups)))
        cg = groups[g]
        order = [shuffle(rng, cells[ix, iy]) for (ix, iy) in cg]
        Threads.@threads for ci in eachindex(cg)
            δ = SVector{2,Float64}[];  Snew = Float64[]
            for a in order[ci]
                st = stars[a];  n = length(st.nb);  n == 0 && continue
                resize!(δ, n);  resize!(Snew, n)
                if kind === :stress
                    propose_stress!(δ, st, dt, oneD, rngs[a])
                    ok = ask_star!(m, Sc, e, V, a, st, δ, 1.0, eos, rngs[a], Snew, vote)
                else
                    propose_resid!(δ, st, dt, λ, oneD, rngs[a])
                    ok = ask_star!(m, Sc, e, V, a, st, δ, λ, eos, rngs[a], Snew, vote)
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

# ── The step ─────────────────────────────────────────────────────────────────
# Frozen data of a step: particle volumes, particle energies, stars.
function stochrep_frozen(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                         eos::AbstractEOS)
    smooth_to_particles!(particles, params, cl)
    update_thermodynamics!(particles, eos)
    update_velocities!(particles, params, cl)        # ẋ_a, the velocity of the comoving metric
    N = length(particles);  V = Vector{Float64}(undef, N);  e = Vector{Float64}(undef, N)
    Threads.@threads for a in 1:N
        pa = particles[a];  V[a] = pa.s / pa.S_smooth;  e[a] = particle_energy(pa, eos)
    end
    return (V = V, e = e, stars = build_stars(particles, params, cl, eos, V))
end

# One stochastic viscous step at fixed positions: a stress sweep and a residual sweep,
# in an order that alternates with `flip`.  With `gradient` (default, unless `frozen` is
# given) the change of representation uses the gradient-corrected map.  Returns acceptance fractions.  Pass
# `frozen = stochrep_frozen(...)` to keep V_a, e_a and the stars fixed across steps
# (a frozen lattice); otherwise they are recomputed from the current state.
function stochrep_step!(particles::Vector{Particle}, params::SimParams, cl::CellList2D,
                        dt::Float64, eos::AbstractEOS;
                        λ::Float64 = 1.0, residual::Bool = true, flip::Bool = false,
                        rng::AbstractRNG = Random.default_rng(),
                        rngs::Vector{Xoshiro} = PHYS_RNG, groups = color_groups(cl),
                        frozen = nothing, vote::Symbol = :bonds, shift::Bool = true,
                        gradient::Bool = frozen === nothing)
    @assert length(rngs) == length(particles) "rng vector length ≠ particle count — call seed_phys!(N, seed)"
    oneD = params.kernel isa WendlandC2_1D
    fz = frozen === nothing ? stochrep_frozen(particles, params, cl, eos) : frozen
    V, e, stars = fz.V, fz.e, fz.stars
    N  = length(particles)
    m  = [pa.m for pa in particles]
    Sc = Vector{Float64}(undef, N)
    Threads.@threads for a in 1:N
        Sc[a] = particle_entropy(e[a], m[a], V[a], eos)
    end
    # Gradient-corrected change of representation (changerep.tex, Appendix A): the gradient
    # energy g_a of the long-wavelength momentum profile is frozen during the sweeps, so
    # e_a = e′_a + g_a is fixed there; the back map refits it, s′ = S_a(e′_a + g′_a, m′_a).
    if gradient
        kst = Vector{Vector{Tuple{Int,SVector{2,Float64},Float64}}}(undef, N)
        Cs  = Vector{SMatrix{2,2,Float64,4}}(undef, N)
        s0  = [pa.s for pa in particles]
        μ0  = [pa.m / pa.s for pa in particles]
        g0  = Vector{Float64}(undef, N)
        Threads.@threads for a in 1:N
            kst[a] = kernel_star(a, particles, params, cl)
            Cs[a]  = inverse_susceptibility(particles[a], eos)
        end
        Threads.@threads for a in 1:N
            g0[a] = gradient_energy(s0[a], Cs[a], star_fit(kst[a], μ0)...)
        end
    end
    kinds = residual ? (flip ? (:resid, :stress) : (:stress, :resid)) : (:stress,)
    acc = Dict(:stress => NaN, :resid => NaN)
    for kind in kinds
        if kind === :resid && isinf(λ)
            resid_deterministic!(m, Sc, e, V, stars, dt, oneD, eos)
        else
            # random shift of the grid for each sweep (1D: along x only)
            u = shift ? SVector(rand(rng) * cl.cell_size[1], oneD ? 0.0 : rand(rng) * cl.cell_size[2]) :
                        zero(SVector{2,Float64})
            cells = shifted_cells(particles, cl, u)
            acc[kind] = color_sweep!(m, Sc, e, V, stars, groups, cells, kind, dt, λ, oneD,
                                     eos, rngs, rng, vote) / N
        end
    end
    if gradient
        μ1 = [m[a] / s0[a] for a in 1:N]                                   # s still old, as in the deterministic step
        Threads.@threads for a in 1:N
            g1 = gradient_energy(s0[a], Cs[a], star_fit(kst[a], μ1)...)
            Sg = particle_entropy(e[a] - g0[a] + g1, m[a], V[a], eos)
            if isfinite(Sg)
                Sc[a] = Sg
            else
                Threads.atomic_add!(GUARD_BACKMAP, 1)                 # keep the plain map
            end
        end
    end
    for (a, pa) in enumerate(particles)
        pa.m = m[a];  pa.s = Sc[a]
    end
    return (stress = acc[:stress], resid = acc[:resid])
end
