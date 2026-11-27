using StaticArrays
using Random
using LinearAlgebra
using Printf

const RNG = MersenneTwister(0)

abstract type AbstractKernel end
struct WendlandC2_2D <: AbstractKernel end   # sim dimension = 2 (default)
struct WendlandC2_1D <: AbstractKernel end   # sim dimension = 1
struct WendlandC4_2D <: AbstractKernel end   # 2D, smoother (SPH_KERNEL=wc4); same support 2h
struct WendlandC6_2D <: AbstractKernel end   # 2D, smoother still (SPH_KERNEL=wc6); same support 2h

# The kernel type is a parameter, so that every function taking params is compiled for one
# concrete kernel: with the field typed AbstractKernel and four kernel methods, Julia stops
# union-splitting and each kernel call in the neighbour loops becomes a dynamic dispatch with
# an allocation (5× slower, and worse with several realizations in one process).
mutable struct SimParams{K<:AbstractKernel}
    L       :: SVector{2,Float64}  # (Lx, Ly) after snapping to lattice
    a       :: Float64             # lattice constant (= h/ha_ratio); uniform in x and y
    h       :: Float64             # smoothing length
    N       :: Int                 # total particle count
    vol     :: Float64             # volume (area in 2D, length in 1D) per particle
    seed    :: Int
    phys_dim :: Int                # physical dimension d of the fluid (2 or 3): ζ_eff and the conformal cs² = 1/d (dimension.tex)
    lattice :: Symbol              # :square or :hex
    kernel  :: K                   # WendlandC2_2D(), C4, C6 (SPH_KERNEL) or WendlandC2_1D()
    dissipation :: Symbol          # :particle (default), :bond or :two_level — see viscous_step!
    bond_h      :: Float64         # smoothing length h_ψ of the bond weight −W′(ρ; h_ψ)/ρ for :bond (support 2h_ψ ≤ 2h)
    hamiltonian :: Symbol          # :smoothed (default) or :energy — see below
    volume      :: Symbol          # :entropy (default) or :number, with :energy — see below
    ideal       :: Symbol          # :generalized (default in 2D), :particle or :grid — see below
    measure     :: Symbol          # :liouville (default in 2D) or :einstein — see below
end

# dissipation: the viscous step.  :common is :particle with the bond weights of the SPH kernel at the lab
#   distance, ψ_ab = −W′(r_ab)/r_ab (bond moment n_a 𝟙), and the comoving metric put in by hand
#   (commonkernel.tex; 2D, hamiltonian = :energy); its residual bonds carry the factor (r̂·ℓ r̂)².  It was the
#   default from 2026-11-14 to 2026-11-15.  :moment (inverse_moment.tex; 2D, hamiltonian = :energy) is :common without the
#   correction Λ_a of the weights: the faces carry the inverse bond moment, A_ab = V_a w_ab 𝓜_a⁻¹ r_ab, the residual is the
#   least-squares residual with the weights c_ab, and the residual move is conditioned on zero virial.  It is the default since
#   2026-11-15 wherever it applies (2D with the energy Hamiltonian), together with the Hastings correction of the stochastic
#   step (SPH_HASTINGS, StochOptions).  :particle (the default in 1D
#   and with SPH_HAMILTONIAN=smoothed; formerly :changerep, still accepted) is the bond form in the particle
#   representation with the gradient-corrected conversion map (changerep.tex); :bond is
#   the bond form on the smoothed b (bond_dissipation.tex); :two_level the SPH strain form
#   (sph2.tex), superseded: it and its stochastic sweep (particle_sweep!, which never
#   worked) are in src/legacy/, loaded by src/legacy/load.jl after src/load.jl.
#   Default from the environment variable SPH_DISSIPATION, else default_dissipation(sim_dim).
# Default scheme (2026-11-10): hamiltonian = :energy with volume = :number, the n-scheme
#   (volume_n.tex, number_tests.tex).  The old schemes are kept: SPH_HAMILTONIAN=smoothed,
#   SPH_VOLUME=entropy.  With SPH_DISSIPATION=two_level the default Hamiltonian is :smoothed
#   (:energy needs the bond viscous step).
# hamiltonian: the ideal step.  :smoothed is H = Σ s_a ℰ(S_a, M_a/S_a) with the
#   smoothed momentum (sph2.tex).  :energy (default) is H = Σ_a e_a, the particle energies at the
#   particle's own momentum density m_a/V_a, V_a = s_a/S_a (energystep.tex): ẋ_a = v_a,
#   a pure pressure force, and a viscous step on the particle variables without a change
#   of representation.  Default from SPH_HAMILTONIAN, else :energy.
# volume (hamiltonian = :energy only): the particle volume.  :entropy is
#   V_a = s_a/S_a, S_a = Σ_b s_b W_ab; :number (default) is V_a = 1/n_a, n_a = Σ_b W_ab, independent
#   of the weights s, so the back map of the stochastic step at fixed positions conserves
#   H exactly (volume_n.tex).  Stored as S_smooth = s_a n_a, so V_a = s_a/S_smooth either
#   way.  Default from SPH_VOLUME, else :number with :energy (:entropy with :smoothed).
# ideal: the ideal step.  :particle is the drift–kick–drift Hamiltonian leapfrog of the moving particles
#   (ideal.jl); :grid is the step on the fixed lattice — kick, drift and return (gridstep.tex,
#   grid.jl): the particles stay on the sites of the hexagonal lattice, which needs lattice = :hex,
#   hamiltonian = :energy and volume = :number.  :generalized is the moving particles with the generalized
#   leapfrog (hmc.jl: implicit, symplectic and reversible for the non-separable H; equilibrium_positions.tex),
#   hamiltonian = :energy and volume = :number, about 3× the cost of :particle.  Default from SPH_IDEAL, else
#   default_ideal: :generalized where it applies (2D, hamiltonian = :energy, volume = :number; since 2026-11-15), else :particle.
#   SPH_IDEAL=particle gives the earlier default (needed to reproduce runs before that date).
# measure: the integration measure of the stochastic step and of hybrid Monte Carlo (equilibrium_positions.tex, Appendix A).
#   :einstein is flat in the conserved energy and momentum (the Einstein formula; the particle entropy read as
#   the Boltzmann entropy).  :liouville is flat in the entropy, the measure the Hamiltonian ideal step preserves (s_a the
#   adiabatic invariant, the Gibbs entropy): the votes, the heat bath and the equilibrium start use S_a − ln T_a; the hybrid
#   Monte Carlo acceptance drops Σ ln T_a.  Needs hamiltonian = :energy.  Default from SPH_MEASURE, else default_measure:
#   :liouville in 2D with hamiltonian = :energy (since 2026-11-15), else :einstein.  SPH_MEASURE=einstein gives the earlier
#   default.  :liouville cannot be used with the deterministic residual (λ = ∞) or the gradient back map.
# Primary constructor: fixes a = h/ha_ratio, then snaps Lx and Ly to the nearest
# integer multiple of the lattice row spacing (a for square, a√3/2 in y for hex).
# sim_dim=1: single row of particles along x; Ly=a, vol=a, kernel=WendlandC2_1D().
# The Hamiltonian when none is given: SPH_HAMILTONIAN, else :energy (:smoothed with the
# two-level form, which needs it).  Drivers that need the scheme before SimParams use this too.
# The viscous step when none is given (2026-11-15): the inverse-moment step :moment where it applies (2D, energy
# Hamiltonian; inverse_moment.tex), else :particle.  SPH_DISSIPATION=common SPH_HASTINGS=0 gives the default of
# 2026-11-14/15 back, SPH_DISSIPATION=particle the one before.
default_dissipation(sim_dim = 2) =
    sim_dim == 2 && setting("SPH_HAMILTONIAN", :energy) === :energy ? :moment : :particle
# The largest Lorentz factor a fluid particle may have (2026-11-15; docs/inverse_moment.tex, Stage 2 results): a state with
# γ > SPH_GAMMA_MAX has no weight (the stochastic votes, the equilibrium start and hybrid Monte Carlo reject it, through
# particle_entropy), and the comoving structures are built with the velocity capped there.  Near the light cone the inversion of
# the equation of state loses accuracy as 1e-16 γ² (relative, in 1 − v², T, S), and the comoving metric and the residual coefficients
# become singular; nothing in our flows needs γ ≳ 10.  Default 100 (|v|² ≤ 1 − 10⁻⁴).
const GAMMA_MAX = setting("SPH_GAMMA_MAX", 100.0)
# SPH_RENORM=split (docs/step5_plan.tex, docs/renormalization_review.tex): Step 5, the label part of the counterterm in the measure, the
# rest in H (renorm_split.jl).  The DEFAULT since 2026-11-23 (merge of branch split): with SPH_RENORM unset the run is renormalized.
# SPH_RENORM=0 is the tree level (the default before).  SPH_RENORM=legendre (docs/legendre_plan.tex) was removed on 2026-11-26.
# Split supports d = 2, the energy Hamiltonian with the number volume, the Liouville measure, not the grid step (check_split): any other
# scheme stops with a message to set SPH_RENORM=0.
const SPLIT = let r = setting_peek("SPH_RENORM");  r === nothing || lowercase(strip(r)) == "split";  end
# SPH_RENORM=sscheme (docs/renorm3.tex, docs/sscheme_plan.tex): state counting at one loop, s = S_t − Δ_χ − Δ_ℓ (sscheme.jl).  The particle
# carries the label pa.s (which can be ≤ 0) and its target entropy pa.St; the volume, energy and velocity are stored (pa.V, pa.H, pa.v), not
# derived from s.  Same requirements as split (check_sscheme).
const SSCHEME = let r = setting_peek("SPH_RENORM");  r !== nothing && lowercase(strip(r)) == "sscheme";  end
# the Hamiltonian has a domain (split, sscheme): a state outside it has NaN fields, and hybrid Monte Carlo rejects the trajectory
const HAS_DOMAIN = SPLIT || SSCHEME
# the values of SPH_RENORM; any other stops here.  Removed on 2026-11-26 (docs/sscheme_plan.tex, sec:prune): SPH_RENORM=1, the entropy
# shift of partition_v2.tex, and SPH_RENORM=legendre (docs/legendre_plan.tex); the last commit with both is 9846a6c (branch sscheme-measure).
let r = setting_peek("SPH_RENORM")
    r === nothing || lowercase(strip(r)) in ("split", "sscheme", "0", "false") ||
        error("SPH_RENORM=$r: the values are split (the default), sscheme and 0 (the tree level).  SPH_RENORM=1 (the entropy shift, not " *
              "consistent with one loop) and SPH_RENORM=legendre were removed on 2026-11-26; to rerun an old line use commit 9846a6c " *
              "(docs/sscheme_plan.tex, sec:prune).")
end
const V2MAX = 1 - 1 / GAMMA_MAX^2
# the velocity with |v|² capped at V2MAX (the same vector, unchanged, below the cap)
cap_velocity(v::SVector{2,Float64}) = (v2 = dot(v, v);  v2 <= V2MAX ? v : v * sqrt(V2MAX / v2))

# The ideal step and the measure when none is given (2026-11-15): the generalized leapfrog and the Liouville measure where
# they apply (2D, energy Hamiltonian, and for the leapfrog the n-scheme), else :particle and :einstein.
default_ideal(sim_dim, hamiltonian, volume) =
    sim_dim == 2 && hamiltonian === :energy && volume === :number ? :generalized : :particle
default_measure(sim_dim, hamiltonian) = sim_dim == 2 && hamiltonian === :energy ? :liouville : :einstein
default_hamiltonian(dissipation = setting("SPH_DISSIPATION", default_dissipation())) =
    setting("SPH_HAMILTONIAN", dissipation === :two_level ? :smoothed : :energy)

function SimParams(Lx_t::Real, Ly_t::Real, h, ha_ratio; seed=42, sim_dim=2, phys_dim=default_phys_dim(sim_dim), lattice=:square,
                   dissipation=setting("SPH_DISSIPATION", default_dissipation(sim_dim)), bond_h=nothing,
                   hamiltonian=default_hamiltonian(dissipation),
                   volume=setting("SPH_VOLUME", hamiltonian === :energy ? :number : :entropy),
                   ideal=setting("SPH_IDEAL", default_ideal(sim_dim, hamiltonian, volume)),
                   measure=setting("SPH_MEASURE", default_measure(sim_dim, hamiltonian)))
    dissipation === :changerep && (dissipation = :particle)    # the old name of :particle
    a = h / ha_ratio
    if sim_dim == 1
        Nx      = round(Int, Lx_t / a)
        Lx      = Nx * a;  Ly = a
        N       = Nx
        vol     = a
        lattice = :square
        kernel  = WendlandC2_1D()
    elseif lattice == :hex
        dy = a * sqrt(3) / 2
        Nx = round(Int, Lx_t / a)
        Ny = 2 * round(Int, Ly_t / (2*dy))   # force even for periodic offset rows
        Lx = Nx * a;  Ly = Ny * dy
        N  = Nx * Ny
        vol = a^2 * sqrt(3) / 2
        kernel  = Dict("wc2" => WendlandC2_2D(), "wc4" => WendlandC4_2D(), "wc6" => WendlandC6_2D())[setting("SPH_KERNEL", "wc2")]
    else
        Nx = round(Int, Lx_t / a)
        Ny = round(Int, Ly_t / a)
        Lx = Nx * a;  Ly = Ny * a
        N  = Nx * Ny
        vol = a^2
        kernel  = WendlandC2_2D()
    end
    L = SVector(Lx, Ly)
    dim_str = kernel isa WendlandC2_1D ? "1D" : kernel isa WendlandC4_2D ? "2D (Wendland C4)" : kernel isa WendlandC6_2D ? "2D (Wendland C6)" : "2D"
    @printf("SimParams: Lx=%-6g  Ly=%-6g  a=%-8g  h=%-8g  N=%-6d  h/a=%.3g  vol=%.4g  lattice=%s  sim=%s  dissipation=%s\n",
            Lx, Ly, a, h, N, h/a, vol, lattice, dim_str, dissipation)
    h/a < 0.5 && @warn "h/a = $(h/a) < 0.5: particles barely overlap, shot noise will dominate"
    h/a > 10  && @warn "h/a = $(h/a) > 10: very diffuse smoothing, resolution is poor"
    dissipation === :common_lab && error("dissipation = :common_lab was removed (commonkernel.tex: the residual must carry the comoving metric); use :common")
    dissipation in (:particle, :bond, :two_level, :common, :moment) ||
        error("dissipation must be :particle, :bond, :two_level, :common or :moment, got $dissipation")
    dissipation in (:common, :moment) && (sim_dim == 2 && hamiltonian === :energy ||
        error("dissipation = :$dissipation needs a 2D simulation and hamiltonian = :energy (commonkernel.tex)"))
    dissipation === :moment && @printf("SimParams: dissipation = moment, inverse-moment faces and the conditioned residual (inverse_moment.tex)\n")
    dissipation === :common && @printf("SimParams: dissipation = common, one kernel for the ideal and the viscous step (commonkernel.tex)\n")
    hamiltonian in (:smoothed, :energy) ||
        error("hamiltonian must be :smoothed or :energy, got $hamiltonian")
    hamiltonian === :energy && dissipation === :two_level &&
        error("hamiltonian = :energy needs the bond viscous step (dissipation = :particle or :bond)")
    hamiltonian === :energy && @printf("SimParams: hamiltonian = energy (energystep.tex)\n")
    volume in (:entropy, :number) || error("volume must be :entropy or :number, got $volume")
    volume === :number && hamiltonian !== :energy &&
        error("volume = :number needs hamiltonian = :energy (volume_n.tex)")
    volume === :number && @printf("SimParams: volume = number, V_a = 1/n_a (volume_n.tex)\n")
    hψ = bond_h === nothing ? Float64(h) : Float64(bond_h)
    0 < hψ <= h || error("bond_h = $hψ must satisfy 0 < bond_h ≤ h (bond support 2·bond_h ≤ 2h stencil)")
    ideal in (:particle, :grid, :generalized) || error("ideal must be :particle, :grid or :generalized, got $ideal")
    ideal === :generalized && (hamiltonian === :energy && volume === :number ||
        error("ideal = :generalized needs hamiltonian = :energy and volume = :number (hmc.jl)"))
    ideal === :generalized && @printf("SimParams: ideal = generalized, the generalized leapfrog (hmc.jl)\n")
    ideal === :grid && (lattice === :hex && sim_dim == 2 && hamiltonian === :energy && volume === :number ||
        error("ideal = :grid needs the hexagonal lattice in 2D, hamiltonian = :energy and volume = :number (gridstep.tex)"))
    ideal === :grid && @printf("SimParams: ideal = grid, the step on the fixed lattice (gridstep.tex)\n")
    measure in (:einstein, :liouville) || error("measure must be :einstein or :liouville, got $measure")
    measure === :liouville && (hamiltonian === :energy ||
        error("measure = :liouville needs hamiltonian = :energy (equilibrium_positions.tex, Appendix A)"))
    measure === :liouville && @printf("SimParams: measure = liouville, flat in the entropy (S_a − ln T_a in the votes)\n")
    sp = SimParams(L, a, h, N, vol, seed, phys_dim, lattice, kernel, dissipation, hψ, hamiltonian, volume, ideal, measure)
    check_split(sp)                               # SPH_RENORM=split: the schemes it supports and V̄, before any map is used (renorm_split.jl)
    check_sscheme(sp)                             # SPH_RENORM=sscheme: the schemes it supports (sscheme.jl)
    sp
end

# Stochastic runs (the stochastic step, hybrid Monte Carlo) are two-dimensional (dimension.tex).
function check_stochastic_dims(params::SimParams)
    check_split(params)                           # SPH_RENORM=split (renorm_split.jl)
    check_sscheme(params)                         # SPH_RENORM=sscheme (sscheme.jl)
    sim_dim(params) == 2 || error("stochastic runs need sim_dim = 2; one-dimensional stochastic runs were removed (dimension.tex)")
    params.phys_dim == sim_dim(params) ||
        error("stochastic runs need phys_dim = sim_dim (= 2), got phys_dim = $(params.phys_dim): with fluctuations the fluid is the " *
              "one the particles span (dimension.tex); phys_dim ≠ sim_dim is for deterministic runs only")
    return nothing
end

# The default physical dimension of a run with simulation dimension sim_dim (dimension.tex): d = d_s.  Before branch dimfix
# it was 3 (a z-independent 3D fluid, cs² = 1/3, ζ_eff = ζ + η/3); a deterministic run may still ask for phys_dim = 3.
default_phys_dim(sim_dim::Integer = 2) = sim_dim

# Square-box convenience wrapper: SimParams(L, h, ha_ratio; kw...) → Lx=Ly=L.
SimParams(L::Real, h, ha_ratio; kw...) = SimParams(L, L, h, ha_ratio; kw...)

# The simulation dimension d_s: the number of directions the particles span (1 with the 1D kernel, else 2).
sim_dim(params::SimParams) = params.kernel isa WendlandC2_1D ? 1 : 2

# ── Equation of state ─────────────────────────────────────────────────────────

abstract type AbstractEOS end

# The equation of state with a constant speed of sound (dimension.tex):
#   p = cs² e,   T_rel = T_scale e^(cs²/(1+cs²)),   s_rel = (e+p)/T_rel,
# thermodynamically consistent (ds/de = 1/T, dp/dT = s) and scale invariant (p ∝ T^(1+1/cs²)) for any cs²;
# conformal only for cs² = 1/d (conformal_eos).
struct ConstantCs2EOS <: AbstractEOS
    cs2_val :: Float64   # sound speed squared
    T_scale :: Float64   # T_rel = T_scale * e^(cs2/(1+cs2)); sets temperature units
    η_by_s  :: Float64   # specific shear viscosity η/s (constant)
    ζ_by_s  :: Float64   # specific bulk viscosity  ζ/s (constant)
end

# The physical entropy of a particle: the label at the tree level (and split, whose label is the target's entropy), the
# target entropy S_t with SPH_RENORM=sscheme, whose label s = S_t − Δ is not an entropy.
entropy_of(pa) = SSCHEME ? pa.St : pa.s

# Gluon-gas Stefan-Boltzmann scale: e = (16π²/30) T⁴  →  T_scale = (30/(16π²))^(1/4)
const T_scale_gluon = (30.0 / (16π^2))^(1/4)

# Natural-units scale (ℓ_Q = T_0 = 1) for the 2D conformal fluid, cs² = 1/2:
# s_rel = (3/2) T_rel²/T_scale³, so s_rel(T_rel=1) = 1 ⟹ T_scale = (3/2)^(1/3).
# (For cs² = 1/3 the analogous value is (4/3)^(1/4).)
const T_scale_2d_lq1 = (3.0 / 2.0)^(1/3)

# Keyword constructor.
ConstantCs2EOS(; cs2_val, T_scale=1.0, η_by_s=0.0, ζ_by_s=0.0) = ConstantCs2EOS(cs2_val, T_scale, η_by_s, ζ_by_s)

# The conformal fluid in d physical dimensions: cs² = 1/d.  The drivers use conformal_eos(params.phys_dim; …).
conformal_eos(d::Integer; T_scale=1.0, η_by_s=0.0, ζ_by_s=0.0) = ConstantCs2EOS(1.0 / d, T_scale, η_by_s, ζ_by_s)

# The parameters of the equation of state and the physical dimension, for the run record (.h5 attributes).
eos_meta(params, eos::ConstantCs2EOS) =
    (phys_dim = params.phys_dim, sim_dim = sim_dim(params), cs2 = eos.cs2_val, T_scale = eos.T_scale,
     eta_by_s = eos.η_by_s, zeta_by_s = eos.ζ_by_s)

pressure(eos::ConstantCs2EOS, e::Float64) = eos.cs2_val * e

# Relativistic temperature
# (exponent 1/4 for c² = 1/3: two square roots; 1/3 for c² = 1/2: cbrt; both much cheaper than the power)
Trel(eos::ConstantCs2EOS, e::Float64) = eos.cs2_val == 1/3 ? eos.T_scale * sqrt(sqrt(e)) :
                                        eos.cs2_val == 1/2 ? eos.T_scale * cbrt(e) :
                                                             eos.T_scale * e^(eos.cs2_val / (1.0 + eos.cs2_val))

# Sound speed squared (two-arg form for generic EOS; one-arg shortcut for ConstantCs2EOS)
cs2(eos::ConstantCs2EOS,  e::Float64) = eos.cs2_val
cs2(eos::ConstantCs2EOS)              = eos.cs2_val

# Specific transport coefficients
eta_by_s(eos::ConstantCs2EOS)  = eos.η_by_s
zeta_by_s(eos::ConstantCs2EOS) = eos.ζ_by_s

# The bulk viscosity of the stochastic step (dimension.tex, "The bulk viscosity").  The in-plane step carries the shear of the d − 2
# unresolved directions in the bulk channel, ζ_eff = ζ + 2η(1/2 − 1/d) (= ζ at d = d_s = 2), and it is floored at ε η: the
# Hastings-corrected stress move needs a full-rank noise (shear and bulk), and with ζ_eff = 0 its Jacobian is singular
# (acceptance 0).  ζ_eff/s = max(ζ/s + 2(η/s)(1/2 − 1/d), ε η/s), ε = SPH_ZETA_FLOOR (default 0.01; 0 removes the floor).
const ZETA_FLOOR = setting("SPH_ZETA_FLOOR", 0.01)
zeta_eff_by_s(eos::AbstractEOS, d::Integer) =
    max(zeta_by_s(eos) + 2eta_by_s(eos) * (1 / 2 - 1 / d), ZETA_FLOOR * eta_by_s(eos))

# Relativistic entropy density: s_rel = (e+p)/T_rel  (thermodynamic identity)
srel(eos::ConstantCs2EOS, e::Float64) = (e + pressure(eos, e)) / Trel(eos, e)
# The rest-frame temperature of the target fluid at the rest-frame entropy density s_rel (docs/inputs.tex, T_rel,0 of a run): the
# equation of state alone, the same for every Hamiltonian (tree level, split, Legendre).  s_rel(e) is monotonic: bisection in e.
function e_of_srel(eos::AbstractEOS, s::Float64)       # the rest-frame energy density at rest-frame entropy density s (bisection)
    lo, hi = 0.0, 1.0
    while srel(eos, hi) < s;  hi *= 2;  end
    for _ in 1:200
        mid = (lo + hi) / 2;  srel(eos, mid) < s ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-15 * hi && break
    end
    return (lo + hi) / 2
end
trel_of_srel(eos::AbstractEOS, s::Float64) = Trel(eos, e_of_srel(eos, s))

# ─────────────────────────────────────────────────────────────────────────────

mutable struct Particle
    # primitive variables:
    x        :: SVector{2, Float64}   # position
    s        :: Float64                # entropy per particle
    m        :: SVector{2, Float64}   # momentum per particle
    
    # Thermodynamic fields computed by smooth_to_particles! 
    S_smooth :: Float64                # smoothed entropy density at x_a
    M_smooth :: SVector{2, Float64}   # smoothed momentum density at x_a
    
    # Thermodynamic fields computed by solve_eos!
    e        :: Float64                # internal energy density (Newton warm start)
    p        :: Float64                # pressure
    T        :: Float64                # T = T_rel/γ = (e+p)/S  (density-frame temperature)
    γ        :: Float64                # Lorentz factor γ = sqrt(1 + u²)
    v_over_S :: SVector{2, Float64}   # (v/S)(x_a)
    F_over_S :: Float64                # (F/S)(x_a)
    H_over_S :: Float64                # ℰ = energy per entropy; H = Σ s_a ℰ(x_a)

    # Particle-level derivatives of the Hamiltonian (smoothed sums)
    dotx     :: SVector{2, Float64}   # ẋ_a = Σ_b s_b (v/S)_b W  [smoothed velocity]
    Teff     :: Float64                # H_over_S_a − Σ_b s_b (F/S)_b W

    # Stochastic-sweep accumulators (per-particle Metropolis sweep).
    # During a Metropolis sweep s and m are held frozen as coefficients; the
    # accepted increments land here and are folded in by apply_sweep! afterwards.
    Δs       :: Float64                # Σ over accepted bonds of −b_a·δ_ab
    Δm       :: SVector{2, Float64}   # Σ over accepted bonds of ±δ_ab

    # Stored by the density pass and solve_eos! (every scheme; the code reads these, not s/S_smooth, s·H_over_S, v_over_S·S_smooth):
    V        :: Float64                # the particle volume (1/n_a with the number volume)
    H        :: Float64                # the particle energy H(s_a, m_a, V_a)
    v        :: SVector{2, Float64}   # the particle velocity ∂H/∂m_a
    St       :: Float64                # the target entropy γ s(T_rel) V of the present state (= s_a at the tree level; SPH_RENORM=sscheme)
end

Particle(x::SVector{2,Float64}, s::Float64) =
    Particle(x, s,
             zero(SVector{2,Float64}),   # m
             0.0,                         # S_smooth
             zero(SVector{2,Float64}),   # M_smooth
             0.0, 0.0, 0.0, 1.0,         # e, p, T, γ
             zero(SVector{2,Float64}),   # v_over_S
             0.0, 0.0,                    # F_over_S, H_over_S
             zero(SVector{2,Float64}), 0.0,   # dotx, Teff
             0.0, zero(SVector{2,Float64}),   # Δs, Δm
             NaN, NaN, zero(SVector{2,Float64}), s)   # V, H, v, St

Particle(x::SVector{2,Float64}, s::Float64, m::SVector{2,Float64}) =
    Particle(x, s, m,
             0.0,                         # S_smooth
             zero(SVector{2,Float64}),   # M_smooth
             0.0, 0.0, 0.0, 1.0,         # e, p, T, γ
             zero(SVector{2,Float64}),   # v_over_S
             0.0, 0.0,                    # F_over_S, H_over_S
             zero(SVector{2,Float64}), 0.0,   # dotx, Teff
             0.0, zero(SVector{2,Float64}),   # Δs, Δm
             NaN, NaN, zero(SVector{2,Float64}), s)   # V, H, v, St

function kernel_W(r::Float64, h::Float64, ::WendlandC2_2D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (7.0 / (4π * h^2)) * (1.0 - q/2.0)^4 * (1.0 + 2.0*q)
end

function kernel_W(r::Float64, h::Float64, ::WendlandC2_1D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (3.0 / (4h)) * (1.0 - q/2.0)^4 * (1.0 + 2.0*q)
end

kernel_W(r::Float64, h::Float64) = kernel_W(r, h, WendlandC2_2D())

# Wendland C4 in 2D, support 2h:  W = 9/(4π h²) (1 − q/2)⁶ (1 + 3q + 35q²/12),  q = r/h
function kernel_W(r::Float64, h::Float64, ::WendlandC4_2D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (9.0 / (4π * h^2)) * (1.0 - q/2.0)^6 * (1.0 + 3.0*q + 35.0*q^2/12.0)
end
function kernel_gradW(r_vec::SVector{2,Float64}, r::Float64, h::Float64, ::WendlandC4_2D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    C    = 9.0 / (4π * h^2)
    dWdr = -(14.0/3.0) * C * q * (1.0 + 2.5*q) * (1.0 - q/2.0)^5 / h
    return dWdr / r .* r_vec
end

function kernel_gradW(r_vec::SVector{2,Float64},
                          r::Float64,
                          h::Float64, ::WendlandC2_2D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    C    = 7.0 / (4π * h^2)
    dWdr = -5C * q * (1 - q/2)^3 / h
    return dWdr / r .* r_vec
end

function kernel_gradW(r_vec::SVector{2,Float64},
                          r::Float64,
                          h::Float64, ::WendlandC2_1D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    C    = 3.0 / (4h)
    dWdr = -5C * q * (1 - q/2)^3 / h
    return dWdr / r .* r_vec
end

kernel_gradW(rv, r, h) = kernel_gradW(rv, r, h, WendlandC2_2D())

# Wendland C6 in 2D, support 2h:  W = 78/(28π h²) (1 − q/2)⁸ (1 + 4q + 25q²/4 + 4q³),  q = r/h
function kernel_W(r::Float64, h::Float64, ::WendlandC6_2D)::Float64
    q = r / h
    q >= 2.0 && return 0.0
    return (78.0 / (28π * h^2)) * (1.0 - q/2.0)^8 * (1.0 + 4.0*q + 6.25*q^2 + 4.0*q^3)
end
function kernel_gradW(r_vec::SVector{2,Float64}, r::Float64, h::Float64, ::WendlandC6_2D)
    q = r / h
    (q >= 2.0 || r < 1e-10) && return SVector(0.0, 0.0)
    s = q / 2;  C = 78.0 / (28π * h^2)
    dWdr = -22.0 * C * s * (1 + 7s + 16s^2) * (1 - s)^7 / (2h)
    return dWdr / r .* r_vec
end

function minimum_image(r::SVector{2,Float64}, L::SVector{2,Float64})
    r .- L .* round.(r ./ L)
end

# Given the entropy density S = s_rel γ and the momentum density M = (e+p) γ² v = w γ u
# (the density-frame T^{0i}; sph2.tex §II.4), find the rest-frame energy density e.
# With w = e + p and G ≡ w γ:  M = G u and γ² = 1 + u²  ⇒  G⁴ − w² G² − w² M² = 0, so
#   G(w, M) = √[(w² + √(w⁴ + 4 w² M²))/2],   γ = G/w,   u = M/G,   v = u/γ = M w/G².
# The residual is f(e) = G/T_rel − S.
function eos_wgamma(w::Float64, M::Float64)
    R = sqrt(w^4 + 4 * w^2 * M^2)
    G = sqrt((w^2 + R) / 2)
    dGdw = (w + (w^3 + 2 * w * M^2) / R) / (2G)       # from G² = (w² + R)/2
    return G, dGdw
end

# Newton solve of G(w(e), |M|)/T_rel(e) = S for the rest-frame energy density e, from the
# warm start e0 (cold start e ~ S if e0 ≤ 0).  Pure: used by solve_eos! and particle_energy.
function eos_newton(S_in::Float64, M_mag::Float64, e0::Float64, eos::AbstractEOS;
                    tol=1e-12, maxiter=50, verbose=false)
    e = e0 > 0.0 ? e0 : S_in
    for iter in 1:maxiter
        w     = e + pressure(eos, e)
        G, Gw = eos_wgamma(w, M_mag)
        T     = Trel(eos, e)
        f     = G / T - S_in
        verbose && @printf("  iter %2d:  e = %+.12e   f = %+.4e\n", iter, e, f)
        abs(f) < tol * S_in && break
        # f'(e) = [G'(w)(1 + cs²) − G cs²/w] / T,  using dT/de = cs² T/w
        c2 = cs2(eos, e)
        fp = (Gw * (1.0 + c2) - G * c2 / w) / T
        Δe = f / fp
        while e - Δe ≤ 0.0          # keep e positive; bisect toward zero if needed
            Δe *= 0.5
        end
        e -= Δe
    end
    return e
end

# The state of a particle at its smoothed fields, through the forward map particle_forward (pointwise.jl; legendre_plan.tex, A).
function solve_eos!(pa::Particle, eos::AbstractEOS; tol=1e-12, maxiter=50, verbose=false)
    SSCHEME && return sscheme_solve!(pa, eos)
    S   = pa.S_smooth
    st  = particle_forward(pa.s, S, pa.M_smooth, pa.e, eos; γ0 = pa.γ, tol, maxiter, verbose)
    pa.e        = st.e
    pa.p        = st.p
    pa.T        = st.T          # T = T_rel/γ = (e+p)/S
    pa.γ        = st.γ
    pa.v_over_S = st.v / S
    pa.F_over_S = st.FS
    pa.H_over_S = st.HS
    pa.H        = pa.s * pa.H_over_S          # the expressions the code used before the fields were stored (bit-identical)
    pa.v        = pa.v_over_S * pa.S_smooth
end

function update_thermodynamics!(particles::Vector{Particle}, eos::AbstractEOS)
    Threads.@threads for pa in particles
        solve_eos!(pa, eos)
    end
end

# smooth_to_grid is defined in celllist.jl (depends on CellList2D)
