# hmc.jl — hybrid Monte Carlo on the positions and momenta (docs/equilibrium_positions.tex, Sects. V and VIII).
# Samples, at fixed multipliers β⁰ = 1/T and β = v/T, the joint distribution with the Liouville measure
#   π_L ∝ exp[Σ_a s_a − β⁰ H + β·P]   (flat in the entropies; Appendix A of the note).
# (The Einstein measure, flat in the particle energies and momenta, was removed on 2026-11-27: docs/focus_plan.tex, Q2.)
# One update:
#   1. heat bath at fixed positions: per particle, an independence Metropolis chain for (e_a, m_a), continued from the
#      current values (so the conditional distribution is preserved exactly), then s_a = S_a(e_a, m_a);
#   2. a Hamiltonian trajectory at fixed s_a with the generalized leapfrog (symplectic and reversible for the
#      non-separable H), accepted with min(1, exp[−β⁰ΔH + β·ΔP]).
# The n-scheme, where ∂H/∂m_a = v_a and −∂H/∂x_a is hamiltonian_force!.

# The multipliers and the proposal of the heat bath: the most probable densities (ε̄, M̄) at (β⁰, β) and the Gaussian
# pieces of a particle of volume Vref (rescaled per particle: Hn ∝ 1/V, L ∝ √V).  The proposal only affects the
# acceptance of the chain, not its target.  Its Gaussian is that of the target's own weight (the Liouville measure, vote_entropy;
# until docs/focus_plan.tex Q9 that of the weight flat in (e, m), the former Einstein measure).
struct HMCTarget
    β0   :: Float64
    β    :: SVector{2,Float64}
    ε̄    :: Float64
    M̄    :: SVector{2,Float64}
    Vref :: Float64
    L    :: SMatrix{3,3,Float64,9}
    Hn   :: SMatrix{3,3,Float64,9}
end
function HMCTarget(β0, β::SVector{2,Float64}, Vref::Float64, eos::AbstractEOS)
    st = state_from_beta(β0, β, eos)
    pm = particle_moments(Vref, β0, β, eos)
    return HMCTarget(β0, β, st.ε, st.M, Vref, pm.L, pm.Hn)
end

# Smooth and solve; returns H and P of the present state.
function hmc_measure!(ps, params, cl, eos)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    return sum(q.H for q in ps), sum(q.m for q in ps)
end

# 1. Heat bath at fixed positions (the smoothed state must be current).  Returns the acceptance of the chains.
function hmc_heatbath!(ps, params, cl, eos, tg::HMCTarget, rng; nmh::Int = 10, widen::Float64 = 2.0)
    nacc = 0                                              # the target S_a − ln T_a (the Liouville measure, sph.jl)
    for pa in ps
        V = pa.V;  r = V / tg.Vref
        ē = V * tg.ε̄;  m̄ = V * tg.M̄;  Lw = sqrt(widen * r) * tg.L;  Hn = tg.Hn / r
        g = SPLIT ? (pa.s, SVector(pa.e, 0.0)) : nothing   # the start of the inverse solves
        logp(x) = vote_entropy(ē + x[1], m̄ + SVector(x[2], x[3]), V, eos; guess = g) - tg.β0 * x[1] + dot(tg.β, SVector(x[2], x[3]))
        logq(x) = -0.5 * dot(x, Hn * x) / widen
        e = particle_energy(pa, eos)
        x = SVector(e - ē, pa.m[1] - m̄[1], pa.m[2] - m̄[2]);  wx = logp(x) - logq(x)
        HAS_DOMAIN && isnan(wx) && (wx = -Inf)             # a current state with no weight (a failed solve): any valid proposal leaves it
        for _ in 1:nmh
            y = Lw * SVector(randn(rng), randn(rng), randn(rng));  wy = logp(y) - logq(y)
            if isfinite(wy) && log(rand(rng)) < wy - wx
                x, wx = y, wy;  nacc += 1
            end
        end
        pa.m = m̄ + SVector(x[2], x[3])
        pa.s = particle_entropy(ē + x[1], pa.m, V, eos; guess = g)
    end
    return nacc / (length(ps) * nmh)
end

# −∂H/∂x (into F) and ∂H/∂m = v of the present (x, m), fixed s.
function hmc_force!(F, ps, params, cl, eos)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    hamiltonian_force!(F, ps, params, cl)
    return F
end
function hmc_velocity!(v, ps, params, cl, eos)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
    for (a, q) in enumerate(ps);  v[a] = q.dotx;  end
    return v
end
setx!(ps, params, cl, x) = (for (a, q) in enumerate(ps);  q.x = x[a];  end;  wrap_positions!(ps, params.L);  rebuild_cell_list!(cl, ps))
setm!(ps, m) = (for (a, q) in enumerate(ps);  q.m = m[a];  end)

# One step of the generalized leapfrog (implicit half steps iterated to tol).  Returns the iterations used, or −1 if the step reached
# a state with no Hamiltonian (a force or a velocity not finite: with a renormalized Hamiltonian a particle whose label is below the edge
# of the single-particle domain at its new volume): H = +∞ there, and the trajectory is rejected (hmc_update!).  strict = true (the ideal
# step, generalized_step!): the finiteness is checked at the tree level too, and an implicit iteration that does not converge returns −2
# instead of a warning.
function generalized_leapfrog!(ps, params, cl, eos, dt, F, v0, v1; tol = 1e-13, maxit = 60, strict::Bool = false)
    chk = HAS_DOMAIN || strict
    N = length(ps);  m0 = [q.m for q in ps];  x0 = [q.x for q in ps];  mh = copy(m0)
    msc = maximum(norm, m0) + eps();  its = 0
    for it in 1:maxit                                     # m½ = m − δt/2 ∂H/∂x(x, m½)
        setm!(ps, mh);  hmc_force!(F, ps, params, cl, eos)
        chk && !all(f -> isfinite(f[1]) && isfinite(f[2]), F) && return -1
        d = 0.0
        for a in 1:N;  mn = m0[a] + (dt / 2) * F[a];  d = max(d, norm(mn - mh[a]));  mh[a] = mn;  end
        its += 1;  d <= tol * msc && break
        it == maxit && (strict ? (return -2) : @warn "generalized_leapfrog!: momentum half step not converged" d / msc)
    end
    setm!(ps, mh);  hmc_velocity!(v0, ps, params, cl, eos)
    chk && !all(v -> isfinite(v[1]) && isfinite(v[2]), v0) && return -1
    xn = [x0[a] + dt * v0[a] for a in 1:N]                # x′ = x + δt/2 [v(x, m½) + v(x′, m½)]
    for it in 1:maxit
        setx!(ps, params, cl, xn);  hmc_velocity!(v1, ps, params, cl, eos)
        chk && !all(v -> isfinite(v[1]) && isfinite(v[2]), v1) && return -1
        d = 0.0
        for a in 1:N
            xa = x0[a] + (dt / 2) * (v0[a] + v1[a])
            d = max(d, norm(minimum_image(xa - ps[a].x, params.L)));  xn[a] = xa
        end
        its += 1;  d <= tol * params.a && break
        it == maxit && (strict ? (return -2) : @warn "generalized_leapfrog!: position step not converged" d / params.a)
    end
    setx!(ps, params, cl, xn);  hmc_force!(F, ps, params, cl, eos)
    chk && !all(f -> isfinite(f[1]) && isfinite(f[2]), F) && return -1
    setm!(ps, [mh[a] + (dt / 2) * F[a] for a in 1:N])     # m′ = m½ − δt/2 ∂H/∂x(x′, m½)
    return its
end

# 2. A trajectory of nsteps generalized leapfrog steps; no acceptance test.  Returns the iterations used.
function hmc_trajectory!(ps, params, cl, eos, dt, nsteps; kw...)
    N = length(ps);  F = zeros(SVector{2,Float64}, N);  v0 = similar(F);  v1 = similar(F);  its = 0
    for _ in 1:nsteps
        i = generalized_leapfrog!(ps, params, cl, eos, dt, F, v0, v1; kw...)
        i < 0 && return -1                                # no Hamiltonian at the state reached: reject
        its += i
    end
    return its
end

# One update: heat bath, trajectory, accept or reject.  Returns (accepted, ΔH/H, Δ ln π, heat-bath acceptance,
# iterations per leapfrog step).  Leaves the smoothed state current.
function hmc_update!(ps, params, cl, eos, tg::HMCTarget, rng; nsteps::Int = 10, dt::Float64, nmh::Int = 10)
    check_stochastic_dims(params)
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    hb = hmc_heatbath!(ps, params, cl, eos, tg, rng; nmh)
    H0, P0 = hmc_measure!(ps, params, cl, eos)
    x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    its = hmc_trajectory!(ps, params, cl, eos, dt, nsteps)
    if its < 0                                            # the trajectory left the domain of H (split, sscheme): rejected
        setm!(ps, m0);  setx!(ps, params, cl, x0);  hmc_measure!(ps, params, cl, eos)
        return (accepted = false, dH = NaN, dlnpi = -Inf, heatbath = hb, iters = NaN)
    end
    H1, P1 = hmc_measure!(ps, params, cl, eos)
    # the Liouville measure is invariant under the trajectory (fixed s_a)
    dlnπ = -tg.β0 * (H1 - H0) + dot(tg.β, P1 - P0)
    acc = isfinite(dlnπ) && log(rand(rng)) < dlnπ
    if !acc
        setm!(ps, m0);  setx!(ps, params, cl, x0);  hmc_measure!(ps, params, cl, eos)
    end
    return (accepted = acc, dH = (H1 - H0) / H0, dlnpi = dlnπ, heatbath = hb, iters = its / nsteps)
end

# Hybrid Monte Carlo to equilibrium (the start of a run, docs/focus_plan.tex Q9; the drivers that sample equilibria): nupdates
# updates at the target tg, trajectories of nsteps leapfrog steps of δt = dt, each of a random length, uniform in
# [(1 − taujit) nsteps, nsteps], when taujit > 0.  callback(n, u) after each update n (u from hmc_update!).  Returns the
# fraction of trajectories accepted.
function hmc_equilibrate!(ps, params, cl, eos, tg::HMCTarget, rng; nupdates::Int, nsteps::Int, dt::Float64, taujit::Float64 = 0.0,
                          callback = nothing)
    nacc = 0
    for n in 1:nupdates
        nst = taujit > 0 ? max(1, round(Int, nsteps * (1 - taujit * rand(rng)))) : nsteps
        u = hmc_update!(ps, params, cl, eos, tg, rng; nsteps = nst, dt);  nacc += u.accepted
        callback === nothing || callback(n, u)
    end
    return nacc / nupdates
end
hmc_equilibrate!(callback::Function, ps, params, cl, eos, tg::HMCTarget, rng; kw...) =      # the do-block form
    hmc_equilibrate!(ps, params, cl, eos, tg, rng; callback, kw...)

# The ideal step with the generalized leapfrog (the only ideal step, ideal.jl leapfrog_step!): one step of
# generalized_leapfrog!, the implicit half steps iterated to GLF_TOL (SPH_GLF_TOL, default 1e-10: second order and
# reversible to that accuracy).  Called by leapfrog_step!; dotm is the force buffer.
# A failed step (docs/focus_plan.tex Q11: a state without a Hamiltonian, or implicit iterations that do not converge) is redone
# from its start as two half steps, recursively, down to Δt/16; each retry adds one to the guard GUARD_LEAPFROG (stochastic.jl,
# guards()).  If it still fails at Δt/16 the particles are put back at the start of the step and a LeapfrogFailure is thrown, with
# that state and the particles without a finite force or velocity: a run never continues from a state without a Hamiltonian.
# When no step fails the arithmetic is that of a plain generalized_leapfrog! call.
const GLF_TOL = setting("SPH_GLF_TOL")
const GLF_MAXDEPTH = 4                                    # Δt/2⁴ = Δt/16

struct LeapfrogFailure <: Exception
    reason :: String
    dt     :: Float64                                     # the step that failed (the full Δt)
    x      :: Vector{SVector{2,Float64}}                  # the state at the start of the step
    m      :: Vector{SVector{2,Float64}}
    s      :: Vector{Float64}
    bad    :: Vector{Int}                                 # particles without a finite force or velocity where the last try failed
end
Base.showerror(io::IO, e::LeapfrogFailure) =
    print(io, "LeapfrogFailure: ", e.reason, " at Δt/", 2^GLF_MAXDEPTH, " (Δt = ", e.dt, "); particles ", first(e.bad, 12),
          length(e.bad) > 12 ? " …" : "")

# one try of step dt, and on failure two half steps; returns 0 on success, else the code of the last failure (−1, −2)
function glf_retry!(ps, params, cl, eos, dt, F, v0, v1, depth)
    x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    its = generalized_leapfrog!(ps, params, cl, eos, dt, F, v0, v1; tol = GLF_TOL, strict = true)
    its >= 0 && return 0
    depth >= GLF_MAXDEPTH && return its
    Threads.atomic_add!(GUARD_LEAPFROG, 1)
    setm!(ps, m0);  setx!(ps, params, cl, x0)
    r = glf_retry!(ps, params, cl, eos, dt / 2, F, v0, v1, depth + 1)
    r == 0 || return r
    return glf_retry!(ps, params, cl, eos, dt / 2, F, v0, v1, depth + 1)
end

function generalized_step!(ps, params, cl, dt, eos, dotm)
    v0 = similar(dotm);  v1 = similar(dotm)
    x0 = [q.x for q in ps];  m0 = [q.m for q in ps]
    r = glf_retry!(ps, params, cl, eos, dt, dotm, v0, v1, 0)
    if r != 0
        finite(u) = isfinite(u[1]) && isfinite(u[2])
        bad = [a for a in eachindex(ps) if !(finite(dotm[a]) && finite(v0[a]) && finite(v1[a]))]
        setm!(ps, m0);  setx!(ps, params, cl, x0)
        smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
        throw(LeapfrogFailure(r == -1 ? "a state without a Hamiltonian (a force or a velocity not finite)" :
                                        "the implicit iterations did not converge", dt, x0, m0, [q.s for q in ps], bad))
    end
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos);  update_velocities!(ps, params, cl)
    return nothing
end
