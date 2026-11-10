# cl_energy_split.jl — how much of the energy is in the resolved modes (k ≲ 1/h)?
# (docs/continuum_limit.tex, campaign cl.)
#
# For a particle state, three energies:
#   E_p  = Σ_a e_a           the n-scheme Hamiltonian: each particle's own T⁰⁰(s_a/V_a, m_a/V_a) V_a,
#                            V_a = 1/n_a (what the runs record as H)
#   E_cg = ∫ d²x T⁰⁰(S̄(x), M̄(x)),  S̄ = Σ_b s_b W(x − x_b),  M̄ = Σ_b m_b W(x − x_b)
#                            the energy of the coarse-grained fields, on a grid of spacing h/8
#   E_0  = A T⁰⁰(S_tot/A, M_tot/A)   the uniform state with the same totals (the minimum)
# also E_sph = Σ_a V_a T⁰⁰(S̄_a, M̄_a), the same coarse-grained energy by SPH quadrature
# (the old :smoothed Hamiltonian, with V_a = 1/n_a).
# The excess E − E_0 is the thermal energy of the fluctuations; the resolved fraction is
#   f_res = (E_cg − E_0) / (E_p − E_0).
# Reference: uncorrelated particle fluctuations with equipartition give
#   f_ref = (1/N) Σ_{k≠0} |W̃(k)|²   (the box modes, W̃(0) = 1),  ∝ (a/h)².
#
# Run:  julia -t 8 --project examples/cl_energy_split.jl  [draw:<h/a>[:<L>] ...] [path.chk ...]
#   draw:<h/a>  a fresh equilibrium start (as stochrep_equilibrium.jl, seed 1; box 0.6 unless :L)
#   path.chk    a checkpoint of a cl run (any code version with the same Particle struct)
# → a table on stdout and $(SR_OUT or data)/cl_energy_split.csv (appended)
using StaticArrays, LinearAlgebra, Printf, Random, Serialization, FFTW
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "rng.jl"))
include(joinpath(@__DIR__, "..", "src", "stochrep.jl"))
include(joinpath(@__DIR__, "..", "src", "measure.jl"))
include(joinpath(@__DIR__, "..", "src", "monitor.jl"))

const S0 = 1e6
const eos = ConformalEOS(1/3, 1.0, 0.2, 0.0)

# lab-frame energy density T⁰⁰ = wγ² − p = G²/w − p at entropy density S, momentum density M
function T00(S, M)
    e = eos_newton(S, M, 0.0, eos)
    p = pressure(eos, e);  w = e + p
    G, _ = eos_wgamma(w, M)
    return G^2 / w - p
end

# the coarse-grained fields on an ng × ng grid (cell-centred), by deposit within 2h
function grid_fields(ps, params)
    Lx, Ly = params.L;  h = params.h
    ng = 8 * ceil(Int, Lx / h);  dx = Lx / ng;  ngy = round(Int, Ly / dx);  dy = Ly / ngy
    nt = Threads.nthreads()
    Sg = [zeros(ng, ngy) for _ in 1:nt];  Mx = [zeros(ng, ngy) for _ in 1:nt];  My = [zeros(ng, ngy) for _ in 1:nt]
    R = 2h;  ri = ceil(Int, R / dx) + 1;  rj = ceil(Int, R / dy) + 1
    chunks = collect(Iterators.partition(eachindex(ps), cld(length(ps), nt)))
    Threads.@threads for c in eachindex(chunks)
        S_, X_, Y_ = Sg[c], Mx[c], My[c]
        for b in chunks[c]
            q = ps[b];  i0 = floor(Int, q.x[1] / dx);  j0 = floor(Int, q.x[2] / dy)
            for di in -ri:ri, dj in -rj:rj
                i = i0 + di;  j = j0 + dj
                r = minimum_image(SVector((i + 0.5) * dx, (j + 0.5) * dy) - q.x, params.L)
                W = wendland_c2(norm(r), h, params.kernel);  W == 0 && continue
                I = mod(i, ng) + 1;  J = mod(j, ngy) + 1
                S_[I, J] += q.s * W;  X_[I, J] += q.m[1] * W;  Y_[I, J] += q.m[2] * W
            end
        end
    end
    return sum(Sg), sum(Mx), sum(My), dx, dy
end

# f_ref = (1/N) Σ_{k≠0} |W̃(k)|², W̃ from the kernel sampled on the same grid
function fref(params, ng, ngy, dx, dy)
    Wg = [wendland_c2(norm(minimum_image(SVector(i * dx, j * dy), params.L)), params.h, params.kernel)
          for i in 0:ng-1, j in 0:ngy-1]
    Wk = abs2.(fft(Wg)) .* (dx * dy)^2
    return (sum(Wk) - Wk[1, 1]) / params.N
end

function energies(ps, params)
    smooth_to_particles!(ps, params, cl_of[])          # n-scheme: S_smooth = s_a n_a, M_smooth = m_a n_a
    update_thermodynamics!(ps, eos)
    Ep = sum(q.s * q.H_over_S for q in ps)
    A = params.L[1] * params.L[2]
    Stot = sum(q.s for q in ps);  Mtot = sum(q.m for q in ps)
    E0 = A * T00(Stot / A, norm(Mtot) / A)
    Sg, Mx, My, dx, dy = grid_fields(ps, params)
    Ecg = sum(T00(Sg[I], hypot(Mx[I], My[I])) for I in eachindex(Sg)) * dx * dy
    # SPH quadrature of the same field energy, V_a = s_a / S_smooth = 1/n_a
    nl = neighbor_list!(cl_of[], ps, params.h)
    Va = [q.s / q.S_smooth for q in ps]
    Eps = zeros(length(ps))
    Threads.@threads for a in eachindex(ps)
        S = 0.0;  M = zero(SVector{2,Float64})
        for b in neighbors(nl, a)
            W = wendland_c2(norm(minimum_image(ps[a].x - ps[b].x, params.L)), params.h, params.kernel)
            S += ps[b].s * W;  M += ps[b].m * W
        end
        Eps[a] = Va[a] * T00(S, norm(M))
    end
    Esph = sum(Eps)
    fr = fref(params, size(Sg)..., dx, dy)
    return (; Ep, Ecg, Esph, E0, f = (Ecg - E0) / (Ep - E0), fsph = (Esph - E0) / (Ep - E0), fr,
            exc_p = (Ep - E0) / E0, exc_cg = (Ecg - E0) / E0, Stot, A)
end

const cl_of = Ref{Any}(nothing)

function state_from(arg)
    if startswith(arg, "draw:")
        f = split(arg, ':');  ha = parse(Float64, f[2]);  L = length(f) > 2 ? parse(Float64, f[3]) : 0.6
        params, ps, cl, _ = setup_ic(0.05, ha, eos, (x, y) -> S0, (x, y) -> SVector(0.0, 0.0);
                                     L = L, lattice = :hex, seed = 1000 + 4)
        initial_state!(ps, params, cl, eos, :eq; rng = Xoshiro(1000 + 3))
        return (label = @sprintf("draw ha=%g", ha), ha = ha, t = 0.0, params, ps, cl)
    else
        st = load_checkpoint(arg)
        return (label = basename(arg), ha = st.params.h / st.params.a, t = st.n * st.dt,
                params = st.params, ps = st.ps, cl = st.cl)
    end
end

out = joinpath(get(ENV, "SR_OUT", joinpath(@__DIR__, "..", "data")), "cl_energy_split.csv");  mkpath(dirname(out))
isfile(out) || open(io -> println(io, "label,ha,t,N,s_a,E0,Ep,Ecg,Esph,exc_p,exc_cg,f_res,f_res_sph,f_ref"), out, "w")
@printf("%-34s %5s %6s %7s %10s %10s %8s %8s %8s\n", "state", "h/a", "t", "N", "(Ep-E0)/E0", "(Ecg-E0)/E0", "f_res", "f_sph", "f_ref")
for arg in ARGS
    s = state_from(arg)
    cl_of[] = s.cl
    r = energies(s.ps, s.params)
    sa = r.Stot / s.params.N
    @printf("%-34s %5.1f %6.2f %7d %10.3e %10.3e %8.4f %8.4f %8.4f\n", s.label, s.ha, s.t, s.params.N,
            r.exc_p, r.exc_cg, r.f, r.fsph, r.fr)
    open(out, "a") do io
        println(io, join((s.label, s.ha, s.t, s.params.N, sa, r.E0, r.Ep, r.Ecg, r.Esph, r.exc_p, r.exc_cg, r.f, r.fsph, r.fr), ","))
    end
end
println("→ ", out)
