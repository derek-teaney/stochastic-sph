# positional.jl — the positional counterterm (docs/sscheme_plan.tex, "Level 2", eq. omega; items H, I), SPH_POSITIONAL=1 (default 0: off).
#
# The one-loop correction of the positions per particle, at the particle's own target state and volume,
#   ω(σ̄, V) = a₁ ν_* + a₂ ν_*²,   ν_* = −I₀(σ̄) V/(2 a₁ h²),   σ̄ = S_t/n = p γ V/T_rel,   a₁ = π/(2 sin(2π/n_W)),
# with I₀(σ) = h² ∫ d²k/(2π)² ln S(k), 1/S = (1 − W̃)[1 + (2σ − 1) W̃] (W̃ the transform of the kernel), tabulated by
# examples/i0_table.py (src/tables/i0_table_<kernel>.csv: σ = 1e-8 to 1e5; constant below, linear in ln σ above).  a₂ = SPH_POSITIONAL_A2 (default 0: one loop; the fit of the
# level-1 data is 0.091, sscheme_plan.tex, "The correction of the positions").  Its two slopes,
#   ω_σ = ∂ω/∂ln σ̄|_V = ω′ ν_* I₀′/I₀,   ω_V = ∂ω/∂ln V|_σ̄ = ω′ ν_*,   ω′ = a₁ + 2a₂ν_*,
# use I₀′ = the derivative of the interpolant (a cubic Hermite spline in ln σ through the tabulated I₀ and I₀′), so that the forces are exact
# derivatives of one Hamiltonian.  Where it enters:
#   SPH_RENORM=sscheme  the label, s = S_t − Δ^H + ω (sscheme.jl: Δ_p → Δ_p − a ω_σ, Δ_θ → Δ_θ − v ω_σ, P_a V/T_a → + ω_V + ω_σ/n);
#   SPH_RENORM=split    the energy, E_dyn = T (Δs_dyn − ω(u/n, V)) (renorm_split.jl).
# h and the kernel are those of the run (positional_set!, called by check_split / check_sscheme when SimParams is built).
const POS_ON = setting("SPH_POSITIONAL", 0) != 0
const POS_A2 = setting("SPH_POSITIONAL_A2", 0.0)
const POS_H = Ref(NaN)                      # the smoothing length of the run
const POS_A1 = π / (2 * sin(2π / 5))        # Wendland C2 (n_W = 5)

struct I0Table
    x0 :: Float64;  dx :: Float64;  I :: Vector{Float64};  Ip :: Vector{Float64}     # uniform grid in x = ln σ; I₀ and dI₀/dx
end
function load_i0_table(file)
    rows = [parse.(Float64, split(l, ",")) for l in readlines(file)[2:end] if !isempty(strip(l))]
    x = [log(r[1]) for r in rows];  dx = (x[end] - x[1]) / (length(x) - 1)
    maximum(abs.(diff(x) .- dx)) < 1e-6 * dx || error("positional.jl: $file is not uniform in ln σ")
    return I0Table(x[1], dx, [r[2] for r in rows], [r[3] for r in rows])
end
const I0_WC2 = load_i0_table(joinpath(@__DIR__, "tables", "i0_table_wc2.csv"))

# I₀ and its derivative in ln σ, from the cubic Hermite interpolant (the derivative is the interpolant's own)
@inline function i0_eval(σ::Float64, t::I0Table = I0_WC2)
    x = log(σ);  u = (x - t.x0) / t.dx;  n = length(t.I)
    u < 0 && return t.I[1], 0.0                 # below the table (σ < 1e-8; the trial labels of a vote's Newton steps): I₀ → const, I₀′ ~ σ ln σ → 0
    u > n - 1 && return t.I[n] + t.Ip[n] * (x - t.x0 - (n - 1) * t.dx), t.Ip[n]     # above (σ > 1e5): linear in ln σ
    i = min(floor(Int, u), n - 2);  τ = u - i
    y0 = t.I[i+1];  y1 = t.I[i+2];  m0 = t.Ip[i+1] * t.dx;  m1 = t.Ip[i+2] * t.dx
    τ2 = τ * τ;  τ3 = τ2 * τ
    I = (2τ3 - 3τ2 + 1) * y0 + (τ3 - 2τ2 + τ) * m0 + (-2τ3 + 3τ2) * y1 + (τ3 - τ2) * m1
    dI = ((6τ2 - 6τ) * y0 + (3τ2 - 4τ + 1) * m0 + (-6τ2 + 6τ) * y1 + (3τ2 - 2τ) * m1) / t.dx
    return I, dI
end

# ω and its slopes (ω_σ = ∂ω/∂ln σ̄ at fixed V, ω_V = ∂ω/∂ln V at fixed σ̄) at σ̄ and V
@inline function positional_omega(σ::Float64, V::Float64)
    I, dI = i0_eval(σ)
    ν = -I * V / (2 * POS_A1 * POS_H[]^2)
    ω = POS_A1 * ν + POS_A2 * ν * ν;  ωp = POS_A1 + 2 * POS_A2 * ν
    νσ = -dI * V / (2 * POS_A1 * POS_H[]^2)              # ∂ν/∂ln σ̄
    return (ω = ω, ωσ = ωp * νσ, ωV = ωp * ν, ν = ν)
end

function positional_set!(params)
    POS_ON || return nothing
    params.kernel isa WendlandC2_2D || error("SPH_POSITIONAL=1: the table of I₀ is for Wendland C2 in 2D only (SPH_KERNEL=wc2)")
    POS_H[] = params.h
    return nothing
end
