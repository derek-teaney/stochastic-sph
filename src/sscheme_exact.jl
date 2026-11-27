# sscheme_exact.jl — the exact count of the single particle as the label, SPH_SSCHEME_FORM=exact (docs/sscheme_plan.tex, "The exact form").
#
# The label of the exact form is the exact number of states of the particle (renorm2.tex, "With momentum"; examples/exact_count.py),
#   s = s_exact = ln G(e, |m|, V) − ln V − 1 = s_min + ln A + g(x, θ),   x = ln S_t,  θ the rapidity of the target,
# with the flat Liouville measure e^s ds d²m d²x (no wall, no β ln S_t, no reference).  For the conformal fluid the label depends on the target
# entropy and rapidity alone (the one-loop Δ does too: Δ = b ln S_t + ½ ln(1 − c_s²v²) + const, testsscheme.jl test 4), and on A only through ln A.
# s_min = ln K(n/2) − 1 = ln(1/2π) − 1 is the floor: as S_t → 0 the count is that of one quantum, g → 0 as S_t³ and ∂s/∂e → 0 (T_a → ∞).
# ℓ = ln g is tabulated by `python3 examples/exact_count.py dyntable` (src/tables/exact_lng_c05.csv, c_s² = 1/2 only) with its nodal derivatives
# ℓ_x, ℓ_θ, ℓ_xθ, on x ∈ [ln 10⁻³, ln 10⁶] (the exact count to S_t = 600, S_1loop + c(a)/S_t + d(θ)/S_t² above, matched), θ ∈ [0, 6], and
# interpolated here by bicubic Hermite polynomials (C¹: the forces are the derivatives of one Hamiltonian).  Outside: ℓ linear in x (below,
# g ∝ S_t³ exactly; above, g ≈ S_t), θ clamped at 6 (γ > 200; ℓ_θ ≈ 10⁻⁴ there).  ℓ_θ = 0 at θ = 0 (the table is mirrored in θ).
#
# The state (sscheme.jl, sscheme_state) follows from G_x = ∂s/∂x and G_θ = ∂s/∂θ|_x (g's slopes, plus ω's: s → s + ω(S_t/n, V), G_x += ω_σ),
# without the one-loop parts, so that no cancellation enters where g is small:
#   y = (a G_x − v G_θ)/(a S_t),   z = (v G_x + G_θ)/S_t,   𝒟 = (1 + v²) y − v z,   T_a = T/𝒟,   v_a = (2 v y − z)/𝒟,   P_a V/T_a = G_x/n + ω_V,
# the general relations of the S-scheme (𝒟 = 1 − (1 + v²) ε_p + v ε_∥, ε_p = Δ_p/(S_t a), ε_∥ = Δ_θ/S_t) with Δ = S_t − s.

struct ExactTable
    x0 :: Float64;  dx :: Float64;  nx :: Int
    t0 :: Float64;  dt :: Float64;  nt :: Int
    F :: Matrix{Float64};  Fx :: Matrix{Float64};  Ft :: Matrix{Float64};  Fxt :: Matrix{Float64}
    smin :: Float64                         # the floor of the exact label at A = V = 1
end
function load_exact_table(file)
    lines = readlines(file)
    m = match(r"nx = (\d+) nt = (\d+); s_min\(A=V=1\) = ([-+0-9.eE]+)", lines[1])
    m === nothing && error("sscheme_exact.jl: no header in $file")
    nx, nt = parse(Int, m[1]), parse(Int, m[2]);  smin = parse(Float64, m[3])
    rows = [parse.(Float64, split(l, ",")) for l in lines[3:end] if !isempty(strip(l))]
    length(rows) == nx * nt || error("sscheme_exact.jl: $file has $(length(rows)) rows, expected $(nx * nt)")
    x = [rows[(i - 1) * nt + 1][1] for i in 1:nx];  t = [rows[j][2] for j in 1:nt]
    dx = (x[end] - x[1]) / (nx - 1);  dt = (t[end] - t[1]) / (nt - 1)
    (maximum(abs.(diff(x) .- dx)) < 1e-9 && maximum(abs.(diff(t) .- dt)) < 1e-9) || error("sscheme_exact.jl: $file is not a uniform grid")
    F, Fx, Ft, Fxt = (zeros(nx, nt) for _ in 1:4)
    for i in 1:nx, j in 1:nt
        r = rows[(i - 1) * nt + j];  F[i, j] = r[3];  Fx[i, j] = r[4];  Ft[i, j] = r[5];  Fxt[i, j] = r[6]
    end
    Ft[:, 1] .= 0.0;  Fxt[:, 1] .= 0.0                     # even in θ
    return ExactTable(x[1], dx, nx, t[1], dt, nt, F, Fx, Ft, Fxt, smin)
end
const SSX_TABLE = Ref{Union{Nothing, ExactTable}}(nothing)
function ssx_table()
    t = SSX_TABLE[]
    t === nothing || return t
    t = load_exact_table(joinpath(@__DIR__, "tables", "exact_lng_c05.csv"))
    SSX_TABLE[] = t
    return t
end

# the bicubic Hermite interpolant of ℓ: (ℓ, ℓ_x, ℓ_θ) at (x, θ) inside the table
@inline function ssx_hermite(t::ExactTable, x::Float64, θ::Float64)
    u = (x - t.x0) / t.dx;  w = (θ - t.t0) / t.dt
    i = clamp(floor(Int, u), 0, t.nx - 2);  j = clamp(floor(Int, w), 0, t.nt - 2)
    u -= i;  w -= j
    h0(s) = 2s^3 - 3s^2 + 1;   h1(s) = s^3 - 2s^2 + s;   g0(s) = -2s^3 + 3s^2;   g1(s) = s^3 - s^2
    dh0(s) = 6s^2 - 6s;        dh1(s) = 3s^2 - 4s + 1;   dg0(s) = -6s^2 + 6s;    dg1(s) = 3s^2 - 2s
    Bu = (h0(u), h1(u) * t.dx, g0(u), g1(u) * t.dx);  dBu = (dh0(u) / t.dx, dh1(u), dg0(u) / t.dx, dg1(u))
    Bw = (h0(w), h1(w) * t.dt, g0(w), g1(w) * t.dt);  dBw = (dh0(w) / t.dt, dh1(w), dg0(w) / t.dt, dg1(w))
    f = 0.0;  fx = 0.0;  ft = 0.0
    @inbounds for (a, ii) in ((0, i + 1), (1, i + 2)), (b, jj) in ((0, j + 1), (1, j + 2))
        cv = t.F[ii, jj];  cx = t.Fx[ii, jj];  ct = t.Ft[ii, jj];  cxt = t.Fxt[ii, jj]
        pu = 1 + 2a;  su = 2 + 2a;  pw = 1 + 2b;  sw = 2 + 2b     # basis: value at the corner (1 + 2a), slope (2 + 2a)
        f  += cv * Bu[pu] * Bw[pw]  + cx * Bu[su] * Bw[pw]  + ct * Bu[pu] * Bw[sw]  + cxt * Bu[su] * Bw[sw]
        fx += cv * dBu[pu] * Bw[pw] + cx * dBu[su] * Bw[pw] + ct * dBu[pu] * Bw[sw] + cxt * dBu[su] * Bw[sw]
        ft += cv * Bu[pu] * dBw[pw] + cx * Bu[su] * dBw[pw] + ct * Bu[pu] * dBw[sw] + cxt * Bu[su] * dBw[sw]
    end
    return f, fx, ft
end

# g = s_exact − s_min (A = V = 1) and its slopes (g_x, g_θ) at x = ln S_t and the rapidity θ ≥ 0
@inline function exact_g(x::Float64, θ::Float64)
    t = ssx_table()
    tmax = t.t0 + (t.nt - 1) * t.dt;  θc = min(θ, tmax)
    xmin = t.x0;  xmax = t.x0 + (t.nx - 1) * t.dx
    if x < xmin                                            # below: ℓ linear in x (g ∝ S_t³)
        ℓ, ℓx, ℓt = ssx_hermite(t, xmin, θc);  ℓ += ℓx * (x - xmin)
    elseif x > xmax                                        # above: ℓ linear in x (g ≈ S_t)
        ℓ, ℓx, ℓt = ssx_hermite(t, xmax, θc);  ℓ += ℓx * (x - xmax)
    else
        ℓ, ℓx, ℓt = ssx_hermite(t, x, θc)
    end
    θc < θ && (ℓt = 0.0)
    g = exp(ℓ)
    return g, g * ℓx, g * ℓt
end

function check_sscheme_exact(eos)
    (eos isa ConstantCs2EOS && abs(eos.cs2_val - 0.5) < 1e-12) ||
        error("SPH_SSCHEME_FORM=exact: tabulated for the conformal fluid in d = 2 only (c_s² = 1/2; got $(eos))")
    ssx_table()
    return nothing
end

# the next order of the one-loop label, c(a) (renorm3.tex, appendix, eq. c; d = 2), and dc/da (for the tests: the rows above S_t = 600)
function ssx_c2(q::Float64, a::Float64)
    l3 = 2q * (q - 1);  l4 = 2q * (q - 1) * (3q + 1)
    return -3 / 8 + l4 / (8a^2) - 5l3^2 / (24a^3), -l4 / (4a^3) + 5l3^2 / (8a^4)
end
