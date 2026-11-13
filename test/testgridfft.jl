# testgridfft.jl — the lattice Fourier transform of the grid (src/grid.jl, GridFFT): against the direct
# sum (from the origin R_0, at k and at its first-zone representative), Parseval, the inverse, and |k_min| ≤ the zone corner.
using StaticArrays, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "src", "load.jl"))
eos = ConformalEOS(1/3, 1.0, 0.0, 0.0)
ok = true
for L in (0.6, 1.2)
    params, ps, cl, _ = redirect_stdout(devnull) do
        setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L, lattice = :hex)
    end
    F = GridFFT(ps, params);  N = length(ps);  rng = Xoshiro(1)
    f = randn(rng, N);  fk = grid_fft(F, f)
    err_direct = maximum(begin
        i, j = rand(rng, 1:F.Nx), rand(rng, 1:F.Ny)
        abs(fk[i, j] - sum(f[a] * cis(-dot(F.k[i, j], ps[a].x - F.R0)) for a in 1:N)) / sqrt(N)
    end for _ in 1:50)
    err_min = maximum(begin
        i, j = rand(rng, 1:F.Nx), rand(rng, 1:F.Ny)
        abs(fk[i, j] - sum(f[a] * cis(-dot(F.kmin[i, j], ps[a].x - F.R0)) for a in 1:N)) / sqrt(N)
    end for _ in 1:50)
    pars = abs(sum(abs2, fk) / (N * sum(abs2, f)) - 1)
    inv = maximum(abs.(grid_ifft(F, fk) .- f))
    kzone = maximum(norm, F.kmin) * params.a / (4π / 3)          # the zone corner is at |k| a = 4π/3
    @printf("L = %.1f, N = %d: direct sum %.1e, at the first-zone k %.1e, Parseval %.1e, inverse %.1e, max|k_min| a/(4π/3) = %.4f\n",
            L, N, err_direct, err_min, pars, inv, kzone)
    global ok &= err_direct < 1e-10 && err_min < 1e-10 && pars < 1e-12 && inv < 1e-12 && kzone < 1 + 1e-9
end
println(ok ? "testgridfft: PASS" : "testgridfft: FAIL");  ok || exit(1)
