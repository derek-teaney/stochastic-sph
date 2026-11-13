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
# record_grid! (the Recorder from the lattice FFT) against record! (direct sums), on a random state
let
    params, ps, cl, _ = redirect_stdout(devnull) do
        setup_ic(0.05, 2.0, eos, (x, y) -> 1e6, (x, y) -> SVector(0.0, 0.0); L = 1.2, lattice = :hex)
    end
    rng = Xoshiro(3)
    for p in ps;  p.s *= 1 + 0.01randn(rng);  p.m = 0.01 * p.s * SVector(randn(rng), randn(rng));  end
    smooth_to_particles!(ps, params, cl);  update_thermodynamics!(ps, eos)
    F = GridFFT(ps, params);  ks = box_wavevectors(params, 1.5)
    r1 = Recorder(ks; energies = true, x0 = [p.x for p in ps], L = params.L);  r2 = deepcopy(r1)
    record!(r1, ps, 0.0, eos);  record_grid!(r2, F, ps, 0.0, eos)
    rel(a, b) = maximum(abs.(a .- b)) / maximum(abs.(a))
    e = maximum((rel(r1.ML[1], r2.ML[1]), rel(r1.MT[1], r2.MT[1]), rel(r1.S[1], r2.S[1]), rel(r1.E[1], r2.E[1])))
    @printf("record_grid! against record! (%d modes, kh ≤ 1.5): max relative difference %.1e\n", length(ks), e)
    global ok &= e < 1e-10
end
println(ok ? "testgridfft: PASS" : "testgridfft: FAIL");  ok || exit(1)
