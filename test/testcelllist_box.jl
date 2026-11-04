# testcelllist_box.jl — cell list in a non-square box.
#
# The cell list must find every pair with r < 2h, for any box (Lx, Ly).
# Bug this guards against: cells were square with side Lx/ncx, reused in y.
# When Ly/ncy < Lx/ncx the rows of cells stop short of / overshoot Ly, and
# the periodic y-stencil no longer connects the top and bottom of the box,
# so pairs straddling y = 0 are silently dropped.
#
# Run:  julia --project test/testcelllist_box.jl

using StaticArrays
using Random
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))
include(joinpath(@__DIR__, "..", "src", "setup.jl"))

# All pairs (a<b) within 2h, by brute force over every pair.
function pairs_brute(particles, h, L)
    P = Set{Tuple{Int,Int}}()
    for a in eachindex(particles), b in a+1:length(particles)
        norm(minimum_image(particles[a].x - particles[b].x, L)) < 2h && push!(P, (a, b))
    end
    return P
end

# All pairs (a<b) within 2h, using the cell-list 3×3 stencil (as the SPH loops do).
function pairs_celllist(particles, cl, h, L)
    P = Set{Tuple{Int,Int}}()
    for a in eachindex(particles)
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(cl.ncy)
            jx = mod(ix + dx - 1, cl.ncx) + 1
            jy = mod(iy + dy - 1, cl.ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b > a || continue
                norm(minimum_image(particles[a].x - particles[b].x, L)) < 2h && push!(P, (a, b))
            end
        end
    end
    return P
end

# ── A. two particles straddling the y boundary ──────────────────────────────
println("[A] two particles across the y boundary, Lx=8.9  Ly=8.2  h=1")
let L = SVector(8.9, 8.2), h = 1.0
    ps = [Particle(SVector(4.0, 0.2), 1.0),     # just above y = 0
          Particle(SVector(4.0, 6.6), 1.0)]     # just below y = Ly; periodic distance 1.8 < 2h
    cl = CellList2D(L, h, length(ps)); rebuild_cell_list!(cl, ps)
    r  = norm(minimum_image(ps[1].x - ps[2].x, L))
    @printf("  ncx=%d ncy=%d   periodic distance r = %.2f  (2h = %.1f)\n", cl.ncx, cl.ncy, r, 2h)
    @printf("  cells: particle 1 in %s, particle 2 in %s\n",
            cell_index(ps[1].x, cl), cell_index(ps[2].x, cl))
    found = (1, 2) in pairs_celllist(ps, cl, h, L)
    println("  cell list finds the pair: ", found)
    @assert found "[A] cell list missed a pair across the y boundary"
    println("  [A] ✓")
end

# ── B. the hex lattice used by the stochastic runs ──────────────────────────
println("\n[B] hex lattice h=15, h/a=2.55, L=150 (box comes out 147.1 × 152.8)")
let eos = ConformalEOS(0.5, T_scale_2d_lq1, 0.0, 0.0)
    params, ps, cl, eos = setup_ic(15.0, 2.55, eos, (x, y) -> 1.0, (x, y) -> SVector(0.0, 0.0);
                                   L=150.0, lattice=:hex, d=2)
    nb = length(pairs_brute(ps, params.h, params.L))
    nc = length(pairs_celllist(ps, cl, params.h, params.L))
    S  = [p.S_smooth for p in ps]
    @printf("  pairs: brute force %d, cell list %d  (missing %d)\n", nb, nc, nb - nc)
    @printf("  S_smooth on a uniform lattice: min %.6f  max %.6f\n", minimum(S), maximum(S))
    @assert nc == nb                         "[B] cell list missed $(nb - nc) pairs"
    @assert maximum(S) - minimum(S) < 1e-10  "[B] S_smooth not uniform on a uniform lattice"
    println("  [B] ✓")
end

# ── C. random particles in many box shapes ──────────────────────────────────
println("\n[C] random particles, cell list vs brute force over box shapes (h=1)")
let h = 1.0, rng = Xoshiro(2024), nfail = 0
    for Lx in (6.0, 6.7, 8.9, 12.0), Ly in (6.0, 6.5, 8.2, 9.3, 12.0)
        L  = SVector(Lx, Ly)
        ps = [Particle(SVector(rand(rng) * Lx, rand(rng) * Ly), 1.0) for _ in 1:200]
        cl = CellList2D(L, h, length(ps)); rebuild_cell_list!(cl, ps)
        nb = length(pairs_brute(ps, h, L));  nc = length(pairs_celllist(ps, cl, h, L))
        ok = nb == nc;  ok || (nfail += 1)
        @printf("  Lx=%-5.1f Ly=%-5.1f  ncx=%d ncy=%d   brute %5d  cell list %5d  %s\n",
                Lx, Ly, cl.ncx, cl.ncy, nb, nc, ok ? "✓" : "✗ MISSING $(nb - nc)")
    end
    @assert nfail == 0 "[C] $nfail box shapes lose pairs"
    println("  [C] ✓")
end

println("\ntestcelllist_box.jl passed.")
