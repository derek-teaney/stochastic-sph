using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))

# ── Core pair-finding functions ────────────────────────────────────────────────

function find_pairs_brute(particles::Vector{Particle}, h::Float64, L::SVector{2,Float64})
    N = length(particles)
    pairs = Tuple{Int,Int,Float64}[]
    for a in 1:N, b in a+1:N
        r_mic = minimum_image(particles[a].x - particles[b].x, L)
        r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
        r_mag < 2h && push!(pairs, (a, b, r_mag))
    end
    return pairs
end

function find_pairs_celllist(particles::Vector{Particle}, cl::CellList2D,
                              h::Float64, L::SVector{2,Float64})
    N   = length(particles)
    ncx = cl.ncx;  ncy = cl.ncy
    pairs = Tuple{Int,Int,Float64}[]
    for a in 1:N
        ix, iy = cell_index(particles[a].x, cl)
        for dx in -1:1, dy in neighbor_range(ncy)
            jx = mod(ix + dx - 1, ncx) + 1
            jy = mod(iy + dy - 1, ncy) + 1
            for b in particles_in_cell(cl, jx, jy)
                b <= a && continue
                r_mic = minimum_image(particles[a].x - particles[b].x, L)
                r_mag = sqrt(r_mic[1]^2 + r_mic[2]^2)
                r_mag < 2h && push!(pairs, (a, b, r_mag))
            end
        end
    end
    return pairs
end

# ── Reusable helpers ───────────────────────────────────────────────────────────

function check_pairs(brute::Vector{Tuple{Int,Int,Float64}},
                     celllist::Vector{Tuple{Int,Int,Float64}})
    sort!(brute,     by = p -> (p[1], p[2]))
    sort!(celllist,  by = p -> (p[1], p[2]))

    brute_set    = Set((a, b) for (a, b, r) in brute)
    celllist_set = Set((a, b) for (a, b, r) in celllist)
    missing_pairs  = setdiff(brute_set, celllist_set)
    spurious_pairs = setdiff(celllist_set, brute_set)

    if isempty(missing_pairs) && isempty(spurious_pairs)
        println("✓ PASS: $(length(brute)) pairs match")
    else
        println("✗ FAIL: $(length(missing_pairs)) missing, $(length(spurious_pairs)) spurious")
        isempty(missing_pairs)  || println("  Missing:  $missing_pairs")
        isempty(spurious_pairs) || println("  Spurious: $spurious_pairs")
    end
    println("Pairs:")
    for (a, b, r) in brute
        println("  ($a,$b)  r = $(round(r, digits=4))")
    end
end

function pair_color(r::Float64, h::Float64)
    r < h       && return "tab:green"
    r < 1.5h    && return "tab:orange"
    return "tab:red"
end

function plot_neighbor_pairs(particles::Vector{Particle},
                              pairs::Vector{Tuple{Int,Int,Float64}},
                              cl::CellList2D;
                              title_str::String = "Neighbor pairs",
                              filename::Union{String,Nothing} = nothing)
    L  = cl.L
    h  = cl.cell_size / 2.0   # cell_size ≈ 2h
    fig, ax = subplots(figsize=(6, 6))

    # grid
    for v in 0:cl.cell_size:cl.L[1]; ax.axvline(v, color="0.80", lw=0.8); end
    for v in 0:cl.cell_size:cl.L[2]; ax.axhline(v, color="0.80", lw=0.8); end

    # neighbor pair lines — draw from pa to pa - r_mic (MIC endpoint)
    for (a, b, r) in pairs
        pa  = particles[a].x
        r_mic = minimum_image(pa - particles[b].x, L)
        pb_mic = pa - r_mic          # MIC position of b as seen from a
        ax.plot([pa[1], pb_mic[1]], [pa[2], pb_mic[2]],
                color=pair_color(r, h), lw=1.2, zorder=2)
    end

    # particles
    pcolors = ["tab:red","tab:blue","tab:green","tab:orange","tab:purple",
               "tab:brown","tab:pink","tab:cyan","tab:gray","tab:olive"]
    for (k, p) in enumerate(particles)
        c = pcolors[mod1(k, length(pcolors))]
        ax.scatter(p.x[1], p.x[2], s=80, color=c, zorder=5)
        ax.annotate(" $k", xy=(p.x[1], p.x[2]), fontsize=7, color=c)
    end

    # auto-range axes to show periodic line endpoints outside [0,L]
    all_x = Float64[p.x[1] for p in particles]
    all_y = Float64[p.x[2] for p in particles]
    for (a, b, _) in pairs
        pa    = particles[a].x
        r_mic = minimum_image(pa - particles[b].x, L)
        pb_mic = pa - r_mic
        push!(all_x, pb_mic[1]);  push!(all_y, pb_mic[2])
    end
    margin = 0.3
    ax.set_xlim(minimum(all_x) - margin, maximum(all_x) + margin)
    ax.set_ylim(minimum(all_y) - margin, maximum(all_y) + margin)

    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_title(title_str)
    ax.set_aspect("equal")
    tight_layout()
    filename !== nothing && savefig(filename, dpi=150)
    return fig
end

# ── Test 1: 20 random particles, 5×5 grid ─────────────────────────────────────

println("=" ^ 55)
println("Test 1: 20 random particles,  L=10  h=1  (5×5 grid)")
println("=" ^ 55)

L, h = 10.0, 1.0
Random.seed!(RNG, 42)
particles1 = [Particle(SVector(rand(RNG)*L, rand(RNG)*L), 1.0) for _ in 1:20]

cl1 = CellList2D(SVector(L,L), h, length(particles1))
rebuild_cell_list!(cl1, particles1)

brute1    = find_pairs_brute(particles1, h, SVector(L,L))
celllist1 = find_pairs_celllist(particles1, cl1, h, SVector(L,L))
check_pairs(brute1, celllist1)
plot_neighbor_pairs(particles1, brute1, cl1;
                    title_str = "Test 1: 20 random particles (5×5 grid)",
                    filename  = joinpath(@__DIR__, "testneighbor1.png"))

# ── Test 2: forced periodic pairs ─────────────────────────────────────────────

println("\n", "=" ^ 55)
println("Test 2: forced periodic pairs across x and xy boundaries")
println("=" ^ 55)

# P1-P2: straddle x boundary  (MIC dist ≈ 0.5, direct dist ≈ 9.5)
# P3-P4: straddle xy corner   (MIC dist ≈ 0.42, direct dist ≈ 13.4)
# P5:    central reference, isolated
particles2 = [
    Particle(SVector(0.3, 5.0), 1.0),   # P1 }  x-periodic pair, MIC r ≈ 0.50
    Particle(SVector(9.8, 5.0), 1.0),   # P2 }
    Particle(SVector(0.2, 0.2), 1.0),   # P3 }  xy-corner pair, MIC r ≈ 0.42
    Particle(SVector(9.9, 9.9), 1.0),   # P4 }
    Particle(SVector(5.0, 5.0), 1.0),   # P5    isolated centre
]

cl2 = CellList2D(SVector(L,L), h, length(particles2))
rebuild_cell_list!(cl2, particles2)

brute2    = find_pairs_brute(particles2, h, SVector(L,L))
celllist2 = find_pairs_celllist(particles2, cl2, h, SVector(L,L))
check_pairs(brute2, celllist2)
plot_neighbor_pairs(particles2, brute2, cl2;
                    title_str = "Test 2: periodic pairs (x-boundary and xy-corner)",
                    filename  = joinpath(@__DIR__, "testneighbor2.png"))

# ── Test 3: asymmetric box  Lx=12, Ly=6 (6×3 grid) ──────────────────────────

println("\n", "=" ^ 55)
println("Test 3: asymmetric box  Lx=12  Ly=6  h=1  (6×3 grid)")
println("=" ^ 55)

# Discriminating pairs:
#   P1-P2: straddle y-boundary  MIC r ≈ 0.60  (missed if y MIC uses Lx=12)
#   P3-P4: straddle x-boundary  MIC r ≈ 0.50
#   P5-P6: straddle xy-corner   MIC r ≈ 0.71
#   P7:    isolated centre
Lx3, Ly3, h3 = 12.0, 6.0, 1.0
L3 = SVector(Lx3, Ly3)

particles3 = [
    Particle(SVector(6.0,  0.3),  1.0),   # P1 }  y-periodic pair
    Particle(SVector(6.0,  5.7),  1.0),   # P2 }  MIC r ≈ 0.60
    Particle(SVector(0.25, 3.0),  1.0),   # P3 }  x-periodic pair
    Particle(SVector(11.75,3.0),  1.0),   # P4 }  MIC r ≈ 0.50
    Particle(SVector(0.5,  0.5),  1.0),   # P5 }  xy-corner pair
    Particle(SVector(11.5, 5.5),  1.0),   # P6 }  MIC r ≈ 0.71
    Particle(SVector(6.0,  3.0),  1.0),   # P7    isolated centre
]

cl3 = CellList2D(L3, h3, length(particles3))
rebuild_cell_list!(cl3, particles3)
@printf("ncx=%d  ncy=%d  cell_size=%.4g\n", cl3.ncx, cl3.ncy, cl3.cell_size)

brute3    = find_pairs_brute(particles3, h3, L3)
celllist3 = find_pairs_celllist(particles3, cl3, h3, L3)
check_pairs(brute3, celllist3)
plot_neighbor_pairs(particles3, brute3, cl3;
                    title_str = "Test 3: asymmetric box Lx=12 Ly=6 (6×3 grid)",
                    filename  = joinpath(@__DIR__, "testneighbor3.png"))

# ── Test 4: 20 random particles, 3×6 grid ─────────────────────────────────────

println("\n", "=" ^ 55)
println("Test 4: 20 random particles,  Lx=6  Ly=12  h=1  (3×6 grid)")
println("=" ^ 55)

Lx4, Ly4, h4 = 6.0, 12.0, 1.0
L4 = SVector(Lx4, Ly4)
Random.seed!(RNG, 99)
particles4 = [Particle(SVector(rand(RNG)*Lx4, rand(RNG)*Ly4), 1.0) for _ in 1:20]

cl4 = CellList2D(L4, h4, length(particles4))
rebuild_cell_list!(cl4, particles4)
@printf("ncx=%d  ncy=%d  cell_size=%.4g\n", cl4.ncx, cl4.ncy, cl4.cell_size)

brute4    = find_pairs_brute(particles4, h4, L4)
celllist4 = find_pairs_celllist(particles4, cl4, h4, L4)
check_pairs(brute4, celllist4)
plot_neighbor_pairs(particles4, brute4, cl4;
                    title_str = "Test 4: 20 random particles (3×6 grid, Lx=6 Ly=12)",
                    filename  = joinpath(@__DIR__, "testneighbor4.png"))

show()
