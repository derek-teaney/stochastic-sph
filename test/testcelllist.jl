using PyPlot
include(joinpath(@__DIR__, "..", "src", "sph.jl"))
include(joinpath(@__DIR__, "..", "src", "celllist.jl"))

# ── Reusable helpers ───────────────────────────────────────────────────────────

function print_cell_list(cl::CellList2D, particles::Vector{Particle})
    println("ncx = $(cl.ncx)  ncy = $(cl.ncy)  cell_size = $(cl.cell_size)")
    println("\ncell_count  [ix=row, iy=col]:")
    display(cl.cell_count)
    println("\ncell_start  [ix=row, iy=col]:")
    display(cl.cell_start)
    println("\nflat indices: ", cl.indices[1:length(particles)])
    println("\nOccupied cells:")
    for iy in cl.ncy:-1:1, ix in 1:cl.ncx
        ids = collect(particles_in_cell(cl, ix, iy))
        isempty(ids) || println("  cell ($ix,$iy) → particle indices $ids")
    end
end

function plot_cell_list(cl::CellList2D, particles::Vector{Particle};
                        title_str::String = "CellList2D",
                        filename::Union{String,Nothing} = nothing)
    fig, ax = subplots(figsize=(5, 5))

    # shade occupied cells
    for iy in 1:cl.ncy, ix in 1:cl.ncx
        n = cl.cell_count[ix, iy]
        if n > 0
            rect = matplotlib.patches.Rectangle(
                ((ix-1)*cl.cell_size, (iy-1)*cl.cell_size),
                cl.cell_size, cl.cell_size,
                facecolor="lightsteelblue", edgecolor="none", alpha=0.45)
            ax.add_patch(rect)
        end
    end

    # grid lines
    for v in 0:cl.cell_size:cl.L[1]; ax.axvline(v, color="k", lw=1.0); end
    for v in 0:cl.cell_size:cl.L[2]; ax.axhline(v, color="k", lw=1.0); end

    # cell labels
    for iy in 1:cl.ncy, ix in 1:cl.ncx
        cx = (ix - 0.5) * cl.cell_size
        cy = (iy - 0.5) * cl.cell_size
        n  = cl.cell_count[ix, iy]
        ax.text(cx, cy + 0.35, "($ix,$iy)", ha="center", va="center",
                fontsize=8, color="gray")
        ax.text(cx, cy - 0.25, "n = $n", ha="center", va="center",
                fontsize=11, fontweight = n > 0 ? "bold" : "normal",
                color = n > 0 ? "navy" : "silver")
    end

    # particles — auto-range axes to include out-of-domain positions
    pcolors = ["tab:red","tab:blue","tab:green","tab:orange","tab:purple",
               "tab:brown","tab:pink","tab:cyan"]
    for (k, p) in enumerate(particles)
        c = pcolors[mod1(k, length(pcolors))]
        ax.scatter(p.x[1], p.x[2], s=140, color=c, zorder=5)
        ax.annotate("  P$k", xy=(p.x[1], p.x[2]), fontsize=9, color=c)
    end

    all_x = [p.x[1] for p in particles]
    all_y = [p.x[2] for p in particles]
    ax.set_xlim(min(0.0, minimum(all_x)) - 0.3, max(cl.L[1], maximum(all_x)) + 0.3)
    ax.set_ylim(min(0.0, minimum(all_y)) - 0.3, max(cl.L[2], maximum(all_y)) + 0.3)
    ax.set_xlabel("x");  ax.set_ylabel("y")
    ax.set_title(title_str)
    ax.set_aspect("equal")
    tight_layout()

    filename !== nothing && savefig(filename, dpi=150)
    return fig
end

# ── Test 1: 3×3 grid, 5 particles in 5 different cells ────────────────────────

println("=" ^ 50)
println("Test 1: 5 particles, all in different cells")
println("=" ^ 50)

L, h = 6.0, 1.0
particles1 = [
    Particle(SVector(1.0, 1.0), 1.0),   # cell (1,1)
    Particle(SVector(3.5, 4.5), 1.0),   # cell (2,3)
    Particle(SVector(5.0, 1.5), 1.0),   # cell (3,1)
    Particle(SVector(0.8, 3.5), 1.0),   # cell (1,2)
    Particle(SVector(2.5, 0.8), 1.0),   # cell (2,1)
]
cl1 = CellList2D(SVector(L,L), h, length(particles1))
rebuild_cell_list!(cl1, particles1)
print_cell_list(cl1, particles1)
plot_cell_list(cl1, particles1;
               title_str = "Test 1: ncx=$(cl1.ncx) ncy=$(cl1.ncy), $(length(particles1)) particles",
               filename  = joinpath(@__DIR__, "testcelllist1.png"))

# ── Test 2: multiple particles in same cell ────────────────────────────────────

println("\n", "=" ^ 50)
println("Test 2: 3 in cell (2,2), 2 in cell (1,3)")
println("=" ^ 50)

particles2 = [
    Particle(SVector(2.2, 2.1), 1.0),   # cell (2,2)
    Particle(SVector(2.8, 2.9), 1.0),   # cell (2,2)
    Particle(SVector(3.1, 2.5), 1.0),   # cell (2,2)
    Particle(SVector(0.5, 4.5), 1.0),   # cell (1,3)
    Particle(SVector(0.9, 5.0), 1.0),   # cell (1,3)
]
cl2 = CellList2D(SVector(L,L), h, length(particles2))
rebuild_cell_list!(cl2, particles2)
print_cell_list(cl2, particles2)
println("\nVerification:")
println("  particles_in_cell(2,2) = ", collect(particles_in_cell(cl2, 2, 2)))
println("  particles_in_cell(1,3) = ", collect(particles_in_cell(cl2, 1, 3)))
plot_cell_list(cl2, particles2;
               title_str = "Test 2: 3 in (2,2), 2 in (1,3)",
               filename  = joinpath(@__DIR__, "testcelllist2.png"))

# ── Test 3: out-of-range positions + wrap ─────────────────────────────────────

println("\n", "=" ^ 50)
println("Test 3A: particles outside [0,L) before wrap")
println("=" ^ 50)

particles3 = [
    Particle(SVector( 1.0, 1.0), 1.0),   # inside        → cell (1,1)
    Particle(SVector( 6.5, 2.5), 1.0),   # x > L, wraps  → cell (1,2)
    Particle(SVector(-0.5, 4.0), 1.0),   # x < 0, wraps  → cell (3,3)
]
cl3 = CellList2D(SVector(L,L), h, length(particles3))
rebuild_cell_list!(cl3, particles3)
println("Positions before wrap:")
for (k, p) in enumerate(particles3)
    ix, iy = cell_index(p.x, cl3)
    println("  P$k at $(p.x) → cell ($ix,$iy)")
end
print_cell_list(cl3, particles3)
plot_cell_list(cl3, particles3;
               title_str = "Test 3A: out-of-range positions (before wrap)",
               filename  = joinpath(@__DIR__, "testcelllist3a.png"))

println("\n", "=" ^ 50)
println("Test 3B: after wrap_positions!")
println("=" ^ 50)

wrap_positions!(particles3, SVector(L,L))
rebuild_cell_list!(cl3, particles3)
println("Positions after wrap:")
for (k, p) in enumerate(particles3)
    ix, iy = cell_index(p.x, cl3)
    println("  P$k at $(p.x) → cell ($ix,$iy)")
end
print_cell_list(cl3, particles3)
plot_cell_list(cl3, particles3;
               title_str = "Test 3B: after wrap_positions!",
               filename  = joinpath(@__DIR__, "testcelllist3b.png"))

# ── Test 4: corner wrapping — off-grid in both x and y ────────────────────────

println("\n", "=" ^ 50)
println("Test 4A: particles off-grid in all four corners")
println("=" ^ 50)

# With L=6, h=1 → nc=3, cell_size=2
# Four off-corner particles + one inside for reference
# Expected cells after wrapping:
#   P1 ( 7.0,  7.0) → wraps to (1.0, 1.0) → cell (1,1)  upper-right corner
#   P2 (-0.5, -0.5) → wraps to (5.5, 5.5) → cell (3,3)  lower-left corner
#   P3 ( 6.8, -0.3) → wraps to (0.8, 5.7) → cell (1,3)  lower-right corner
#   P4 (-0.2,  6.7) → wraps to (5.8, 0.7) → cell (3,1)  upper-left corner
#   P5 ( 3.0,  3.0) → inside               → cell (2,2)  centre reference

particles4 = [
    Particle(SVector( 7.0,  7.0), 1.0),   # x>L, y>L  → cell (1,1)
    Particle(SVector(-0.5, -0.5), 1.0),   # x<0, y<0  → cell (3,3)
    Particle(SVector( 6.8, -0.3), 1.0),   # x>L, y<0  → cell (1,3)
    Particle(SVector(-0.2,  6.7), 1.0),   # x<0, y>L  → cell (3,1)
    Particle(SVector( 3.0,  3.0), 1.0),   # inside     → cell (2,2)
]
cl4 = CellList2D(SVector(L,L), h, length(particles4))
rebuild_cell_list!(cl4, particles4)
println("Positions before wrap:")
for (k, p) in enumerate(particles4)
    ix, iy = cell_index(p.x, cl4)
    println("  P$k at $(p.x) → cell ($ix,$iy)")
end
print_cell_list(cl4, particles4)
plot_cell_list(cl4, particles4;
               title_str = "Test 4A: four corners off-grid (before wrap)",
               filename  = joinpath(@__DIR__, "testcelllist4a.png"))

println("\n", "=" ^ 50)
println("Test 4B: after wrap_positions!")
println("=" ^ 50)

wrap_positions!(particles4, SVector(L,L))
rebuild_cell_list!(cl4, particles4)
println("Positions after wrap:")
for (k, p) in enumerate(particles4)
    ix, iy = cell_index(p.x, cl4)
    println("  P$k at $(p.x) → cell ($ix,$iy)")
end
print_cell_list(cl4, particles4)
plot_cell_list(cl4, particles4;
               title_str = "Test 4B: four corners after wrap_positions!",
               filename  = joinpath(@__DIR__, "testcelllist4b.png"))

show()
