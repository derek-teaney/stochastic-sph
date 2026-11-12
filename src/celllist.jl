# celllist.jl — the cell list and the neighbour list (and threaded_chunks).

# With ncy==1 (1D mode), mod(anything, 1)==0 maps all three dy offsets to the same
# cell, triple-counting every particle.  Clamp the stencil to a single offset.
# With ncy==2 the offsets −1 and +1 both reach the other cell, double-counting it
# (thin strips, e.g. examples/shearwave_scan.jl): use the offsets 0 and 1.
neighbor_range(n::Int) = n == 1 ? (0:0) : n == 2 ? (0:1) : (-1:1)

mutable struct CellList2D
    cell_start :: Matrix{Int}       # (ncx, ncy)  start index into flat array
    cell_count :: Matrix{Int}       # (ncx, ncy)  particles currently in cell
    indices    :: Vector{Int}       # (N,)        flat contiguous particle index array
    ncx        :: Int               # cells in x direction
    ncy        :: Int               # cells in y direction
    cell_size  :: SVector{2,Float64} # (Lx/ncx, Ly/ncy), each ≥ 2h; cells tile the box exactly
    L          :: SVector{2,Float64}
    nl         :: Any                # NeighborList of the current positions, built on demand
end

function CellList2D(L::SVector{2,Float64}, h::Float64, N::Int)
    ncx = floor(Int, L[1] / (2h))
    ncy = max(1, floor(Int, L[2] / (2h)))
    @assert ncx >= 3 "Box too small in x: need ncx ≥ 3, got $ncx"
    # Separate widths in x and y.  A single width Lx/ncx reused in y does not tile
    # [0, Ly) when Ly/ncy ≠ Lx/ncx, and the periodic y-stencil then misses pairs
    # across y = 0 (test/testcelllist_box.jl).
    cell_size = SVector(L[1] / ncx, L[2] / ncy)
    CellList2D(zeros(Int,ncx,ncy), zeros(Int,ncx,ncy), zeros(Int,N), ncx, ncy, cell_size, L, nothing)
end

# Run body(range) over contiguous chunks of 1:N, one chunk per thread, so that each
# chunk can allocate its scratch buffers once.  Results do not depend on the chunking
# as long as body writes only the entries of its own range.
function threaded_chunks(body, N::Int)
    n = max(1, min(N, Threads.nthreads()))
    rs = [((c - 1) * N ÷ n + 1):(c * N ÷ n) for c in 1:n]
    Threads.@threads for c in 1:n
        body(rs[c])
    end
end

# ── Neighbour list ───────────────────────────────────────────────────────────
# For each particle a, the particles b (a itself included) with |x_b − x_a| < 2h(1 + 1e-9),
# in exactly the order of the cell scan (dx in −1:1, dy in neighbor_range(ncy), cell
# order), so that a pass over the list sums in the same order as the scan.  Each pass
# keeps its own exact distance test.  The list remembers the positions it was built from
# and is rebuilt whenever they differ (one comparison per particle per call).
mutable struct NeighborList
    offsets :: Vector{Int}                 # neighbours of a: index[offsets[a]:offsets[a+1]-1]
    index   :: Vector{Int}
    x       :: Vector{SVector{2,Float64}}  # positions at the build
    h       :: Float64
    valid   :: Bool
    parts   :: Vector{Vector{Int}}         # per-chunk buffers, reused between builds
    counts  :: Vector{Int}
end
NeighborList() = NeighborList(Int[], Int[], SVector{2,Float64}[], 0.0, false, Vector{Int}[], Int[])
neighbors(nl::NeighborList, a::Int) = @view nl.index[nl.offsets[a]:nl.offsets[a+1]-1]

function neighbor_list!(cl::CellList2D, particles::Vector{Particle}, h::Float64)
    cl.nl isa NeighborList || (cl.nl = NeighborList())
    nl = cl.nl::NeighborList;  N = length(particles)
    if nl.valid && nl.h == h && length(nl.x) == N && all(i -> nl.x[i] == particles[i].x, 1:N)
        return nl
    end
    cut2 = (2h * (1 + 1e-9))^2
    ncx = cl.ncx;  ncy = cl.ncy
    nch = max(1, min(N, Threads.nthreads()))
    length(nl.parts) == nch || (nl.parts = [Int[] for _ in 1:nch])
    resize!(nl.counts, N);  counts = nl.counts;  parts = nl.parts
    Threads.@threads for c in 1:nch
        idx = empty!(parts[c])
        for a in ((c - 1) * N ÷ nch + 1):(c * N ÷ nch)
            ix, iy = cell_index(particles[a].x, cl);  n0 = length(idx)
            for dx in -1:1, dy in neighbor_range(ncy)
                jx = mod(ix + dx - 1, ncx) + 1
                jy = mod(iy + dy - 1, ncy) + 1
                for b in particles_in_cell(cl, jx, jy)
                    r = minimum_image(particles[a].x - particles[b].x, cl.L)
                    r[1]^2 + r[2]^2 < cut2 && push!(idx, b)
                end
            end
            counts[a] = length(idx) - n0
        end
    end
    resize!(nl.offsets, N + 1);  nl.offsets[1] = 1
    for a in 1:N;  nl.offsets[a + 1] = nl.offsets[a] + counts[a];  end
    resize!(nl.index, nl.offsets[N + 1] - 1);  k = 0
    for c in 1:nch, b in parts[c];  k += 1;  nl.index[k] = b;  end
    resize!(nl.x, N)
    for a in 1:N;  nl.x[a] = particles[a].x;  end
    nl.h = h;  nl.valid = true
    return nl
end

function cell_index(x::SVector{2,Float64}, cl::CellList2D)
    ix = mod(floor(Int, x[1] / cl.cell_size[1]), cl.ncx) + 1
    iy = mod(floor(Int, x[2] / cl.cell_size[2]), cl.ncy) + 1
    return ix, iy
end

function rebuild_cell_list!(cl::CellList2D, particles::Vector{Particle})
    N   = length(particles)
    ncx = cl.ncx
    ncy = cl.ncy

    # pass 1: count particles per cell
    cl.cell_count .= 0
    for a in 1:N
        ix, iy = cell_index(particles[a].x, cl)
        cl.cell_count[ix, iy] += 1
    end

    # prefix sum → start indices (column-major order: ix fast)
    idx = 1
    for iy in 1:ncy, ix in 1:ncx
        cl.cell_start[ix, iy] = idx
        idx += cl.cell_count[ix, iy]
    end

    # pass 2: fill flat index array (reuse cell_count as fill counter)
    cl.cell_count .= 0
    for a in 1:N
        ix, iy = cell_index(particles[a].x, cl)
        offset = cl.cell_count[ix, iy]
        cl.indices[cl.cell_start[ix, iy] + offset] = a
        cl.cell_count[ix, iy] += 1
    end
end

function particles_in_cell(cl::CellList2D, ix::Int, iy::Int)
    start = cl.cell_start[ix, iy]
    count = cl.cell_count[ix, iy]
    return @view cl.indices[start : start + count - 1]
end

function wrap_positions!(particles::Vector{Particle}, L::SVector{2,Float64})
    Threads.@threads for a in eachindex(particles)
        particles[a].x = mod.(particles[a].x, L)
    end
end
