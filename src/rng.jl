# ── Physics random-number streams ────────────────────────────────────────────
# One independent Xoshiro per particle.  Every stochastic draw for particle a —
# the noise ξ_a and the Metropolis accept in particle_sweep! — comes from
# rngs[a].  Because the stream is tied to the particle index, the sweep result
# is independent of visit order and of how particles are chunked across threads;
# the streams simply advance from one sweep to the next.
#
# Size and seed the default vector once per run, before the step loop:
#
#     seed_phys!(params.N, params.seed)
#
# The per-particle streams are seeded from one master Xoshiro(seed), so their
# initial states are well separated.  A checkpoint must save the Xoshiro states
# (they are not a pure function of the step count).
using Random

# A fresh per-particle stream vector — does not touch global state.
function make_phys_rng(N::Integer, seed::Integer)
    master = Xoshiro(seed)
    return Xoshiro[Xoshiro(rand(master, UInt64)) for _ in 1:N]
end

# The default stream vector used by particle_sweep!.
const PHYS_RNG = Xoshiro[]

function seed_phys!(N::Integer, seed::Integer)
    resize!(PHYS_RNG, N)
    PHYS_RNG .= make_phys_rng(N, seed)
    return PHYS_RNG
end
