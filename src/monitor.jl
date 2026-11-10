# monitor.jl — running statistics of a stochastic run, printed to the log (stdout).
#
# A StepMonitor is fed after every stochastic step (observe_step!, with the NamedTuple that
# stochrep_step! returns) and, at the record points, with the particles (observe_particles!).
# report(mon, t) prints one line for the window since the last report and the totals;
# monitor_summary(mon) returns the totals as a NamedTuple (for the run metadata); print_summary
# writes a block at the end of the run.  It is plain data, so it goes into checkpoints.
#
# Recorded:
#   acceptance of the stress (star) and residual proposals, as fractions of all proposals
#     (one per particle per sweep);
#   the fraction of each kind rejected by the guard: a proposal that would leave some particle
#     in a state no fluid can have — for the conformal EOS exactly |v| ≥ 1;
#   the particle speeds at the record points: the largest |v|, and the fractions of particles
#     with |v| > 0.9 and > 0.99.
# Include after sph.jl.

mutable struct StepMonitor
    N        :: Int
    tot      :: Vector{Float64}   # sums over steps: acc_s, acc_r, grej_s, grej_r
    win      :: Vector{Float64}   # the same since the last report
    nsteps   :: Int
    nwin     :: Int
    vmax     :: Float64           # largest |v| over all record points
    vmaxwin  :: Float64
    nfast    :: Vector{Int}       # particle samples with |v| > 0.9, > 0.99
    nsamp    :: Int               # particle samples
    wall0    :: Float64
end
StepMonitor(N::Integer) = StepMonitor(N, zeros(4), zeros(4), 0, 0, 0.0, 0.0, [0, 0], 0, time())

function observe_step!(mon::StepMonitor, a)
    v = (a.stress, isnan(a.resid) ? 0.0 : a.resid, get(a, :gstress, 0.0), get(a, :gresid, 0.0))
    mon.tot .+= v;  mon.win .+= v;  mon.nsteps += 1;  mon.nwin += 1
    return mon
end

function observe_particles!(mon::StepMonitor, particles)
    for q in particles
        s = norm(q.v_over_S * q.S_smooth)
        mon.vmax = max(mon.vmax, s);  mon.vmaxwin = max(mon.vmaxwin, s)
        s > 0.9 && (mon.nfast[1] += 1);  s > 0.99 && (mon.nfast[2] += 1)
    end
    mon.nsamp += length(particles)
    return mon
end

# one log line: the window since the last report, then the totals
function report(mon::StepMonitor, t; io = stdout, label = "")
    w = mon.win ./ max(mon.nwin, 1);  T = mon.tot ./ max(mon.nsteps, 1)
    @printf(io, "%s t=%-8.3g steps=%-7d wall=%-7.0fs | window: acc stress %.4f resid %.4f  |v|≥1 rejected: stress %.2e resid %.2e  vmax %.3f | total: %.4f %.4f %.2e %.2e  vmax %.3f  |v|>0.9: %.2e\n",
            label, t, mon.nsteps, time() - mon.wall0, w..., mon.vmaxwin, T..., mon.vmax, mon.nfast[1] / max(mon.nsamp, 1))
    flush(io)
    mon.win .= 0;  mon.nwin = 0;  mon.vmaxwin = 0.0
    return mon
end

monitor_summary(mon::StepMonitor) = (acc_stress = mon.tot[1] / max(mon.nsteps, 1), acc_resid = mon.tot[2] / max(mon.nsteps, 1),
                             vrej_stress = mon.tot[3] / max(mon.nsteps, 1), vrej_resid = mon.tot[4] / max(mon.nsteps, 1),
                             vmax = mon.vmax, frac_v09 = mon.nfast[1] / max(mon.nsamp, 1),
                             frac_v099 = mon.nfast[2] / max(mon.nsamp, 1), nsteps = mon.nsteps)

function print_summary(mon::StepMonitor; io = stdout, label = "")
    s = monitor_summary(mon)
    println(io, "── monitor summary $label ──")
    @printf(io, "  steps %d, %d particles, %d proposals per sweep\n", s.nsteps, mon.N, mon.N)
    @printf(io, "  acceptance:              stress %.5f   residual %.5f\n", s.acc_stress, s.acc_resid)
    @printf(io, "  rejected for |v| ≥ 1:    stress %.3e   residual %.3e   (fraction of proposals)\n", s.vrej_stress, s.vrej_resid)
    @printf(io, "  particle speeds:         max %.4f   |v| > 0.9: %.3e   |v| > 0.99: %.3e   (fraction of samples)\n",
            s.vmax, s.frac_v09, s.frac_v099)
    flush(io)
    return s
end
