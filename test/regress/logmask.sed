# logmask.sed — renames applied to the reference's test output before it is compared
# (test/run.sh regress), so that renaming a test or a printed label is not a difference.
# Remove the lines once the reference commit has the new names.
s/dissipation=changerep/dissipation=particle/g
s/teststochrep_guard/teststochastic_guard/g
s/(^|[^a-z_])teststochastic\.(jl|log)/\1testparticle_sweep.\2/g
