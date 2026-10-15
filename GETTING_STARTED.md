# Getting started

Everything you change stays on your own computer until you push, so feel free
to experiment.

## Layout

- `src/`: the library. Shared code we keep.
- `examples/`: physics runs (sound waves, shear waves, Gaussian pulse,
  stochastic background). These are the scripts to run and play with.
- `test/`: checks that pass or fail.
- `docs/`: the notes behind the code (`sph2.pdf`, `firstrun.pdf`).

## Setup (once)

You need Julia 1.12 and a Python with matplotlib (used by PyPlot).

```sh
git clone https://github.com/derek-teaney/stochastic-sph.git
cd stochastic-sph
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Play

Make a scratch branch so you can throw everything away afterwards:

```sh
git switch -c play
cd examples
julia --project testsound_viscous.jl      # or testdiffuse.jl, testgaussian.jl, ...
```

Plots are written next to the script. Open the script, change the parameters
near the bottom (`h`, `h/a`, `η/s`, amplitude, run time), and run it again.
You can edit anything, including `src/`.

`--project` makes Julia use the package versions pinned in the repository.
The scripts find `src/` relative to their own location, so they work from any
directory.

## When you're done

To discard everything:

```sh
git restore .            # undo edits to tracked files
git switch main
git branch -D play
```

To keep something, commit it on your branch and tell us. If it should go into
the repository, open a pull request.

## Tips

- Running inside one Julia session avoids recompiling every time. Install
  [Revise](https://github.com/timholy/Revise.jl) in your global environment,
  start `julia --project`, then `using Revise` and
  `includet("examples/testsound_viscous.jl")`. Edits to that script are then
  picked up when you run it again.
- Before `git pull` on `main`, put any uncommitted edits aside with `git stash`.
