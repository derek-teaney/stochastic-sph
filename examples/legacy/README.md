# examples/legacy

Analysis scripts of the Legendre scheme (`SPH_RENORM=legendre`) and of the entropy shift (`SPH_RENORM=1`), both removed from the code
on 2026-11-26 (`docs/sscheme_plan.tex`, sec:prune).  They read the CSV files and run outputs of their campaigns and draw the figures of
`docs/legendre_plan.tex` and `docs/step5_plan.tex`; they do not load `src/`.  To rerun the Julia side of those campaigns, check out
commit 9846a6c (branch `sscheme-measure`), the last one with both schemes.

- `eos_legendre.py`: campaign `eos_legendre`, the figures `legendre_EP.png`, `legendre_moving.png`, `legendre_chi.png`.
- `eos_compare.py`: the side-by-side of `eos_legendre` and `eos_split` (`step5_plan.tex`).
- `renorm_quadrature_legendre.py`: the single particle of the Legendre Hamiltonian (also imported by `examples/renorm_quadrature_split.py`).
