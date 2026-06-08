# EM for Latent Object Localization in Noisy Images

Project page: https://vskates.github.io/em-latent-object-localization/

This repository contains a reproducible implementation and analysis of soft EM and hard EM for a probabilistic graphical model with latent object locations. Given heavily corrupted grayscale images containing a shared static background and a small object at unknown coordinates, the model recovers the object, background, noise level, and location prior.

## Highlights

- Closed-form derivation of the E-step, M-step, and EM lower bound.
- Numerically stable posterior computation in log space.
- Multi-start EM with lower-bound based model selection.
- Synthetic experiments over noise level, sample size, and initialization.
- Comparison of uncertainty-aware soft EM with MAP-style hard EM.
- GitHub Pages write-up in `docs/` for a research-style work sample.

This project began as coursework on EM for noisy image reconstruction. I extended it with additional experiments, reproducibility tooling, visualizations, failure-mode analysis, and a research-style write-up.

## Reproduce

```bash
python3 run_tests.py
python3 run_experiments.py
python3 data2_analysis.py
python3 clip_effect_analysis.py
pdflatex -interaction=nonstopmode REPORT.tex
```

The static project page is served from `docs/`.
