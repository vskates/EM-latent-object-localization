# Data2 Analysis

- source: `/home/kate/Downloads/Telegram Desktop/data2.npy`
- face size: `h=100`, `w=67`
- ELBO iterations: `[-2565796.678, -2565796.678, -2565796.678, -2565796.678, -2565796.678, -2565796.678]`
- observation: on this dataset ELBO becomes flat almost immediately, which means the posterior over shifts is very peaked and EM reaches a fixed point after the first update.
- noise study: when added noise grows from `0.0` to `40.0`, face contrast drops from `43.793` to `39.161`.
- normalized ELBO changes from `-6.10904` to `-6.02176`; its trend is less interpretable by itself because the model can partially absorb extra noise through the estimated sigma.
