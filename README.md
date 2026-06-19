# EM for Latent Object Localization in Noisy Images

<table>
<tr>
<td bgcolor="#f6f8fa">
<b>Contribution.</b> Latent object localization in heavily corrupted images is considered under a probabilistic latent-variable model with unknown object position, background, object template, noise level, and location prior. Soft EM and hard EM are studied in the same setting, with the central result that uncertainty-aware soft EM yields more stable reconstruction and model fitting than MAP-style hard EM under severe noise and ambiguous localization. On synthetic and real noisy images, accurate foreground/background recovery and robust latent-location estimation are obtained.<br><br>
<b>Author:</b> Ekaterina Vasiagina<br>
<b>Project page:</b> <a href="https://vskates.github.io/em-latent-object-localization/">Project page</a>
</td>
</tr>
</table>

## Method

A collection of noisy grayscale images is modeled with a shared background, a foreground object, and a latent object location for each image.

```math
p(X, z \mid F, B, \pi, \sigma^2)
=
p(z \mid \pi)\, p(X \mid z, F, B, \sigma^2)
```

Soft EM uses the full posterior over latent locations:

```math
q_i(z) = p(z \mid X_i, \theta)
```

and updates parameters by maximizing the expected complete-data objective:

```math
\mathcal{Q}(\theta;\theta^{old})
=
\sum_i \sum_z q_i(z)\,\log p(X_i, z \mid \theta)
```

Hard EM replaces the posterior with a single MAP estimate:

```math
z_i^\star = \arg\max_z p(z \mid X_i, \theta)
```

The implementation includes numerically stable posterior computation, closed-form M-step updates, multi-start initialization, and lower-bound based model selection.

## Main Result

Under strong corruption, soft EM recovers the foreground object, background, and latent locations more reliably than hard EM, especially when localization uncertainty is high.

## Repository

```text
Student.py                core EM implementation
run_tests.py              correctness checks
run_experiments.py        synthetic experiments
data2_analysis.py         real-data analysis
clip_effect_analysis.py   clipping/no-clipping analysis
results/                  figures and summaries
docs/                     static project page
REPORT.tex                report source
```
