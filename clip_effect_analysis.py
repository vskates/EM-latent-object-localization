from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

import Student as model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA2_FALLBACK = Path("/home/kate/Downloads/Telegram Desktop/data2.npy")
FACE_H = 100
FACE_W = 67


def resolve_data2_path():
    local = ROOT / "data2.npy"
    if local.exists():
        return local
    if DATA2_FALLBACK.exists():
        return DATA2_FALLBACK
    raise FileNotFoundError("data2.npy not found")


def run_curve(base, noise_levels, clip):
    rng = np.random.default_rng(0)
    rows = []
    for extra_noise in noise_levels:
        noisy = base + rng.normal(0.0, extra_noise, size=base.shape)
        if clip:
            noisy = np.clip(noisy, 0.0, 255.0)
        F0, B0, s0, A0 = model._init_parameters(noisy, FACE_H, FACE_W, seed=0)
        F, B, s, A, ll = model.run_EM(
            noisy,
            FACE_H,
            FACE_W,
            F=F0,
            B=B0,
            s=s0,
            A=A0,
            tolerance=-1.0,
            max_iter=3,
        )
        rows.append(
            {
                "extra_noise": float(extra_noise),
                "elbo_norm": float(ll[-1] / np.prod(noisy.shape)),
                "sigma_est": float(s),
                "dataset_std": float(noisy.std()),
            }
        )
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    data_path = resolve_data2_path()
    base = np.asarray(np.load(data_path, mmap_mode="r")[:, :, :20], dtype=np.float64)
    noise_levels = [0, 5, 10, 20, 40]

    clipped = run_curve(base, noise_levels, clip=True)
    noclip = run_curve(base, noise_levels, clip=False)

    with open(RESULTS / "data2_clip_effect.json", "w", encoding="utf-8") as f:
        json.dump({"clip": clipped, "no_clip": noclip}, f, indent=2)

    x = [row["extra_noise"] for row in clipped]
    y_clip = [row["elbo_norm"] for row in clipped]
    y_noclip = [row["elbo_norm"] for row in noclip]
    s_clip = [row["sigma_est"] for row in clipped]
    s_noclip = [row["sigma_est"] for row in noclip]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(x, y_clip, marker="o", linewidth=2, label="with clip")
    axes[0].plot(x, y_noclip, marker="o", linewidth=2, label="without clip")
    axes[0].set_title("ELBO vs noise")
    axes[0].set_xlabel("added Gaussian noise std")
    axes[0].set_ylabel("ELBO / (H W K)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, s_clip, marker="o", linewidth=2, label="with clip")
    axes[1].plot(x, s_noclip, marker="o", linewidth=2, label="without clip")
    axes[1].set_title("Estimated sigma vs noise")
    axes[1].set_xlabel("added Gaussian noise std")
    axes[1].set_ylabel("estimated sigma")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(RESULTS / "data2_clip_vs_noclip.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
