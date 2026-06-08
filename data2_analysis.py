from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import Student as model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FACE_H = 100
FACE_W = 67
DATA2_FALLBACK = Path("/home/kate/Downloads/Telegram Desktop/data2.npy")


def resolve_data2_path():
    local = ROOT / "data2.npy"
    if local.exists():
        return local
    if DATA2_FALLBACK.exists():
        return DATA2_FALLBACK
    raise FileNotFoundError("data2.npy not found in repo or Downloads/Telegram Desktop")


def save_data2_gallery(path, sample, mean_img, face_raw, face_smooth, A):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    items = [
        (sample, "sample image", "gray", 0, 255),
        (mean_img, "mean image", "gray", 0, 255),
        (face_raw, "face estimate", "gray", np.percentile(face_raw, 1), np.percentile(face_raw, 99)),
        (face_smooth, "face estimate smoothed", "gray", np.percentile(face_smooth, 1), np.percentile(face_smooth, 99)),
        (A, "A heatmap", "magma", None, None),
    ]
    for ax, (arr, title, cmap, lo, hi) in zip(axes, items):
        ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_elbo(path, ll_values, image_shape):
    x = np.arange(1, len(ll_values) + 1)
    ll_norm = ll_values / np.prod(image_shape)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, ll_norm, marker="o", linewidth=2)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("ELBO / (H W K)")
    ax.set_title("ELBO on data2.npy")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def face_quality_metrics(face):
    smooth = gaussian_filter(face, 1.2)
    gy, gx = np.gradient(smooth)
    grad = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    contrast = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    return contrast, grad


def noise_study(base_subset, noise_levels, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for extra_noise in noise_levels:
        noisy = np.clip(
            base_subset + rng.normal(0.0, extra_noise, size=base_subset.shape),
            0.0,
            255.0,
        )
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
        contrast, grad = face_quality_metrics(F)
        rows.append(
            {
                "extra_noise": float(extra_noise),
                "elbo_norm": float(ll[-1] / np.prod(noisy.shape)),
                "sigma_est": float(s),
                "face_contrast": contrast,
                "face_grad": grad,
            }
        )
    return rows


def plot_noise_study(path, rows):
    noise = [row["extra_noise"] for row in rows]
    elbo = [row["elbo_norm"] for row in rows]
    contrast = [row["face_contrast"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(noise, elbo, marker="o", linewidth=2)
    axes[0].set_title("Normalized ELBO vs extra noise")
    axes[0].set_xlabel("added Gaussian noise std")
    axes[0].set_ylabel("ELBO / (H W K)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(noise, contrast, marker="o", linewidth=2, color="tab:orange")
    axes[1].set_title("Face contrast vs extra noise")
    axes[1].set_xlabel("added Gaussian noise std")
    axes[1].set_ylabel("smoothed face contrast")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(path, data_path, elbo_values, noise_rows):
    first = noise_rows[0]
    last = noise_rows[-1]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Data2 Analysis\n\n")
        f.write(f"- source: `{data_path}`\n")
        f.write(f"- face size: `h={FACE_H}`, `w={FACE_W}`\n")
        f.write(f"- ELBO iterations: `{[round(float(x), 3) for x in elbo_values]}`\n")
        f.write("- observation: on this dataset ELBO becomes flat almost immediately, which means the posterior over shifts is very peaked and EM reaches a fixed point after the first update.\n")
        f.write(
            f"- noise study: when added noise grows from `{first['extra_noise']}` to `{last['extra_noise']}`, "
            f"face contrast drops from `{first['face_contrast']:.3f}` to `{last['face_contrast']:.3f}`.\n"
        )
        f.write(
            f"- normalized ELBO changes from `{first['elbo_norm']:.5f}` to `{last['elbo_norm']:.5f}`; "
            "its trend is less interpretable by itself because the model can partially absorb extra noise through the estimated sigma.\n"
        )


def main():
    RESULTS.mkdir(exist_ok=True)
    data_path = resolve_data2_path()
    X = np.load(data_path, mmap_mode="r")

    preview_subset = X[:, :, :200]
    F_preview, B_preview, s_preview, A_preview = model._init_parameters(preview_subset, FACE_H, FACE_W, seed=0)
    F_preview_smooth = gaussian_filter(F_preview, sigma=1.2)
    save_data2_gallery(
        RESULTS / "data2_final_visualization.png",
        np.asarray(X[:, :, 0], dtype=np.float64),
        np.asarray(preview_subset.mean(axis=2), dtype=np.float64),
        F_preview,
        F_preview_smooth,
        A_preview,
    )

    em_subset = np.asarray(X[:, :, :20], dtype=np.float64)
    F0, B0, s0, A0 = model._init_parameters(em_subset, FACE_H, FACE_W, seed=0)
    F_em, B_em, s_em, A_em, ll_values = model.run_EM(
        em_subset,
        FACE_H,
        FACE_W,
        F=F0,
        B=B0,
        s=s0,
        A=A0,
        tolerance=-1.0,
        max_iter=6,
    )
    plot_elbo(RESULTS / "data2_elbo.png", ll_values, em_subset.shape)

    noise_rows = noise_study(em_subset, noise_levels=[0, 5, 10, 20, 40], seed=0)
    plot_noise_study(RESULTS / "data2_noise_dependence.png", noise_rows)

    metrics = {
        "data_path": str(data_path),
        "face_shape": [FACE_H, FACE_W],
        "preview_sigma": float(s_preview),
        "em_subset_sigma": float(s_em),
        "elbo_values": [float(x) for x in ll_values],
        "noise_study": noise_rows,
    }
    with open(RESULTS / "data2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_summary(RESULTS / "data2_summary.md", data_path, ll_values, noise_rows)


if __name__ == "__main__":
    main()
