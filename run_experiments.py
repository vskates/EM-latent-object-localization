from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import Student as model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def generate_toy_data(H=20, W=30, K=40, h=15, w=15, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    B = np.zeros((H, W), dtype=np.float64)
    B[:, :] = 40
    B[2:6, 2:10] = 140
    B[10:15, 20:28] = 110

    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    face = 210.0 * np.exp(-((yy - cy) ** 2 / (2 * (0.22 * h) ** 2) + (xx - cx) ** 2 / (2 * (0.18 * w) ** 2)))
    face += 35.0 * (np.sin(xx / max(w - 1, 1) * np.pi) > 0.9)

    X = np.empty((H, W, K), dtype=np.float64)
    coords = []
    for k in range(K):
        dh = int(rng.integers(0, H - h + 1))
        dw = int(rng.integers(0, W - w + 1))
        coords.append((dh, dw))
        clean = B.copy()
        clean[dh:dh + h, dw:dw + w] = face
        X[:, :, k] = clean + rng.normal(0.0, noise, size=(H, W))
    return X, B, face, np.array(coords)


def save_gallery(path, arrays, titles, cmap="gray", vmin=None, vmax=None):
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, arr, title in zip(axes, arrays, titles):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_real_gallery(path, sample, mean_img, face_raw, face_smooth, A):
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


def main():
    RESULTS.mkdir(exist_ok=True)

    X_toy, B_true, F_true, _ = generate_toy_data()
    F_em, B_em, s_em, A_em, LL_em = model.run_EM_with_restarts(
        X_toy, h=15, w=15, max_iter=30, tolerance=1e-4, n_restarts=4
    )
    F_map, B_map, s_map, A_map, LL_map = model.run_EM_with_restarts(
        X_toy, h=15, w=15, max_iter=30, tolerance=1e-4, use_MAP=True, n_restarts=4
    )

    save_gallery(
        RESULTS / "toy_reconstruction.png",
        [F_true, F_em, F_map, B_true, B_em, B_map],
        ["F true", "F EM", "F hard EM", "B true", "B EM", "B hard EM"],
    )

    X_real = np.load(ROOT / "data.npy", mmap_mode="r")
    preview_subset = X_real[:, :, :200]
    F_preview, B_preview, s_preview, A_preview = model._init_parameters(preview_subset, 45, 70, seed=0)
    F_preview_smooth = gaussian_filter(F_preview, sigma=1.2)

    em_subset = np.asarray(X_real[:, :, :8], dtype=np.float64)
    F_real, B_real, s_real, A_real, LL_real = model.run_EM_with_restarts(
        em_subset, h=45, w=70, max_iter=2, tolerance=1e-3, n_restarts=1
    )

    save_real_gallery(
        RESULTS / "real_reconstruction.png",
        np.asarray(X_real[:, :, 0], dtype=np.float64),
        np.asarray(preview_subset.mean(axis=2), dtype=np.float64),
        F_preview,
        F_preview_smooth,
        A_preview,
    )

    with open(RESULTS / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Toy EM final L: {LL_em:.6f}\n")
        f.write(f"Toy hard EM final L: {LL_map:.6f}\n")
        f.write(f"Toy EM sigma: {s_em:.6f}\n")
        f.write(f"Toy hard EM sigma: {s_map:.6f}\n")
        f.write(f"Real EM subset final L: {LL_real:.6f}\n")
        f.write(f"Real EM subset sigma: {s_real:.6f}\n")
        f.write(f"Real preview sigma: {s_preview:.6f}\n")


if __name__ == "__main__":
    main()
