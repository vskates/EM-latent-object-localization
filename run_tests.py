import time

import numpy as np
from numpy.testing import assert_almost_equal

import Student as model


def check_shape(**kwargs):
    np.random.seed(42)
    H, W, K = 10, 12, 20
    h, w = 4, 5
    max_iter = 1
    X = np.random.rand(H, W, K)
    F, B, s, A, LL = model.run_EM(X, h, w, max_iter=max_iter, **kwargs)
    assert F.shape == (h, w)
    assert B.shape == (H, W)
    assert A.shape == (H - h + 1, W - w + 1)
    assert np.array(LL).shape == (max_iter,)


def generate_data(H, W, K, h, w, use_MAP=False, seed=42):
    np.random.seed(seed)
    X = np.zeros((H, W, K))
    F = np.zeros((h, w))
    B = np.random.rand(H, W)

    coords = []
    q = np.zeros((H - h + 1, W - w + 1, K))
    for k in range(K):
        x = np.random.randint(0, H - h + 1)
        y = np.random.randint(0, W - w + 1)
        coords.append((x, y))
        X[:, :, k] = np.copy(B)
        X[x:x + h, y:y + w, k] = F
        q[x, y, k] = 1.0

    A = np.random.rand(H - h + 1, W - w + 1)
    A /= A.sum()

    if use_MAP:
        q = np.array(coords).T

    return X, F, B, A, q


def check_e_step(use_MAP=False):
    H, W, K = 4, 5, 2
    h, w = 2, 3
    s = 1e-1
    X, F, B, A, q = generate_data(H, W, K, h, w, use_MAP=use_MAP)
    pred_q = model.run_e_step(X, F, B, s, A, use_MAP=use_MAP)
    if use_MAP:
        assert_almost_equal(q, pred_q)
    else:
        assert_almost_equal(q, pred_q, 5)


def check_m_step(use_MAP=False):
    H, W, K = 7, 8, 2
    h, w = 2, 3
    X, F, B, A, q = generate_data(H, W, K, h, w, use_MAP=use_MAP)
    pred_F, pred_B, pred_s, pred_A = model.run_m_step(X, q, h, w, use_MAP=use_MAP)
    assert_almost_equal(F, pred_F)
    assert_almost_equal(B, pred_B)
    assert_almost_equal(pred_A.sum(), 1.0)


def check_repeated_map_coordinates():
    H, W, K = 5, 6, 4
    h, w = 2, 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(H, W, K))
    q_map = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])

    _, _, _, A_map = model.run_m_step(X, q_map, h, w, use_MAP=True)
    assert_almost_equal(A_map.sum(), 1.0)
    assert_almost_equal(A_map[1, 2], 1.0)

    q_full = np.zeros((H - h + 1, W - w + 1, K))
    q_full[1, 2, :] = 1.0
    _, _, _, A_full = model.run_m_step(X, q_full, h, w)
    assert_almost_equal(A_full.sum(), 1.0)
    assert_almost_equal(A_full[1, 2], 1.0)


def check_e_step_time(use_MAP=False):
    H, W, K = 50, 100, 50
    h, w = 40, 50
    s = 0.1
    X, F, B, A, q = generate_data(H, W, K, h, w)
    t_start = time.perf_counter()
    model.run_e_step(X, F, B, s, A, use_MAP=use_MAP)
    computation_time = time.perf_counter() - t_start
    assert computation_time < 1


def check_m_step_time(use_MAP=False):
    H, W, K = 50, 100, 50
    h, w = 40, 50
    X, F, B, A, q = generate_data(H, W, K, h, w, use_MAP=use_MAP)
    t_start = time.perf_counter()
    model.run_m_step(X, q, h, w, use_MAP=use_MAP)
    computation_time = time.perf_counter() - t_start
    assert computation_time < 1


def main():
    check_shape()
    check_shape(use_MAP=True)
    check_e_step()
    check_e_step(use_MAP=True)
    check_m_step()
    check_m_step(use_MAP=True)
    check_repeated_map_coordinates()
    check_e_step_time()
    check_e_step_time(use_MAP=True)
    check_m_step_time()
    check_m_step_time(use_MAP=True)
    print("all open checks passed")


if __name__ == "__main__":
    main()
