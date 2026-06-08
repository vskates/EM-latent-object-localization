import numpy as np
from scipy.signal import convolve2d, correlate2d


EPS = 1e-12


def _ones_kernel(h, w):
    return np.ones((h, w), dtype=np.float64)


def _residual_based_init(X, h, w, rng):
    H, W, K = X.shape
    dh_max = H - h + 1
    dw_max = W - w + 1
    B0 = X.mean(axis=2)
    ones = _ones_kernel(h, w)

    coords = []
    patches = []
    for k in range(K):
        resid = (X[:, :, k] - B0) ** 2
        energy = convolve2d(resid, ones, mode="valid")
        flat = energy.reshape(-1)
        top = min(10, flat.size)
        top_idx = np.argpartition(flat, -top)[-top:]
        chosen = int(rng.choice(top_idx))
        dh, dw = np.unravel_index(chosen, (dh_max, dw_max))
        coords.append((int(dh), int(dw)))
        patches.append(X[dh:dh + h, dw:dw + w, k])

    F0 = np.mean(patches, axis=0)
    A0 = np.full((dh_max, dw_max), 1e-3, dtype=np.float64)
    for dh, dw in coords:
        A0[dh, dw] += 1.0
    A0 /= A0.sum()
    return F0, A0


def _q_to_full(q, dh_max, dw_max, K, use_MAP):
    if not use_MAP:
        return np.asarray(q, dtype=np.float64)
    q_full = np.zeros((dh_max, dw_max, K), dtype=np.float64)
    q_full[q[0], q[1], np.arange(K)] = 1.0
    return q_full


def _normalize_log_probs(log_values):
    max_log = np.max(log_values, axis=(0, 1), keepdims=True)
    probs = np.exp(log_values - max_log)
    probs /= probs.sum(axis=(0, 1), keepdims=True)
    return probs


def _init_parameters(X, h, w, F=None, B=None, s=None, A=None, seed=0):
    H, W, K = X.shape
    rng = np.random.default_rng(seed)

    if B is None:
        B = X.mean(axis=2)
    else:
        B = np.asarray(B, dtype=np.float64)

    if F is None:
        F, A_guess = _residual_based_init(X, h, w, rng)
    else:
        F = np.asarray(F, dtype=np.float64)
        A_guess = None

    if s is None:
        s = float(max(np.std(X - B[:, :, None]), 20.0) + 1e-3)
    else:
        s = float(s)

    if A is None:
        if A_guess is None:
            A = np.full((H - h + 1, W - w + 1), 1.0, dtype=np.float64)
        else:
            A = A_guess
        A /= A.sum()
    else:
        A = np.asarray(A, dtype=np.float64)
        A = A / A.sum()

    return F, B, s, A


def _extract_map_coords(q, use_MAP):
    if use_MAP:
        return q[0].astype(int), q[1].astype(int)
    if np.all((q == 0.0) | (q == 1.0)) and np.allclose(q.sum(axis=(0, 1)), 1.0):
        flat = q.reshape(-1, q.shape[2]).argmax(axis=0)
        return np.unravel_index(flat, q.shape[:2])
    return None


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.
    """
    X = np.asarray(X, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    s = float(max(s, EPS))

    H, W, K = X.shape
    h, w = F.shape
    dh_max = H - h + 1
    dw_max = W - w + 1

    ones = _ones_kernel(h, w)
    ll = np.empty((dh_max, dw_max, K), dtype=np.float64)
    const = -0.5 * H * W * np.log(2.0 * np.pi * s * s)
    sum_F2 = np.sum(F * F)

    for k in range(K):
        Xk = X[:, :, k]
        bg_sq = (Xk - B) ** 2
        total_bg_sq = np.sum(bg_sq)
        bg_patch_sq = convolve2d(bg_sq, ones, mode="valid")
        x_patch_sq = convolve2d(Xk * Xk, ones, mode="valid")
        x_face = correlate2d(Xk, F, mode="valid")
        face_sq = x_patch_sq - 2.0 * x_face + sum_F2
        sse = total_bg_sq - bg_patch_sq + face_sq
        ll[:, :, k] = const - sse / (2.0 * s * s)

    return ll


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    log_prob = calculate_log_probability(X, F, B, s)
    log_A = np.log(np.maximum(A, EPS))

    if use_MAP:
        K = X.shape[2]
        dh = q[0]
        dw = q[1]
        return float(np.sum(log_prob[dh, dw, np.arange(K)] + log_A[dh, dw]))

    mask = q > 0
    energy = np.sum(q * (log_prob + log_A[:, :, None]))
    entropy = np.sum(q[mask] * np.log(q[mask]))
    return float(energy - entropy)


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).
    """
    log_prob = calculate_log_probability(X, F, B, s)
    log_post = log_prob + np.log(np.maximum(A, EPS))[:, :, None]

    if use_MAP:
        dh_max, dw_max, K = log_post.shape
        argmax = np.argmax(log_post.reshape(dh_max * dw_max, K), axis=0)
        dh, dw = np.unravel_index(argmax, (dh_max, dw_max))
        return np.vstack([dh, dw])

    return _normalize_log_probs(log_post)


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.
    """
    X = np.asarray(X, dtype=np.float64)
    H, W, K = X.shape
    dh_max = H - h + 1
    dw_max = W - w + 1
    q_full = _q_to_full(q, dh_max, dw_max, K, use_MAP)

    map_coords = _extract_map_coords(q, use_MAP)
    if map_coords is not None:
        dh, dw = map_coords
        A = np.zeros((dh_max, dw_max), dtype=np.float64)
        np.add.at(A, (dh, dw), 1.0)
        A /= K

        F_num = np.zeros((h, w), dtype=np.float64)
        bg_num = np.zeros((H, W), dtype=np.float64)
        bg_den = np.ones((H, W), dtype=np.float64) * K
        for k in range(K):
            x0 = int(dh[k])
            y0 = int(dw[k])
            F_num += X[x0:x0 + h, y0:y0 + w, k]
            bg_num += X[:, :, k]
            bg_num[x0:x0 + h, y0:y0 + w] -= X[x0:x0 + h, y0:y0 + w, k]
            bg_den[x0:x0 + h, y0:y0 + w] -= 1.0

        F = F_num / K
        fallback_B = X.mean(axis=2)
        B = np.divide(bg_num, bg_den, out=fallback_B.copy(), where=bg_den > EPS)

        total_sse = 0.0
        for k in range(K):
            x0 = int(dh[k])
            y0 = int(dw[k])
            resid = X[:, :, k] - B
            total_sse += np.sum(resid * resid)
            face_resid_old = resid[x0:x0 + h, y0:y0 + w]
            face_resid_new = X[x0:x0 + h, y0:y0 + w, k] - F
            total_sse += np.sum(face_resid_new * face_resid_new) - np.sum(face_resid_old * face_resid_old)

        s = float(np.sqrt(max(total_sse / (H * W * K), EPS)))
        return F, B, s, A

    A = q_full.mean(axis=2)

    ones = _ones_kernel(h, w)
    F_num = np.zeros((h, w), dtype=np.float64)
    bg_num = np.zeros((H, W), dtype=np.float64)
    bg_den = np.zeros((H, W), dtype=np.float64)

    for k in range(K):
        Xk = X[:, :, k]
        qk = q_full[:, :, k]
        F_num += correlate2d(Xk, qk, mode="valid")
        face_prob = convolve2d(qk, ones, mode="full")
        bg_prob = np.clip(1.0 - face_prob, 0.0, 1.0)
        bg_num += bg_prob * Xk
        bg_den += bg_prob

    F = F_num / K
    fallback_B = X.mean(axis=2)
    B = np.divide(bg_num, bg_den, out=fallback_B.copy(), where=bg_den > EPS)

    total_sse = 0.0
    sum_F2 = np.sum(F * F)
    for k in range(K):
        Xk = X[:, :, k]
        qk = q_full[:, :, k]
        bg_sq = (Xk - B) ** 2
        total_bg_sq = np.sum(bg_sq)
        bg_patch_sq = convolve2d(bg_sq, ones, mode="valid")
        x_patch_sq = convolve2d(Xk * Xk, ones, mode="valid")
        x_face = correlate2d(Xk, F, mode="valid")
        face_sq = x_patch_sq - 2.0 * x_face + sum_F2
        sse = total_bg_sq - bg_patch_sq + face_sq
        total_sse += np.sum(qk * sse)

    s = float(np.sqrt(max(total_sse / (H * W * K), EPS)))
    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.
    """
    F, B, s, A = _init_parameters(X, h, w, F=F, B=B, s=s, A=A)
    LL = []

    for _ in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP=use_MAP)
        L_old = calculate_lower_bound(X, F, B, s, A, q, use_MAP=use_MAP)
        F_new, B_new, s_new, A_new = run_m_step(X, q, h, w, use_MAP=use_MAP)
        L_new = calculate_lower_bound(X, F_new, B_new, s_new, A_new, q, use_MAP=use_MAP)
        LL.append(L_new)

        F, B, s, A = F_new, B_new, s_new, A_new
        if L_new - L_old < tolerance:
            break

    return F, B, s, A, np.asarray(LL, dtype=np.float64)


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).
    """
    best = None
    best_L = -np.inf

    for restart in range(n_restarts):
        F0, B0, s0, A0 = _init_parameters(X, h, w, seed=restart)
        F, B, s, A, LL = run_EM(
            X, h, w, F=F0, B=B0, s=s0, A=A0,
            tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP
        )
        final_L = float(LL[-1])
        if final_L > best_L:
            best_L = final_L
            best = (F, B, s, A, final_L)

    return best
