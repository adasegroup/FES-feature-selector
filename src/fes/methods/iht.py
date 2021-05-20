import numpy as np

"""
The implementation of Normalized Iterative Hard Thresholding algorithms
"""


def l0_reg(X, y, k, tol=1e-4, max_iter=100, max_step=50, verbose=False):
    """
    L0 penalized least-squares regression with iterative hard thresholding
    Parameters
    ----------
    X: n x m; design matrix
    y: n x 1; vector of observations
    k: int; desired model (support) size
    tol: float; global tolerance
    max_iter: int; maximum number of iterations for the algorithm
    max_step: int; maximum number of backtracking steps for the step size calculation
    verbose: bool; Log flag
    """
    w_prev = np.zeros((X.shape[1], 1))
    Xw_prev = np.zeros_like(y)

    topk = get_topk(X.transpose() @ y, k, return_value=False)
    sup_prev = get_support(w_prev, topk)

    dy_prev = y - Xw_prev

    for _iter in range(max_iter):

        if _iter == max_iter - 1:
            raise RuntimeError("IHT didn't converge! Maybe you should increase the number of iterations or the tolerance")

        w, sup, Xw, mu, mu_step = iht_step(X, w_prev, sup_prev, Xw_prev, dy_prev, k, _iter, max_step)

        dy = y - Xw

        loss = (dy ** 2).sum() / 2

        if not np.isfinite(loss):
            raise RuntimeError("The loss is not finite")

        norm = np.linalg.norm((w - w_prev).reshape(-1), ord=np.inf)
        scaled_norm = norm / (np.linalg.norm(w_prev.reshape(-1), ord=np.inf) + 1)

        converged = scaled_norm < tol

        if verbose:
            if _iter % (max_iter // 10) == 0:
                print(f"Iteration {_iter}, loss {loss:.4f}, weights norm {norm:.4f}, scaled norm {scaled_norm:.4f}")
                print(f"Gradient step size mu is {mu:.5f}")
                if mu_step != 0:
                    print(f"Backtracking finished in {mu_step} steps")

        if converged:
            print(f"IHT has converged in {_iter} iterations with loss {loss:.4f}, weights norm {norm:.4f}")

            return w, sup

        w_prev = w
        sup_prev = sup
        Xw_prev = Xw
        dy_prev = dy


def iht_step(X, w_prev, sup_prev, Xw_prev, dy_prev, k, _iter, max_step):
    """
    A single step of iterative hard thresholding

    Parameters
    ----------
    X: n x m; design matrix
    y: n x 1; vector of observations
    w_prev: m x 1; vector of weights
    sup_prev: m; support set
    Xw_prev: n x 1; vector of the result of X @ w_prev
    dy_prev: n x 1; vector of the result of y - X @ w_prev
    k: int; desired model (support) size
    _iter: int; current iteration index
    max_step: int; maximum number of backtracking steps for the step size calculation
    """
    g = X.transpose() @ dy_prev

    g_sup = g[sup_prev]
    X_sup = X[:, sup_prev]

    gX_sup = X_sup @ g_sup
    mu = (g_sup ** 2).sum() / (gX_sup ** 2).sum()

    w, topk = get_topk(w_prev + mu * g, k)
    sup = get_support(w, topk)

    Xw = X @ w

    mu_step = 0

    if (sup != sup_prev).any():
        omega_top = ((w - w_prev) ** 2).sum()
        omega_bot = ((Xw - Xw_prev) ** 2).sum()

        while mu * omega_bot > 0.99 * omega_top and \
                mu_step < max_step:
            mu /= 2

            w, topk = get_topk(w_prev + mu * g, k)

            mu_step += 1

        Xw = X @ w
        sup = get_support(w, topk)

    return w, sup, Xw, mu, mu_step


"""
Support utils
"""


def get_topk(v, k, return_value=True):
    """
    Parameters
    ----------
    v: m x 1 vector
    k: int
    return_value: bool
    """
    topk = np.argpartition(abs(v), k, axis=0)[-k:]

    if return_value:
        sup_v = np.zeros_like(v)
        sup_v[topk] = v[topk]

        return sup_v, topk

    else:
        return topk


def get_support(v, top_k):
    sup = np.zeros((v.shape[0]), dtype=np.bool)
    sup[top_k] = True

    return sup
