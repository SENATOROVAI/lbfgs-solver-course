import numpy as np
from typing import List


def project(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def projected_gradient(
    x: np.ndarray,
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    pg = g.copy()
    for i in range(len(x)):
        if x[i] <= lower[i] and g[i] > 0:
            pg[i] = 0.0
        elif x[i] >= upper[i] and g[i] < 0:
            pg[i] = 0.0
    return pg


def generalized_cauchy_point(
    x: np.ndarray,
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    return project(x - g, lower, upper)


def free_variable_mask(
    x: np.ndarray,
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    mask = np.ones_like(x, dtype=bool)
    for i in range(len(x)):
        if x[i] <= lower[i] and g[i] > 0:
            mask[i] = False
        elif x[i] >= upper[i] and g[i] < 0:
            mask[i] = False
    return mask


def two_loop_recursion(
    g: np.ndarray,
    s_list: List[np.ndarray],
    y_list: List[np.ndarray]
) -> np.ndarray:
    q = g.copy()
    alphas = []
    rhos = []

    for s, y in zip(reversed(s_list), reversed(y_list)):
        ys = np.dot(y, s)
        if ys <= 1e-12:
            continue
        rho = 1.0 / ys
        alpha = rho * np.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    if s_list:
        s_last = s_list[-1]
        y_last = y_list[-1]
        yy = np.dot(y_last, y_last)
        gamma = np.dot(s_last, y_last) / yy if yy > 1e-12 else 1.0
    else:
        gamma = 1.0

    r = gamma * q
    used_pairs = [(s, y) for s, y in zip(s_list, y_list) if np.dot(y, s) > 1e-12]

    for (s, y), alpha, rho in zip(used_pairs, reversed(alphas), reversed(rhos)):
        beta = rho * np.dot(y, r)
        r = r + s * (alpha - beta)

    return -r


def backtracking_line_search_with_projection(
    f,
    x: np.ndarray,
    p: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    g: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4
) -> float:
    fx = f(x)
    while True:
        trial = project(x + alpha * p, lower, upper)
        if f(trial) <= fx + c * alpha * np.dot(g, p):
            return alpha
        alpha *= beta
        if alpha < 1e-12:
            return alpha


def lbfgsb(f, grad_f, x0, lower, upper, m=10, tol=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)
    x = project(x, lower, upper)

    s_list = []
    y_list = []
    history = [x.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        pg = projected_gradient(x, g, lower, upper)

        if np.linalg.norm(pg) < tol:
            break

        xc = generalized_cauchy_point(x, pg, lower, upper)
        free_mask = free_variable_mask(xc, g, lower, upper)

        if not np.any(free_mask):
            x_new = xc
        else:
            sub_grad = np.zeros_like(g)
            sub_grad[free_mask] = pg[free_mask]
            sub_s_list = [s * free_mask for s in s_list]
            sub_y_list = [y * free_mask for y in y_list]
            p = two_loop_recursion(sub_grad, sub_s_list, sub_y_list)
            p[~free_mask] = 0.0
            alpha = backtracking_line_search_with_projection(f, xc, p, lower, upper, pg)
            x_new = project(xc + alpha * p, lower, upper)

        g_new = grad_f(x_new)
        s = x_new - x
        y = g_new - g
        ys = np.dot(y, s)

        if ys > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)

        x = x_new
        history.append(x.copy())

    return x, f(x), history


def main():
    x1, x2 = map(float, input().split())

    def f(x):
        return (x[0] - 3.0) ** 2 + (x[1] - 0.5) ** 2

    def grad_f(x):
        return np.array([
            2.0 * (x[0] - 3.0),
            2.0 * (x[1] - 0.5)
        ])

    x_star, f_star, history = lbfgsb(
        f=f,
        grad_f=grad_f,
        x0=[x1, x2],
        lower=[0.0, 0.0],
        upper=[1.0, 2.0],
        m=5,
        tol=1e-8,
        max_iter=100
    )
    print("{:.6f} {:.6f} {:.6f} {}".format(x_star[0], x_star[1], f_star, len(history) - 1))


if __name__ == '__main__':
    main()
