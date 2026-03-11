import numpy as np


def project(x, lower, upper):
    # Clamp each coordinate into the feasible interval.
    return np.minimum(np.maximum(x, lower), upper)


def projected_gradient(x, g, lower, upper):
    # Remove gradient components that point outside the feasible box.
    pg = g.copy()
    for i in range(len(x)):
        if x[i] <= lower[i] and g[i] > 0:
            pg[i] = 0.0
        elif x[i] >= upper[i] and g[i] < 0:
            pg[i] = 0.0
    return pg


def generalized_cauchy_point(x, g, lower, upper):
    # Educational simplification: take a step along the anti-gradient,
    # then project it back into the feasible region.
    return project(x - g, lower, upper)


def free_variable_mask(x, g, lower, upper):
    # A variable is free if it is not blocked by an active bound.
    mask = np.ones_like(x, dtype=bool)
    for i in range(len(x)):
        if x[i] <= lower[i] and g[i] > 0:
            mask[i] = False
        elif x[i] >= upper[i] and g[i] < 0:
            mask[i] = False
    return mask


def two_loop_recursion(g, s_list, y_list):
    # Standard limited-memory two-loop recursion.
    q = g.copy()
    alphas = []
    rhos = []
    used_pairs = []

    for s, y in zip(reversed(s_list), reversed(y_list)):
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            continue
        rho = 1.0 / ys
        alpha = rho * float(np.dot(s, q))
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)
        used_pairs.append((s, y))

    if used_pairs:
        s_last, y_last = used_pairs[0]
        yy = float(np.dot(y_last, y_last))
        gamma = float(np.dot(s_last, y_last)) / yy if yy > 1e-12 else 1.0
    else:
        gamma = 1.0

    r = gamma * q

    for (s, y), alpha, rho in zip(reversed(used_pairs), reversed(alphas), reversed(rhos)):
        beta = rho * float(np.dot(y, r))
        r = r + s * (alpha - beta)

    return -r


def backtracking_line_search_with_projection(f, x, p, lower, upper, g, alpha=1.0, beta=0.5, c=1e-4):
    # Projected Armijo backtracking line search.
    fx = f(x)
    while True:
        trial = project(x + alpha * p, lower, upper)
        if f(trial) <= fx + c * alpha * float(np.dot(g, p)):
            return alpha
        alpha *= beta
        if alpha < 1e-12:
            return alpha


def lbfgsb(x0, m=5, tol=1e-8, max_iter=100):
    lower = np.array([0.0, 0.0], dtype=float)
    upper = np.array([1.0, 2.0], dtype=float)

    def f(x):
        return (x[0] - 3.0) ** 2 + (x[1] - 0.5) ** 2

    def grad_f(x):
        return np.array([
            2.0 * (x[0] - 3.0),
            2.0 * (x[1] - 0.5)
        ])

    x = project(np.array(x0, dtype=float), lower, upper)
    s_list = []
    y_list = []

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
        ys = float(np.dot(y, s))
        if ys > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)

        x = x_new

    return x, f(x)


def main() -> None:
    x1, x2 = map(float, input().split())
    x_star, f_star = lbfgsb([x1, x2])
    print(f"{x_star[0]:.6f} {x_star[1]:.6f} {f_star:.6f}")


if __name__ == '__main__':
    main()
