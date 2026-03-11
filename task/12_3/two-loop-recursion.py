# FULL PYTHON SOLUTION

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def two_loop_recursion(g, s_list, y_list):
    q = g[:]
    alphas = []
    rhos = []
    for s, y in zip(reversed(s_list), reversed(y_list)):
        ys = dot(y, s)
        if ys <= 1e-12:
            continue
        rho = 1.0 / ys
        alpha = rho * dot(s, q)
        q = [q[0] - alpha * y[0], q[1] - alpha * y[1]]
        alphas.append(alpha)
        rhos.append(rho)

    if s_list:
        s_last = s_list[-1]
        y_last = y_list[-1]
        yy = dot(y_last, y_last)
        gamma = dot(s_last, y_last) / yy if yy > 1e-12 else 1.0
    else:
        gamma = 1.0

    r = [gamma * q[0], gamma * q[1]]
    used_pairs = [(s, y) for s, y in zip(s_list, y_list) if dot(y, s) > 1e-12]
    for (s, y), alpha, rho in zip(used_pairs, reversed(alphas), reversed(rhos)):
        beta = rho * dot(y, r)
        r = [r[0] + s[0] * (alpha - beta), r[1] + s[1] * (alpha - beta)]

    return [-r[0], -r[1]]


def main() -> None:
    m = int(input())
    g = list(map(float, input().split()))
    s_list = []
    y_list = []
    for _ in range(m):
        s1, s2, y1, y2 = map(float, input().split())
        s_list.append([s1, s2])
        y_list.append([y1, y2])
    p = two_loop_recursion(g, s_list, y_list)
    print(f"{p[0]:.6f} {p[1]:.6f}")


if __name__ == '__main__':
    main()
