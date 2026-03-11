# FULL PYTHON SOLUTION

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


def main() -> None:
    x1, x2, g1, g2, l1, l2, u1, u2 = map(float, input().split())
    x = np.array([x1, x2], dtype=float)
    g = np.array([g1, g2], dtype=float)
    lower = np.array([l1, l2], dtype=float)
    upper = np.array([u1, u2], dtype=float)
    mask = free_variable_mask(x, g, lower, upper)
    print(f"{int(mask[0])} {int(mask[1])}")


if __name__ == '__main__':
    main()
