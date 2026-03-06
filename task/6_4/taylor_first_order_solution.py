"""First-order Taylor approximation solution for Stepik task.

Input:  f0 df0 x x0
Output: L(x) = f0 + df0 * (x - x0)
"""

from typing import Tuple


def parse_input(line: str) -> Tuple[float, float, float, float]:
    f0_s, df0_s, x_s, x0_s = line.strip().split()
    return float(f0_s), float(df0_s), float(x_s), float(x0_s)


def linear_approximation(f0: float, df0: float, x: float, x0: float) -> float:
    return f0 + df0 * (x - x0)


def main() -> None:
    f0, df0, x, x0 = parse_input(input())
    result = linear_approximation(f0, df0, x, x0)
    print('{:.6f}'.format(result))


if __name__ == '__main__':
    main()
