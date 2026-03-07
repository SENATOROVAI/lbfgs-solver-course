from typing import List


def main() -> None:
    data: List[float] = list(map(float, input().split()))
    f0, df0, x0, x = data
    delta = x - x0
    ans = f0 + df0 * delta
    print(f"{ans:.6f}")


if __name__ == "__main__":
    main()
