def main() -> None:
    n = int(input().strip())
    total_bytes = 8 * n * n
    mb = total_bytes / 10**6
    tb = total_bytes / 10**12
    print(f"{mb:.3f} {tb:.3f}")


if __name__ == '__main__':
    main()
