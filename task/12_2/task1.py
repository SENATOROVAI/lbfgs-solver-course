# FULL PYTHON SOLUTION

def main() -> None:
    fk, fk1, ftol = map(float, input().split())
    ratio = (fk - fk1) / max(abs(fk), abs(fk1), 1.0)
    if ratio <= ftol:
        print('STOP')
    else:
        print('CONTINUE')


if __name__ == '__main__':
    main()
