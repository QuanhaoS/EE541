#!/usr/bin/env python3
"""
Root finding 
"""

import sys
from func import f

TOL = 1e-10


def parse_args():
    """Parse a, b from command line"""
    if len(sys.argv) != 3:
        return None
    try:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
    except ValueError:
        return None
    if a >= b:
        return None
    if f(a) * f(b) >= 0:
        return None
    return a, b


def secant(a, b):
    """Secant method"""
    xs = [a, b]
    while True:
        x_prev, x_curr = xs[-2], xs[-1]
        f_prev, f_curr = f(x_prev), f(x_curr)
        denom = f_curr - f_prev
        if abs(denom) < 1e-20:  
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        xs.append(x_next)
        if abs(x_next - x_curr) < TOL:
            break
    N = len(xs) - 1  
    return N, xs[N - 2], xs[N - 1], xs[N]


def main():
    parsed = parse_args()
    if parsed is None:
        print("Range error", file=sys.stderr)
        sys.exit(1)
    a, b = parsed
    N, x_n2, x_n1, x_n = secant(a, b)
    print(N)
    print(f"{x_n2:.15g}")
    print(f"{x_n1:.15g}")
    print(f"{x_n:.15g}")


if __name__ == "__main__":
    main()
