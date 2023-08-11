
import numpy as np
import manifolds
from manifolds.plot import plot


def main():
    domain = manifolds.IntervalPartition(0, 10)

    # 1. Plot a constant polynomial
    q = manifolds.linear_manifolds.GLPolynomial.constant(
        domain, np.random.rand(7).reshape(-1))

    plot(q)
    q = manifolds.linear_manifolds.GLPolynomial.zero(domain)
    plot(q)

    qc = manifolds.linear_manifolds.GLPolynomial.CGLPolynomial.random(domain)

    inc = manifolds.linear_manifolds.GLPolynomial.CGLPolynomial.Inclusion(
        domain)

    q = inc(qc)
    plot(q)


if __name__ == "__main__":
    main()
