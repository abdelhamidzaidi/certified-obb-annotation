"""Numerical checks of Section 4.1 (Propositions 1, 2 and 4; Tables 3 and 4).

Proposition 1: for 2000 random point sets of 4 to 20 points, the RANGE-MBR angle given by the
closed form tan(2 theta*) = 2 C_xy / (V_x - V_y) (Equation (5) with C_st = 0) is compared, modulo
pi/2, with the principal axis of the empirical covariance Gamma; the largest difference is printed.
Table 3 (Proposition 2): A_M(theta*) / A*(M) for the multisets M_m made of the four vertices of
[0, L] x [0, 1] and m copies of each of (0, 0) and (1, 1), with the limit (L + 1)^2 / (2L) of
Equation (7); the exact RANGE-MBR angle of Equation (9),
tan(2 theta*) = m (m + 2) / (2 (L - 1) (L m + L + 1)), and its approximation m / (2 L^2) of
Equation (10), compared with the principal axis of the empirical covariance (Section 3.2).
Table 4 (Proposition 4): for ellipses of aspect ratio rho, the worst ratio (rho^2 + 1) / (2 rho) of
A_K(theta) / A*(K) and the measure, in degrees out of a period of 90, of the set Theta_tau of
admissible orientations, on a grid of 90,001 angles.

Usage:  python numerical_checks.py
"""
import numpy as np
import geometry as G


def range_angle_bruteforce(X, n=200001):
    """Angle minimising the product of the projected variances, by exhaustive search."""
    th = np.linspace(0, np.pi / 2, n)
    C = np.cov(X.T, bias=True)
    u = np.stack([np.cos(th), np.sin(th)], 1); v = np.stack([-np.sin(th), np.cos(th)], 1)
    Vs = np.einsum('ij,jk,ik->i', u, C, u); Vt = np.einsum('ij,jk,ik->i', v, C, v)
    return th[np.argmin(Vs * Vt)]


def check_prop1(trials=2000, seed=0):
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(trials):
        X = rng.normal(size=(rng.integers(4, 21), 2)) @ rng.normal(size=(2, 2))
        C = np.cov(X.T, bias=True)
        # closed form of the minimiser, Equation (5) with C_st = 0
        t = 0.5 * np.arctan2(2 * C[0, 1], C[0, 0] - C[1, 1])
        w, V = np.linalg.eigh(C); u = V[:, 1]; te = np.arctan2(u[1], u[0])
        d = abs(np.mod(t - te + np.pi / 4, np.pi / 2) - np.pi / 4)
        worst = max(worst, d)
    return worst


def table_unbounded():
    rows = []
    for L in (2, 5, 10, 20, 50):
        V = [(0, 0), (L, 0), (L, 1), (0, 1)]
        row = [L]
        for m in (0, 10, 100, 1000):
            X = np.array(V + [(0, 0)] * m + [(1, 1)] * m, float)
            row.append(G.area(G.range_mbr_2022(X)) / G.area(G.mbr(X)))
        row.append((L + 1) ** 2 / (2 * L)); rows.append(row)
    return rows


def check_exact_angle():
    """Angle of RANGE-MBR for M_m: principal axis, exact formula (9) and approximation (10), in degrees."""
    out = []
    for L in (10, 50):
        m = 1000
        X = np.array([(0, 0), (L, 0), (L, 1), (0, 1)] + [(0, 0)] * m + [(1, 1)] * m, float)
        w, V = np.linalg.eigh(np.cov(X.T, bias=True)); u = V[:, 1]
        pca = np.degrees(np.mod(np.arctan2(u[1], u[0]), np.pi / 2))
        exact = np.degrees(0.5 * np.arctan(m * (m + 2) / (2 * (L - 1) * (L * m + L + 1))))
        approx = np.degrees(0.5 * np.arctan(m / (2 * L ** 2)))
        out.append((L, m, pca, exact, approx))
    return out


def table_theta(n=90001):
    psi = np.linspace(0, np.pi / 2, n)
    rows = []
    for rho in (1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0):
        a, b = rho, 1.0
        c, s = np.cos(psi), np.sin(psi)
        ratio = 4 * np.sqrt((a*a*c*c + b*b*s*s) * (a*a*s*s + b*b*c*c)) / (4 * a * b)
        row = [rho, ratio.max()] + [90 * np.mean(ratio <= 1 + tau) for tau in (0.01, 0.05, 0.10)]
        rows.append(row)
    return rows


if __name__ == '__main__':
    print('Proposition 1: largest angle difference %.1e rad' % check_prop1())
    print('Table 3 (L, m=0, 10, 100, 1000, limit)')
    for r in table_unbounded(): print('  ' + '  '.join('%.2f' % v if i else '%d' % v for i, v in enumerate(r)))
    for L, m, p, e, a in check_exact_angle():
        print('  angle for L = %d, m = %d: principal axis %.2f deg, Equation (9) %.2f deg, Equation (10) %.2f deg' % (L, m, p, e, a))
    print('Table 4 (rho, worst ratio, measure of Theta_tau in degrees for tau = 0.01, 0.05, 0.10)')
    for r in table_theta(): print('  %.2f  %.3f  %.1f  %.1f  %.1f' % tuple(r))
