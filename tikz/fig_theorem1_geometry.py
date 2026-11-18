"""Coordinates and curves of Figure 3 (Theorem 1, Section 3.4).

K is an ellipse with semi-axes 2.2 and 1.25 (strictly convex), and M is a finite subset of K made of
the four points q_1, ..., q_4 at which the rectangle R with orientation theta_0 = 28 degrees supports
K, and six points inside K. The rectangle at theta_0 - 11 degrees keeps q_1, ..., q_4 extreme and
encloses M with a smaller area (panel (a)). Panel (b) plots A_M(theta), the area of the smallest
rectangle with orientation theta enclosing M, and the local expression of Equation (17),
r_1 r_2 cos(theta - phi_1) cos(theta - phi_2), with r_1 = |q_1 - q_3|, r_2 = |q_2 - q_4| and phi_1, phi_2
the angles of q_1 - q_3 and q_2 - q_4 measured from u(0) and v(0).
Output: printed coordinates, and the curves in thm1_area.txt and thm1_local.txt (pgfplots format).
"""
import numpy as np

a, b = 2.2, 1.25                                   # semi-axes of the ellipse K


def ell(t):
    return np.array([a * np.cos(t), b * np.sin(t)])


def tang(u):
    """Point of K maximising <x, u>, i.e. where the support line orthogonal to u touches K."""
    p = np.array([a * a * u[0], b * b * u[1]])
    return p / np.sqrt(a * a * u[0] ** 2 + b * b * u[1] ** 2)


U = lambda t: np.array([np.cos(t), np.sin(t)])     # u(theta)
V = lambda t: np.array([-np.sin(t), np.cos(t)])    # v(theta)


def rect(S, t):
    """Corners and area A_S(t) of the smallest rectangle with orientation t enclosing S."""
    u, v = U(t), V(t); A = S @ u; B = S @ v
    C = [A.min() * u + B.min() * v, A.max() * u + B.min() * v, A.max() * u + B.max() * v, A.min() * u + B.max() * v]
    return np.array(C), (A.max() - A.min()) * (B.max() - B.min())


th0 = np.radians(28)
q1 = tang(U(th0)); q3 = -q1; q2 = tang(V(th0)); q4 = -q2       # unique extreme points of M at theta_0
inner = [0.72 * ell(np.radians(d)) for d in (100, 200, 300)] + [np.array([0.5, 0.3]), np.array([-0.8, -0.2]), np.array([0.9, -0.5])]
M = np.array([q1, q2, q3, q4] + inner)
f = lambda C: ' -- '.join(f'({x:.3f},{y:.3f})' for x, y in C) + ' -- cycle'
pt = lambda x: f'({x[0]:.3f},{x[1]:.3f})'

C0, A0 = rect(M, th0)
C1, A1 = rect(M, th0 - np.radians(11))
print('R at theta_0      ', f(C0), ' area', round(A0, 2))
print('R at theta_0 - 11 ', f(C1), ' area', round(A1, 2))
for n, x in zip(['q1', 'q2', 'q3', 'q4'], M[:4]): print(n, pt(x))
print('inner points', ' '.join(pt(x) for x in M[4:]))

r1 = np.linalg.norm(q1 - q3); r2 = np.linalg.norm(q2 - q4)
p1 = np.arctan2(*(q1 - q3)[::-1]); p2 = np.arctan2(*(q2 - q4)[::-1]) - np.pi / 2
d_minus_b = r1 * r2 * np.cos(p1 - p2) / 2 - r1 * r2 / 2            # value d - b of Equation (20)
print('r1, r2', round(r1, 3), round(r2, 3), ' phi1, phi2 (deg)', round(np.degrees(p1), 2), round(np.degrees(p2), 2), ' d - b', round(d_minus_b, 3))

# interval of orientations on which q_1, ..., q_4 remain extreme (shaded in panel (b))
valid = [dd for dd in np.arange(-45, 45.01, 0.1)
         if (lambda t: (int(np.argmax(M @ U(t))), int(np.argmin(M @ U(t))), int(np.argmax(M @ V(t))), int(np.argmin(M @ V(t)))))(th0 + np.radians(dd)) == (0, 2, 1, 3)]
print('q1..q4 extreme for theta in [%.1f, %.1f] degrees' % (28 + min(valid), 28 + max(valid)))

ts = np.arange(-20, 85.01, 0.5)
area = [rect(M, np.radians(t))[1] for t in ts]
local = [r1 * r2 * np.cos(np.radians(t) - p1) * np.cos(np.radians(t) - p2) for t in ts]
i = int(np.argmin(area)); print('minimum of A_M at theta = %.1f degrees, value %.3f' % (ts[i], area[i]))
open('thm1_area.txt', 'w').write(' '.join(f'({t:.1f},{x:.3f})' for t, x in zip(ts, area)))
open('thm1_local.txt', 'w').write(' '.join(f'({t:.1f},{x:.3f})' for t, x in zip(ts, local)))
