"""Planar geometry of Section 3.1.

A rectangle R with orientation theta is stored as b = (theta, a0, a1, b0, b1): with
u(theta) = (cos(theta), sin(theta)) and v(theta) = (-sin(theta), cos(theta)), a point x has
coordinates a(x) = <x, u(theta)> and b(x) = <x, v(theta)> in the frame of R, and
R = [a0, a1] x [b0, b1] in these coordinates. Its side lengths are a1 - a0 and b1 - b0; the
manuscript names them w <= h, and since every quantity computed here is symmetric in the two axes,
the code does not reorder them.

Functions:
  hull_pts(P)        vertices of conv(P),
  box_at(P, theta)   smallest rectangle with orientation theta enclosing P; its sides are the widths
                     w_P(theta) and w_P(theta + pi/2) of Equation (1), and its area is A_P(theta),
  area(b)            area of a rectangle,
  mbr(P)             MBR(P), a rectangle of area A*(P) = min over theta of A_P(theta),
  corners(b, q)      corners of R, or of R^(q) (same centre and orientation, sides (1 + 2q) w and
                     (1 + 2q) h),
  pca_hull_box(H)    rectangle aligned with the principal axes of the region conv(H)
                     (Dimitrov et al.),
  range_mbr_2022(X)  RANGE-MBR of the points X (Section 3.2, Proposition 1),
  protrusion(K, b)   Euclidean distance from each point of K to R.
"""
import numpy as np


# conv(P), Andrew's monotone chain
def hull_pts(P):
    """Vertices of the convex hull of the points P (Andrew's monotone chain), counterclockwise."""
    P = np.unique(np.asarray(P, float), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1])


def box_at(P, th):
    """Smallest rectangle with orientation th enclosing the points P."""
    P = np.asarray(P, float)
    c, s = np.cos(th), np.sin(th)
    a = P[:, 0] * c + P[:, 1] * s
    t = -P[:, 0] * s + P[:, 1] * c
    return (float(th), float(a.min()), float(a.max()), float(t.min()), float(t.max()))


def area(b):
    return (b[2] - b[1]) * (b[4] - b[3])


# MBR(P): by Freeman and Shapira, a side of an optimal rectangle is collinear with an edge of conv(P)
def mbr(P):
    """Exact minimum-area enclosing rectangle: by Freeman and Shapira, one side of an optimal
    rectangle is collinear with an edge of the convex hull, so the edge directions are tried."""
    H = hull_pts(P)
    if len(H) < 3:
        if len(H) == 2:
            d = H[1] - H[0]
            return box_at(H, np.arctan2(d[1], d[0]))
        return box_at(H, 0.0)
    E = np.roll(H, -1, axis=0) - H
    ths = np.mod(np.arctan2(E[:, 1], E[:, 0]), np.pi / 2)
    best = None
    for th in np.unique(np.round(ths, 12)):
        b = box_at(H, th)
        if best is None or area(b) < area(best):
            best = b
    return best


def corners(b, q=0.0):
    """Corners of b, or of its proportional enlargement b^(q) when q > 0."""
    th = b[0]
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-np.sin(th), np.cos(th)])
    la, lb = b[2] - b[1], b[4] - b[3]
    a0, a1, b0, b1 = b[1] - q * la, b[2] + q * la, b[3] - q * lb, b[4] + q * lb
    return np.array([a0 * u + b0 * v, a1 * u + b0 * v, a1 * u + b1 * v, a0 * u + b1 * v])


# area, centroid and second central moments of the polygon region with vertices H
def polygon_moments(H):
    """Area, centroid and second central moments of the polygon with vertices H."""
    x, y = H[:, 0], H[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    Ixx = ((x * x + x * x1 + x1 * x1) * cr).sum() / (12 * A) - cx * cx
    Iyy = ((y * y + y * y1 + y1 * y1) * cr).sum() / (12 * A) - cy * cy
    Ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / (24 * A) - cx * cy
    return A, cx, cy, np.array([[Ixx, Ixy], [Ixy, Iyy]])


def pca_hull_box(H):
    """Rectangle aligned with the principal axes of the convex hull region (Dimitrov et al.)."""
    H = hull_pts(H)
    _, _, _, C = polygon_moments(H)
    w, V = np.linalg.eigh(C)
    u = V[:, 1]
    return box_at(H, float(np.mod(np.arctan2(u[1], u[0]), np.pi / 2)))


# RANGE-MBR: the angle theta* minimising V_s V_t is a principal axis of the empirical covariance Gamma (Proposition 1)
def range_mbr_2022(X):
    """RANGE-MBR (Zaidi, 2022): the angle minimising the product of the variances of the points
    projected on the two axes, which is a principal axis of their empirical covariance
    (Proposition 1), and the enclosing rectangle of the points at that angle."""
    X = np.asarray(X, float)
    C = np.cov(X.T, bias=True)
    w, V = np.linalg.eigh(C)
    u = V[:, 1]
    return box_at(X, float(np.mod(np.arctan2(u[1], u[0]), np.pi / 2)))


def protrusion(K, b):
    """Euclidean distance from each point of K to the rectangle b."""
    th = b[0]
    c, s = np.cos(th), np.sin(th)
    a = K[:, 0] * c + K[:, 1] * s
    t = -K[:, 0] * s + K[:, 1] * c
    dx = np.maximum(np.maximum(a - b[2], b[1] - a), 0)
    dy = np.maximum(np.maximum(t - b[4], b[3] - t), 0)
    return np.hypot(dx, dy)
