"""Relevant points of an outline (Definition 1 of Zaidi 2022; Section 3.12, "Methods compared").

The relevant points are the input of RANGE-MBR in the baseline "RANGE-MBR on mask points". After
Douglas-Peucker simplification of each contour with tolerance tol = 1 pixel, they are: the four
axis-extreme points of the outline; the local extrema of each coordinate along the contour; the
corners, at which the outline turns by more than 30 degrees; the local maxima of the turning angle
(discrete curvature); and the inflection points, at which the turn changes sign. Turns smaller than
5 degrees are treated as straight, and points closer than merge = 1 pixel are merged.
"""
import numpy as np
import cv2


# signed turning angle (degrees) at each vertex of the closed contour C
def _turns(C):
    d1 = C - np.roll(C, 1, axis=0)
    d2 = np.roll(C, -1, axis=0) - C
    a = np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1))
    return np.degrees(a)


def relevant_points(parts, tol=1.0, merge=1.0, corner=30.0, straight=5.0):
    pts = []
    allP = []
    for P in parts:
        P = np.asarray(P, float)
        if len(P) < 3:
            continue
        C = cv2.approxPolyDP(P.reshape(-1, 1, 2).astype(np.float32), tol, True).reshape(-1, 2).astype(float)
        if len(C) < 3:
            C = P
        allP.append(C)
        n = len(C)
        t = _turns(C)
        t_eff = np.where(np.abs(t) < straight, 0.0, t)
        for k in range(n):
            p, q = (k - 1) % n, (k + 1) % n
            for j in (0, 1):
                x = C[:, j]
                if (x[k] > x[p] and x[k] >= x[q]) or (x[k] < x[p] and x[k] <= x[q]):
                    pts.append(C[k])
            # corner: turn larger than 30 degrees
            if abs(t_eff[k]) > corner:
                pts.append(C[k])
            if abs(t_eff[k]) > 0 and abs(t_eff[k]) >= abs(t_eff[p]) and abs(t_eff[k]) >= abs(t_eff[q]):
                pts.append(C[k])
            # inflection point: sign change of the turn
            if t_eff[k] != 0 and t_eff[q] != 0 and np.sign(t_eff[k]) != np.sign(t_eff[q]):
                pts.append((C[k] + C[q]) / 2)
    if not allP:
        return np.zeros((0, 2))
    # four axis-extreme points of the whole outline
    A = np.vstack(allP)
    for j in (0, 1):
        pts.append(A[A[:, j].argmin()])
        pts.append(A[A[:, j].argmax()])
    X = np.array(pts)
    keep = []
    for p in X:
        if all(np.hypot(*(p - k)) >= merge for k in keep):
            keep.append(p)
    return np.array(keep)
