"""Step 3 (Sections 3.6, 3.7 and 3.12): proposal rectangles, outer scores and uncovered-area curves.

For every matched object with reference hull K and mask proposal P (hull of the SAM outlines), five
proposal rectangles R are computed (Section 3.12, "Methods compared"):
  det        detected rectangle (its exact MBR),
  sam        MBR(P), the exact minimum-area rectangle of the mask hull,
  fuse       fused proposal MBR(detected rectangle U P),
  sam_pca    rectangle aligned with the principal axes of P (Dimitrov et al.),
  sam_range  RANGE-MBR on the relevant points of the mask outline (Definition 1 of Zaidi 2022).
For each R, in the frame of R = [a0, a1] x [b0, b1] with side lengths w = a1 - a0 and h = b1 - b0:
  e_a, e_b   largest excesses of K beyond R along the two axes (Equation (28)),
  s(K, R)    outer score max(e_a / w, e_b / h) (Equation (29)); K is contained in R^(q) iff s <= q
             (Lemma 4),
  L(q)       uncovered fraction area(K minus R^(q)) / area(K) (Section 3.7), on the margin grid Q of
             Appendix A, made nonincreasing in q,
  raw        area(R) / A*(K), the area ratio of the uncalibrated proposal (Table 5).
R^(q) has the centre and orientation of R and sides (1 + 2q) w and (1 + 2q) h.
The labelling of the two axes does not affect s or L, which are symmetric in them.

Usage:  python 03_score_curves.py results/match.json results/riskgrid.json
Output: {qgrid = Q, objects = [{img, id, cls, Astar, conf, det, sam, fuse, sam_pca, sam_range}]},
        each method holding s, raw, L (list over Q), la = w and lb = h.
"""
import sys, json, numpy as np
import shapely
from shapely.geometry import Polygon
import geometry as E
from relevant import relevant_points
from multiprocessing import Pool

# margin grid Q: 21 values on [0, 0.05] and 35 values on [0.06, 0.40] (Appendix A)
QGRID = np.round(np.concatenate([np.linspace(0, 0.05, 21), np.linspace(0.06, 0.40, 35)]), 4)

# corners of R^(q) for R = b = (theta, a0, a1, b0, b1); q = 0 gives R itself
def corners(b, q=0.0):
    th = b[0]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    la, lb = b[2] - b[1], b[4] - b[3]
    a0, a1, b0, b1 = b[1] - q * la, b[2] + q * la, b[3] - q * lb, b[4] + q * lb
    return np.array([a0*u + b0*v, a1*u + b0*v, a1*u + b1*v, a0*u + b1*v])

# outer score s(K, R) = max(e_a / w, e_b / h), Equations (28)-(29)
def axis_score(K, b):
    th = b[0]; c, s = np.cos(th), np.sin(th)
    a = K[:, 0]*c + K[:, 1]*s; t = -K[:, 0]*s + K[:, 1]*c
    la, lb = b[2] - b[1], b[4] - b[3]
    # e_a, e_b: largest excesses of K beyond R along the two axes
    ea = max((a - b[2]).max(), (b[1] - a).max(), 0); eb = max((t - b[4]).max(), (b[3] - t).max(), 0)
    return float(max(ea / la, eb / lb))

def run(r):
    if 'det' not in r or 'sam' not in r: return None
    # K = reference hull, A*(K) = minimum enclosing area
    K = np.array(r['hull']); Kp = Polygon(K); AK = Kp.area; As = r['Astar']
    p = np.array(r['det']['poly']); bd = E.mbr(p)
    parts = [np.array(c) for c in r['sam']]
    # mask proposal P = convex hull of the SAM outlines
    H = E.hull_pts(np.vstack(parts))
    if len(H) < 3: return None
    # the five proposal rectangles R of Section 3.12
    rects = dict(det=bd, sam=E.mbr(H), fuse=E.mbr(np.vstack([corners(bd), H])), sam_pca=E.pca_hull_box(H))
    # relevant points of the mask outline, input of RANGE-MBR
    X = relevant_points(parts, tol=1.0, merge=1.0)
    if len(X) >= 3: rects['sam_range'] = E.range_mbr_2022(X)
    out = dict(img=r['img'], id=r['id'], cls=r['cls'], Astar=As, conf=r['det']['score'])
    for m, b in rects.items():
        # R^(q) for every q in Q, and L(q) = area(K minus R^(q)) / area(K)
        polys = shapely.polygons(np.array([corners(b, q) for q in QGRID]))
        inside = shapely.area(shapely.intersection(Kp, polys))
        L = np.clip(1 - inside / AK, 0, 1)
        L = np.minimum.accumulate(L)
        out[m] = dict(s=axis_score(K, b), raw=float((b[2]-b[1])*(b[4]-b[3]) / As), L=np.round(L, 5).tolist(),
                      la=float(b[2]-b[1]), lb=float(b[4]-b[3]))
    return out

if __name__ == '__main__':
    R = json.load(open(sys.argv[1]))
    R = [r for r in R if 'det' in r and 'sam' in r]
    with Pool(2) as pool: A = [a for a in pool.map(run, R, chunksize=100) if a]
    json.dump(dict(qgrid=QGRID.tolist(), objects=A), open(sys.argv[2], 'w')); print('done', len(A))
