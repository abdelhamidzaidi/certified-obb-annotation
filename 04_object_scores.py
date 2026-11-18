"""Step 4 (Sections 3.12, 3.13 and 4.2): per-object quantities for every retained object.

For every retained object (matched or not) with reference hull K, this script records:
  lK, sK        long and short sides of MBR(K),
  dota          for objects with an expert label: its area ratio and protrusion outside K,
  score, det_iou  confidence of the matched detection and its IoU with MBR(K) (Section 3.12);
                det_iou >= 0.7 defines the stricter population of Section 4.8,
and, for the proposal rectangles R of step 3, alternative scores examined on the development subset
only (Section 3.13, "Development subset"):
  s_cov         Chebyshev protrusion e(K, R) = max(e_a, e_b) of Section 3.5 divided by the long side
                l(R) = h, i.e. a margin proportional to the long side, which was not retained,
  s_H           Hausdorff distance between K and the proposal region divided by l(R),
  eu            largest Euclidean protrusion of K outside R.
The quantities used in the reported results are the matched-proposal recall (Section 4.2, Table 9),
det_iou (Section 4.8) and the detection confidences (precision, Section 4.2).

Usage:  python 04_object_scores.py results/match.json results/scores.json
"""
import sys, json, numpy as np
import shapely
from shapely.geometry import Polygon
import geometry as E
from relevant import relevant_points
from multiprocessing import Pool

# Chebyshev protrusion e(K, R) = max(e_a, e_b) of Section 3.5, Equation (24)
def cheb(K, b):
    th = b[0]; c, s = np.cos(th), np.sin(th)
    a = K[:, 0]*c + K[:, 1]*s; t = -K[:, 0]*s + K[:, 1]*c
    dx = np.maximum(np.maximum(a - b[2], b[1] - a), 0); dy = np.maximum(np.maximum(t - b[4], b[3] - t), 0)
    return float(np.maximum(dx, dy).max())

def corners(b):
    th = b[0]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    return np.array([b[1]*u + b[3]*v, b[2]*u + b[3]*v, b[2]*u + b[4]*v, b[1]*u + b[4]*v])

# Hausdorff distance between K and the proposal region
def haus(K, Kpoly, P, Ppoly):
    d1 = shapely.distance(shapely.points(K), Ppoly).max()
    d2 = shapely.distance(shapely.points(P), Kpoly).max()
    return float(max(d1, d2))

def entry(K, Kpoly, b, P, As):
    w, h = b[2]-b[1], b[4]-b[3]; l = max(w, h)
    Ppoly = Polygon(P).buffer(0)
    return dict(w=w, h=h, l=l, area=w*h/As, s_cov=cheb(K, b)/l, s_H=haus(K, Kpoly, P, Ppoly)/l, eu=float(E.protrusion(K, b).max()))

def run(r):
    K = np.array(r['hull']); Kpoly = Polygon(K); As = r['Astar']
    out = dict(img=r['img'], id=r['id'], cls=r['cls'], Astar=As)
    # long side lK and short side sK of MBR(K)
    bK = E.mbr(K); out['lK'] = max(bK[2]-bK[1], bK[4]-bK[3]); out['sK'] = min(bK[2]-bK[1], bK[4]-bK[3])
    # expert DOTA label (human reference of Table 5)
    if 'dota' in r:
        q = np.array(r['dota']['quad']); qp = Polygon(q).buffer(0)
        out['dota'] = dict(prot=float(shapely.distance(shapely.points(K), qp).max()), area=qp.area/As,
                           difficult=r['dota']['difficult'], s_H=haus(K, Kpoly, q, qp)/out['lK'])
    if 'det' in r:
        p = np.array(r['det']['poly']); b = E.mbr(p)
        # matched detection: confidence and IoU with MBR(K) (Section 4.8 uses det_iou >= 0.7)
        out['score'] = r['det']['score']; out['det_iou'] = r['det']['iou']
        out['det'] = entry(K, Kpoly, b, corners(b), As)
        if 'sam' in r:
            parts = [np.array(c) for c in r['sam']]
            M = np.vstack(parts); H = E.hull_pts(M)
            if len(H) >= 3:
                b2 = E.mbr(H); out['sam'] = entry(K, Kpoly, b2, H, As)
                b3 = E.pca_hull_box(H); out['sam_pca'] = entry(K, Kpoly, b3, corners(b3), As)
                X = relevant_points(parts, tol=1.0, merge=1.0)
                if len(X) >= 3:
                    b4 = E.range_mbr_2022(X); out['sam_range'] = entry(K, Kpoly, b4, corners(b4), As)
                    out['sam_range']['npts'] = len(X)
                # does the SAM mask hull lie inside the object?
                # largest distance from the mask hull P to K: P is an inner proposal only if this is 0
                out['sam']['out_of_K'] = float(shapely.distance(shapely.points(H), Kpoly).max())
    return out

if __name__ == '__main__':
    R = json.load(open(sys.argv[1]))
    with Pool(2) as pool: S = pool.map(run, R, chunksize=200)
    json.dump(S, open(sys.argv[2], 'w'))
    print('done', len(S), sum('det' in s for s in S), sum('sam' in s for s in S))
