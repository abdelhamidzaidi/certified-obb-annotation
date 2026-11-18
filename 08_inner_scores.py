"""Step 8 (Sections 3.8 and 4.7): inner proposals and inner scores for the statistical sandwich.

The inner proposal is P_in = conv(P intersected with the detected rectangle), where P is the hull of
the SAM mask (Section 3.8); c is its centroid and P_in^[r] = c + (1 - r)(P_in - c) its homothetic
contraction (Section 3.1). The inner score is
    r(K, P_in) = min{ r in [0, 1] : P_in^[r] contained in K },
with r = 1 when c is not in K. Since K is convex, P_in^[r] lies in K iff its vertices do; for a
vertex x, the largest admissible factor 1 - r is the length of the part of the segment [c, x] inside
K divided by |x - c|, and r(K, P_in) is obtained from the most restrictive vertex.
Stored per object: r = r(K, P_in), AH = A*(P_in) and Astar = A*(K); by Theorem 5,
A*(P_in^[r]) = (1 - r)^2 A*(P_in).

Usage:  python 08_inner_scores.py results/match.json results/inner2.json
"""
import sys, json, numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString
import geometry as E
from multiprocessing import Pool

def run(r):
    if 'det' not in r or 'sam' not in r: return None
    K = Polygon(r['hull']).buffer(0)
    H0 = E.hull_pts(np.vstack([np.array(c) for c in r['sam']]))
    if len(H0) < 3: return None
    # P_in = conv(P intersected with the detected rectangle)
    I = Polygon(H0).buffer(0).intersection(Polygon(np.array(r['det']['poly'])).buffer(0))
    if I.is_empty or I.area <= 0: return None
    I = I.convex_hull; H = np.array(I.exterior.coords[:-1])
    # centroid c of P_in
    Hp = Polygon(H); c = np.array(Hp.centroid.coords[0])
    if not K.contains(Point(c)):
        # convention r = 1 when c is not in K
        rr = 1.0
    else:
        ts = []
        for x in H:
            d = np.linalg.norm(x - c)
            if d < 1e-9: continue
            seg = LineString([c, x]); inter = seg.intersection(K)
            # largest factor 1 - r for which the contracted vertex stays in K
            ts.append(min(1.0, inter.length / d))
        rr = float(1 - min(ts)) if ts else 0.0
    # A*(P_in)
    bH = E.mbr(H)
    return dict(id=r['id'], r=rr, AH=float(E.area(bH)), Astar=r['Astar'])

if __name__ == '__main__':
    R = json.load(open(sys.argv[1]))
    with Pool(2) as pool: A = [a for a in pool.map(run, R, chunksize=300) if a]
    json.dump(A, open(sys.argv[2], 'w')); print('done', len(A))
