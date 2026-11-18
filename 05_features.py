"""Step 5 (Sections 3.9, 3.13 and 4.7): observable features of each proposal.

These quantities are computed from the proposals alone, without the reference hull K, and can
therefore be used at labelling time:
  iou_ds        agreement statistic g of selective certification (Section 3.9): IoU between the
                detected rectangle and MBR(P), the rectangle of the mask hull P,
  conf          detector confidence,
  m_out_*, d_out_*  per-axis excess of P outside the detected rectangle and of the detected
                rectangle outside MBR(P), relative to the side lengths,
  dth, logl, aspect, sam_fill  angle difference, size, aspect ratio and fill ratio.
The features other than iou_ds enter only the local-scale and affine margins tried and rejected on
the development subset (Section 3.13). For each of det, sam and fuse, the excesses e_a, e_b of K
beyond R (Equation (28)) and the side lengths w, h are stored as targets.

Usage:  python 05_features.py results/match.json results/features.json
"""
import sys, json, numpy as np
from shapely.geometry import Polygon
import geometry as E
from multiprocessing import Pool

# side lengths (la = w, lb = h) of R and largest excesses e_a, e_b of the points K beyond R, Equation (28)
def ex(K, b):
    th = b[0]; c, s = np.cos(th), np.sin(th)
    a = K[:, 0]*c + K[:, 1]*s; t = -K[:, 0]*s + K[:, 1]*c
    ea = float(max((a - b[2]).max(), (b[1] - a).max(), 0)); eb = float(max((t - b[4]).max(), (b[3] - t).max(), 0))
    return dict(la=float(b[2]-b[1]), lb=float(b[4]-b[3]), ea=ea, eb=eb, th=float(b[0]))

def corners(b):
    th = b[0]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    return np.array([b[1]*u + b[3]*v, b[2]*u + b[3]*v, b[2]*u + b[4]*v, b[1]*u + b[4]*v])

def run(r):
    if 'det' not in r or 'sam' not in r: return None
    K = np.array(r['hull']); As = r['Astar']
    p = np.array(r['det']['poly']); bd = E.mbr(p)
    M = np.vstack([np.array(c) for c in r['sam']]); H = E.hull_pts(M)
    if len(H) < 3: return None
    # MBR(P) and the fused proposal MBR(detected rectangle U P)
    bs = E.mbr(H); bf = E.mbr(np.vstack([corners(bd), H]))
    Pd, Ps = Polygon(corners(bd)).buffer(0), Polygon(corners(bs)).buffer(0)
    # agreement statistic g = IoU(detected rectangle, MBR(P)), Section 3.9
    inter = Pd.intersection(Ps).area; iou = inter / (Pd.area + Ps.area - inter)
    m_out = ex(H, bd); d_out = ex(corners(bd), bs)
    dth = abs(((bd[0] - bs[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    f = dict(conf=r['det']['score'], iou_ds=iou,
             m_out_a=m_out['ea'] / m_out['la'], m_out_b=m_out['eb'] / m_out['lb'],
             d_out_a=d_out['ea'] / d_out['la'], d_out_b=d_out['eb'] / d_out['lb'],
             dth=float(np.degrees(dth)), logl=float(np.log(max(bf[2]-bf[1], bf[4]-bf[3]))),
             aspect=float(max(bf[2]-bf[1], bf[4]-bf[3]) / max(min(bf[2]-bf[1], bf[4]-bf[3]), 1e-6)),
             sam_fill=float(Polygon(H).area / max(Ps.area, 1e-9)))
    return dict(img=r['img'], id=r['id'], cls=r['cls'], Astar=As, f=f, det=ex(K, bd), sam=ex(K, bs), fuse=ex(K, bf))

if __name__ == '__main__':
    R = json.load(open(sys.argv[1]))
    with Pool(2) as pool: A = [a for a in pool.map(run, R, chunksize=300) if a]
    json.dump(A, open(sys.argv[2], 'w')); print('done', len(A))
