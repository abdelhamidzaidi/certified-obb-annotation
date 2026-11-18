"""Step 2 (Sections 3.11 and 3.12): reference shapes, matching and segmentation.

Reference shapes (Section 3.11). For each iSAID validation instance with outline O (the union of its
polygons), K = conv(O) is its convex hull and A*(K) the area of its minimum-area rectangle MBR(K).
Instances with A*(K) >= 25 px^2 and a shorter side of MBR(K) of at least 3 px are retained
(103,524 instances, Table 2).

Matching (Section 3.12, "Matching"). A detection is matched to a retained object of the same class
when the IoU between the detected rectangle and MBR(K) is at least 0.5; matching is greedy by
decreasing confidence and one-to-one. Matched objects are the eligible objects of Section 3.6.
The expert DOTA quadrilaterals are matched to the same objects in the same way, greedily by
decreasing IoU (human reference of Table 5).

Segmentation (Section 3.12, "Segmentation"). Each matched detection prompts MobileSAM with the
axis-aligned box of the detected rectangle on a 1024 x 1024 window (larger windows are reduced to
1024 pixels). Mask components smaller than 5% of the largest are discarded; the remaining outlines
are stored, and their convex hull is the mask proposal P used in steps 3-5.

Usage:  python 02_match_segment.py iSAID_val.json DET_DIR LABEL_DIR IMG_DIR results/match.json sam
Output: one record per retained object: {img, id, cls, Astar = A*(K), hull = vertices of K,
        mbr = corners of MBR(K), dota (if matched), det (if matched: rectangle, confidence, IoU),
        sam (mask outlines, if segmented)}. Per-image files in OUT.d make the run resumable.
"""
import sys, os, json, glob, time
import numpy as np
import cv2
import torch
torch.set_num_threads(1)
from shapely.geometry import Polygon
from shapely import STRtree
import geometry as E

ISAID, DET_DIR, LBL_DIR, IMG_DIR, OUT = sys.argv[1:6]
USE_SAM = len(sys.argv) > 6 and sys.argv[6] == 'sam'
TILE, STRIDE = 1024, 768
# iSAID category names -> DOTA category names (the 15 categories are the same)
MAP = {'plane': 'plane', 'ship': 'ship', 'storage_tank': 'storage tank', 'baseball_diamond': 'baseball diamond',
       'tennis_court': 'tennis court', 'basketball_court': 'basketball court', 'Ground_Track_Field': 'ground track field',
       'Harbor': 'harbor', 'Bridge': 'bridge', 'Large_Vehicle': 'large vehicle', 'Small_Vehicle': 'small vehicle',
       'Helicopter': 'helicopter', 'Roundabout': 'roundabout', 'Soccer_ball_field': 'soccer ball field',
       'Swimming_pool': 'swimming pool'}

# corners of a rectangle b = (theta, a0, a1, b0, b1) given in its own frame
def rect_poly(b):
    th = b[0]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    return np.array([b[1]*u + b[3]*v, b[2]*u + b[3]*v, b[2]*u + b[4]*v, b[1]*u + b[4]*v])

def iou(a, b):
    i = a.intersection(b).area; u = a.area + b.area - i
    return i / u if u > 0 else 0.0

d = json.load(open(ISAID))
imgs = {im['id']: im for im in d['images']}
gt = {}
for a in d['annotations']:
    # outline O of the instance: union of its polygons
    parts = [np.array(s, float).reshape(-1, 2) for s in a['segmentation'] if len(s) >= 6]
    if not parts: continue
    # K = conv(O)
    P = np.vstack(parts); H = E.hull_pts(P)
    if len(H) < 3: continue
    # MBR(K), A*(K) and the shorter side of MBR(K)
    b = E.mbr(H); As = E.area(b); sh = min(b[2]-b[1], b[4]-b[3])
    # size filter of Section 3.11
    if As < 25 or sh < 3: continue
    stem = imgs[a['image_id']]['file_name'][:-4]
    gt.setdefault(stem, []).append(dict(id=a['id'], cls=MAP[a['category_name']], hull=H, mbr=rect_poly(b), Astar=As))

sam = None
if USE_SAM:
    from ultralytics import SAM
    sam = SAM(os.environ.get('MOBILE_SAM_WEIGHTS', 'mobile_sam.pt'))

# 1024 x 1024 window (or larger, for large objects) containing the prompt box
def window_for(box, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(TILE, x2 - x1 + 64, y2 - y1 + 64)
    X0 = int(np.clip(cx - side / 2, 0, max(W - side, 0))); Y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
    # snap small objects to the regular grid so several prompts share one image embedding
    if side == TILE:
        X0 = int(min((X0 // 256) * 256, max(W - TILE, 0))); Y0 = int(min((Y0 // 256) * 256, max(H - TILE, 0)))
        if X0 + TILE < x2: X0 = int(min(x2 - TILE + 16, max(W - TILE, 0)))
        if Y0 + TILE < y2: Y0 = int(min(y2 - TILE + 16, max(H - TILE, 0)))
    return (X0, Y0, int(side))

OUTDIR = OUT + '.d'; os.makedirs(OUTDIR, exist_ok=True)
t0 = time.time()
files = sorted(glob.glob(os.path.join(DET_DIR, '*.json')))
for k, f in enumerate(files):
    D = json.load(open(f)); stem = D['file'][:-4]; W, H = D['W'], D['H']
    if os.path.exists(os.path.join(OUTDIR, stem + '.json')): continue
    out = []
    G = gt.get(stem, [])
    gpoly = [Polygon(g['mbr']).buffer(0) for g in G]
    tree = STRtree(gpoly) if gpoly else None
    # greedy one-to-one matching by decreasing detection confidence, IoU(detection, MBR(K)) >= 0.5
    taken = set(); pairs = []
    dets = sorted(D['dets'], key=lambda x: -x['score'])
    for det in dets:
        if tree is None: break
        pd = Polygon(det['poly']).buffer(0)
        best, bi = 0.5, None
        for j in tree.query(pd):
            if j in taken or G[j]['cls'] != det['cls']: continue
            v = iou(pd, gpoly[j])
            if v >= best: best, bi = v, j
        if bi is not None:
            taken.add(bi); pairs.append((det, bi, best))
    # DOTA expert labels
    # expert DOTA labels, matched greedily by decreasing IoU with MBR(K)
    dota = {}
    lf = os.path.join(LBL_DIR, stem + '.txt')
    if os.path.exists(lf) and tree is not None:
        Q = []
        for line in open(lf):
            t = line.split()
            if len(t) < 9: continue
            try: q = np.array(t[:8], float).reshape(4, 2)
            except ValueError: continue
            Q.append((q, t[8].replace('-', ' '), int(t[9]) if len(t) > 9 else 0))
        cand = []
        for qi, (q, c, diff) in enumerate(Q):
            pq = Polygon(q).buffer(0)
            if pq.area <= 0: continue
            for j in tree.query(pq):
                if G[j]['cls'] != c: continue
                v = iou(pq, gpoly[j])
                if v >= 0.5: cand.append((v, qi, j))
        used_q, used_g = set(), set()
        for v, qi, j in sorted(cand, reverse=True):
            if qi in used_q or j in used_g: continue
            used_q.add(qi); used_g.add(j); dota[j] = (Q[qi][0], Q[qi][2], v)
    # SAM on matched detections
    # MobileSAM prompted with the axis-aligned box of each matched detection
    masks = {}
    if sam is not None and pairs:
        im = cv2.imread(os.path.join(IMG_DIR, stem + '.jpg'))
        groups = {}
        for pi, (det, j, v) in enumerate(pairs):
            p = np.array(det['poly'])
            box = [max(p[:, 0].min(), 0), max(p[:, 1].min(), 0), min(p[:, 0].max(), W - 1), min(p[:, 1].max(), H - 1)]
            groups.setdefault(window_for(box, W, H), []).append((pi, box))
        for (X0, Y0, S), L in groups.items():
            crop = im[Y0:Y0 + S, X0:X0 + S]
            f_ = 1.0
            if max(crop.shape[:2]) > TILE:
                f_ = TILE / max(crop.shape[:2])
                crop = cv2.resize(crop, (int(round(crop.shape[1] * f_)), int(round(crop.shape[0] * f_))), interpolation=cv2.INTER_AREA)
            boxes = [[(b[0] - X0) * f_, (b[1] - Y0) * f_, (b[2] - X0) * f_, (b[3] - Y0) * f_] for _, b in L]
            for c0 in range(0, len(boxes), 16):
                r = sam(crop, bboxes=boxes[c0:c0 + 16], verbose=False)[0]
                if r.masks is None: continue
                M = r.masks.data.cpu().numpy().astype(np.uint8)
                del r
                for (pi, _), m in zip(L[c0:c0 + 16], M):
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if not cs: continue
                    amax = max(cv2.contourArea(c) for c in cs)
                    # discard mask components smaller than 5% of the largest
                    keep = [c for c in cs if cv2.contourArea(c) >= 0.05 * amax and len(c) >= 3]
                    polys = []
                    for c in keep:
                        c = cv2.approxPolyDP(c, 0.5, True).reshape(-1, 2).astype(float)
                        if len(c) >= 3: polys.append((c / f_ + [X0, Y0]).round(1).tolist())
                    if polys: masks[pi] = polys
    for j, g in enumerate(G):
        rec = dict(img=stem, id=g['id'], cls=g['cls'], Astar=g['Astar'], hull=g['hull'].round(2).tolist(), mbr=g['mbr'].round(2).tolist())
        if j in dota:
            rec['dota'] = dict(quad=dota[j][0].tolist(), difficult=dota[j][1], iou=round(dota[j][2], 4))
        out.append(rec)
    idx = {g['id']: r for g, r in zip(G, out[len(out) - len(G):])}
    for pi, (det, j, v) in enumerate(pairs):
        r = idx[G[j]['id']]
        r['det'] = dict(poly=det['poly'], score=det['score'], iou=round(v, 4))
        if pi in masks: r['sam'] = masks[pi]
    json.dump(out, open(os.path.join(OUTDIR, stem + '.json'), 'w'))
    if k % 20 == 0: print(k, stem, len(G), 'gt', len(pairs), 'tp', len(masks), 'masks', round(time.time() - t0), 's', flush=True)
allr = []
for f in sorted(glob.glob(os.path.join(OUTDIR, '*.json'))): allr += json.load(open(f))
json.dump(allr, open(OUT, 'w'))
print('done', len(allr), round(time.time() - t0), 's')
