"""Step 1 (Section 3.12, "Oriented detector"): tiled YOLO11m-OBB inference.

Each DOTA v1.0 validation image is cut into 1024 x 1024 tiles with a stride of 768 pixels.
Detections with confidence >= 0.1 are mapped back to image coordinates, and duplicates across tiles
are removed by class-wise non-maximum suppression with a polygon IoU threshold of 0.5.

Usage:  python 01_detect.py IMG_DIR OUT_DIR yolo11m-obb.pt
Output: one JSON file per image, {file, W, H, dets}, where each detection has the four corners of
        its oriented rectangle ("poly"), its confidence ("score") and its class ("cls").
        These detected rectangles are the first proposal mechanism of the paper.
"""
import sys, os, json, glob, time
import numpy as np
from shapely.geometry import Polygon
from shapely import STRtree
from ultralytics import YOLO

IMG_DIR, OUT_DIR, WEIGHTS = sys.argv[1], sys.argv[2], sys.argv[3]
TILE, STRIDE, CONF = 1024, 768, 0.10          # tile size, stride and confidence threshold (Appendix A)
os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(WEIGHTS)
names = model.names


def tiles(W, H):
    """Top-left corners of the tiles covering a W x H image; the last row and column are shifted
    inwards so that every tile lies inside the image."""
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE)) or [0]
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)) or [0]
    if xs[-1] + TILE < W: xs.append(W - TILE)
    if ys[-1] + TILE < H: ys.append(H - TILE)
    return [(max(x, 0), max(y, 0)) for y in ys for x in xs]


def rnms(polys, scores, cls, thr=0.5):
    """Class-wise non-maximum suppression on oriented rectangles, by decreasing confidence:
    a rectangle is removed when its polygon IoU with a kept rectangle of the same class exceeds thr."""
    keep = []
    order = np.argsort(-scores)
    P = [Polygon(p).buffer(0) for p in polys]
    tree = STRtree(P)
    removed = np.zeros(len(P), bool)
    for i in order:
        if removed[i]: continue
        keep.append(i)
        for j in tree.query(P[i]):
            if j == i or removed[j] or cls[j] != cls[i]: continue
            inter = P[i].intersection(P[j]).area
            u = P[i].area + P[j].area - inter
            if u > 0 and inter / u > thr: removed[j] = True
        removed[i] = True
    return keep


files = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
t0 = time.time()
for k, f in enumerate(files):
    stem = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(OUT_DIR, stem + '.json')
    if os.path.exists(out): continue            # resumable: images already processed are skipped
    import cv2
    im = cv2.imread(f)
    H, W = im.shape[:2]
    polys, scores, cls = [], [], []
    T = tiles(W, H)
    crops = []
    for (x, y) in T:
        c = im[y:y + TILE, x:x + TILE]
        if c.shape[0] < TILE or c.shape[1] < TILE:          # images smaller than a tile are padded
            c = cv2.copyMakeBorder(c, 0, TILE - c.shape[0], 0, TILE - c.shape[1], cv2.BORDER_CONSTANT, value=0)
        crops.append(c)
    for b in range(0, len(crops), 4):
        res = model.predict(crops[b:b + 4], imgsz=TILE, conf=CONF, iou=0.5, verbose=False, max_det=1000)
        for (x, y), r in zip(T[b:b + 4], res):
            if r.obb is None or len(r.obb) == 0: continue
            q = r.obb.xyxyxyxy.cpu().numpy() + np.array([x, y])   # tile -> image coordinates
            polys += list(q); scores += list(r.obb.conf.cpu().numpy()); cls += list(r.obb.cls.cpu().numpy().astype(int))
    dets = []
    if polys:
        scores = np.array(scores); cls = np.array(cls)
        for i in rnms(polys, scores, cls):
            dets.append(dict(poly=np.round(polys[i], 2).tolist(), score=round(float(scores[i]), 4), cls=names[int(cls[i])]))
    json.dump(dict(file=os.path.basename(f), W=W, H=H, dets=dets), open(out, 'w'))
    if k % 20 == 0: print(k, stem, len(dets), round(time.time() - t0), 's', flush=True)
print('done', round(time.time() - t0), 's')
