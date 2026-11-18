"""Step 12 (Section 4.8): automatic consistency check of the matching.

For each matched pair (detection, reference object K), the largest IoU between the detected
rectangle and MBR(K') over the other retained reference objects K' of the same class in the same
image is computed. A pair is ambiguous when this runner-up IoU reaches the matching threshold 0.5
(another object would also pass the criterion of Section 3.12) and contested when it reaches 0.3.
Printed: the number of such pairs, the median and 10th percentile of the IoU of the matches, and the
share of ambiguous pairs per category. The check is geometric only; it does not detect errors in the
reference annotations themselves.

Usage:  python 12_audit_matching.py    (reads results/match.json)
"""
import json, numpy as np, collections
from shapely.geometry import Polygon
from shapely.strtree import STRtree
R=json.load(open('results/match.json'))
byimg=collections.defaultdict(list)
# reference objects grouped by image and class
for r in R: byimg[(r['img'],r['cls'])].append(r)
iou_m=[]; run=[]; ncls=collections.Counter(); amb_cls=collections.Counter()
for key,L in byimg.items():
    polys=[Polygon(r['mbr']).buffer(0) for r in L]; tree=STRtree(polys)
    for i,r in enumerate(L):
        if 'det' not in r: continue
        # runner-up IoU of the detected rectangle with MBR(K') for K' other than K
        d=Polygon(r['det']['poly']).buffer(0); best=0.0
        for j in tree.query(d):
            j=int(j)
            if j==i: continue
            u=d.union(polys[j]).area
            if u>0: best=max(best,d.intersection(polys[j]).area/u)
        iou_m.append(r['det']['iou']); run.append(best); ncls[r['cls']]+=1
        if best>=0.5: amb_cls[r['cls']]+=1
iou_m=np.array(iou_m); run=np.array(run)
print('matched pairs',len(run))
print("n>=0.5",int(np.sum(run>=0.5)),"n>=0.3",int(np.sum(run>=0.3)))
print('match IoU median %.3f  p10 %.3f'%(np.median(iou_m),np.percentile(iou_m,10)))
print('margin (match - runner-up) < 0.1: %.2f%%'%(100*np.mean(iou_m-run<0.1)))
print('runner-up exceeds match IoU: %.2f%%'%(100*np.mean(run>iou_m)))
for c,n in ncls.most_common(): print('  %-20s %6d  ambiguous %.2f%%'%(c,n,100*amb_cls[c]/n))
