"""Step 10 (Section 4.9, Table 9): results by category for the fused proposal.

For each category: number of retained reference objects in the images outside the development
subset, matched-proposal recall, number of evaluation objects, mean uncovered fraction of the expert
DOTA labels and of the uncalibrated fused proposal L(0), share of objects already contained
(s(K, R) = 0), and, with class-conditional margins q_hat_y at alpha = 0.10 (Corollary 1), the
image-weighted coverage, the median margin, the share of splits with a finite margin and the median
area ratio (1 + 2 q_hat_y)^2 area(R) / A*(K).

Usage:  python 10_per_class.py    (reads results/riskgrid.json, scores.json and match.json)
"""
import json, numpy as np, collections
import shapely
from shapely.geometry import Polygon
G=json.load(open('results/riskgrid.json')); Q=np.array(G['qgrid']); O=G['objects']
ME=['det','sam','fuse','sam_pca','sam_range']
allimgs=sorted({o['img'] for o in O}); dev=set(np.random.default_rng(7).choice(allimgs, len(allimgs)//4, replace=False))
O=[o for o in O if o['img'] not in dev and all(m in o for m in ME)]
S=json.load(open('results/scores.json')); M=json.load(open('results/match.json'))
gt_eval=[s for s in S if s['img'] not in dev]
ev={o['id'] for o in O}
dota_unc={}
for r in M:
    if r['id'] in ev and 'dota' in r:
        K=Polygon(r['hull']); q=Polygon(r['dota']['quad']).buffer(0); dota_unc[r['id']]=1-K.intersection(q).area/K.area
imgs=sorted({o['img'] for o in O}); cls=np.array([o['cls'] for o in O]); classes=sorted(set(cls))
byimg=collections.defaultdict(list)
for i,o in enumerate(O): byimg[o['img']].append(i)
# q_hat_y = k-th smallest score within the category, Equation (30)
def cq(x,a):
    n=len(x); k=int(np.ceil((n+1)*(1-a))); return np.inf if k>n else float(np.sort(x)[k-1])
m='fuse'; s=np.array([o[m]['s'] for o in O]); raw=np.array([o[m]['raw'] for o in O]); L0=np.array([o[m]['L'][0] for o in O])
rng=np.random.default_rng(2026); qc=collections.defaultdict(list); ac=collections.defaultdict(list); cc=collections.defaultdict(list)
for _ in range(200):
    p=rng.permutation(len(imgs)); h=len(imgs)//2; cal=[imgs[i] for i in p[:h]]; test=[imgs[i] for i in p[h:]]
    for c in classes:
        # one object of category y per calibration image containing that category (Corollary 1)
        sc=[s[rng.choice([i for i in byimg[j] if cls[i]==c])] for j in cal if any(cls[i]==c for i in byimg[j])]
        q=cq(np.array(sc),0.10)
        per=[]; ars=[]
        for j in test:
            ids=[i for i in byimg[j] if cls[i]==c]
            if ids: per.append(np.mean(s[ids]<=q)); ars+=list((1+2*q)**2*raw[ids])
        if per: cc[c].append(np.mean(per)); qc[c].append(q); ac[c].append(np.median(ars) if np.isfinite(q) else np.inf)
NAMES={'plane':'Plane','ship':'Ship','storage tank':'Storage tank','baseball diamond':'Baseball diamond','tennis court':'Tennis court','basketball court':'Basketball court','ground track field':'Ground track field','harbor':'Harbor','bridge':'Bridge','large vehicle':'Large vehicle','small vehicle':'Small vehicle','helicopter':'Helicopter','roundabout':'Roundabout','soccer ball field':'Soccer field','swimming pool':'Swimming pool'}
print('class | GT objects (eval imgs) | recall % | eval objects | DOTA uncov mean % | fuse uncov mean % | fuse exact % | mondrian cov % | q median | share finite q | area median')
rows=[]
for c in sorted(classes,key=lambda c:NAMES[c]):
    g=[x for x in gt_eval if x['cls']==c]; rec=100*np.mean(['det' in x for x in g])
    ids=np.flatnonzero(cls==c); du=[dota_unc[O[i]['id']] for i in ids if O[i]['id'] in dota_unc]
    qs=np.array(qc[c]); fin=np.isfinite(qs); A_=np.array(ac[c])
    print(f"{NAMES[c]} | {len(g)} | {rec:.1f} | {len(ids)} | {100*np.mean(du):.2f} | {100*L0[ids].mean():.2f} | {100*np.mean(s[ids]==0):.1f} | {100*np.mean(cc[c]):.1f} | {np.median(qs[fin]) if fin.any() else float('nan'):.3f} | {100*fin.mean():.0f} | {np.median(A_[np.isfinite(A_)]) if np.isfinite(A_).any() else float('nan'):.2f}")
g=gt_eval; print('All', len(g), f"{100*np.mean(['det' in x for x in g]):.1f}", len(O), f"{100*np.mean(list(dota_unc.values())):.2f}", f"{100*L0.mean():.2f}", f"{100*np.mean(s==0):.1f}")
