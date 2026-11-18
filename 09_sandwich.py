"""Step 9 (Section 3.8, Theorem 5; Section 4.7): statistical sandwich for the fused proposal.

On each split, the same randomly selected object of each calibration image provides an outer score
s(K, R) (step 3) and an inner score r(K, P_in) (step 8). q_hat is computed from the outer scores at
level alpha_1 and r_hat from the inner scores at level alpha_2 by Equation (30). For a test object,
the interval of Equation (43) is
    (1 - min(r_hat, 1))^2 A*(P_in)  <=  A*(K)  <=  (1 + 2 q_hat)^2 area(R),
valid with probability at least 1 - alpha_1 - alpha_2 for a uniformly selected eligible object.
Printed for (alpha_1, alpha_2) = (0.05, 0.05), (0.10, 0.10) and (0.05, 0.15): the image-weighted
frequency of the joint event {s <= q_hat and r <= r_hat}, the share of objects whose interval
contains A*(K), and the median ratio of the two ends (Section 4.7).

Usage:  python 09_sandwich.py    (reads results/riskgrid.json and results/inner2.json)
"""
import json, numpy as np, collections
B = ''
G=json.load(open(B+'results/riskgrid.json')); I={a['id']:a for a in json.load(open(B+'results/inner2.json'))}
ME=['det','sam','fuse','sam_pca','sam_range']
Oall=[o for o in G['objects'] if all(m in o for m in ME) and o['id'] in I]
allimgs=sorted({o['img'] for o in G['objects']}); dev=set(np.random.default_rng(7).choice(allimgs, len(allimgs)//4, replace=False))
O=[o for o in Oall if o['img'] not in dev]
imgs=sorted({o['img'] for o in O}); byimg=collections.defaultdict(list)
for i,o in enumerate(O): byimg[o['img']].append(i)
s=np.array([o['fuse']['s'] for o in O]); raw=np.array([o['fuse']['raw'] for o in O])
r=np.array([I[o['id']]['r'] for o in O]); ain=np.array([I[o['id']]['AH']/I[o['id']]['Astar'] for o in O])
# q_hat (or r_hat) = k-th smallest score, k = ceil((n + 1)(1 - alpha)), Equation (30)
def cq(x,a):
    n=len(x); k=int(np.ceil((n+1)*(1-a))); return np.inf if k>n else float(np.sort(x)[k-1])
rng=np.random.default_rng(2026); out=collections.defaultdict(list)
for a1,a2 in ((0.05,0.05),(0.10,0.10),(0.05,0.15)):
    V=[];W=[];Tin=[];Lo=[];Up=[]
    for _ in range(200):
        p=rng.permutation(len(imgs)); h=len(imgs)//2; cal=[imgs[i] for i in p[:h]]; test=[imgs[i] for i in p[h:]]
        # one object per calibration image, shared by the outer and inner calibrations
        pick=[rng.choice(byimg[j]) for j in cal]
        q=cq(s[pick],a1); rh=cq(r[pick],a2)
        # joint event of Theorem 5
        ok=(s<=q)&(r<=rh)
        V.append(np.mean([ok[byimg[j]].mean() for j in test]))
        T=np.concatenate([byimg[j] for j in test])
        # ends of the interval (42), in units of A*(K)
        lo=(1-min(rh,1))**2*ain[T]; up=(1+2*q)**2*raw[T]       # in units of A*(K)
        Tin.append(np.mean([np.mean(((1-min(rh,1))**2*ain[byimg[j]]<=1)&(1<=(1+2*q)**2*raw[byimg[j]])) for j in test]))
        ratio=up/np.maximum(lo,1e-12); W.append(np.median(ratio)); Lo.append(np.median(lo)); Up.append(np.median(up))
    print(a1,a2,'joint event (img)',round(np.mean(V),4),'p5',round(np.percentile(V,5),4),'| A* inside interval',round(np.mean(Tin),4),
          '| median certified ratio up/lo',round(np.mean(W),2),'| median lower',round(np.mean(Lo),3),'upper',round(np.mean(Up),3))
