"""Step 14 (Sections 4.4 and 4.7): object-level (pooled) calibration and inner-score statistics.

Pooled calibration (Section 4.4, "Object-level calibration fails"). As a check of the role of
dependence, the margin of Equation (30) is computed from all eligible objects of the calibration
images pooled, as in the standard object-level split procedure, instead of one object per image:
marginally, and within each category (class-conditional). Printed, for the fused proposal and the
detected rectangle at alpha = 0.10 over the 200 splits: image-weighted and object-weighted coverage
and the smallest per-category coverage, with the category that attains it. The pooled objects do
not form an exchangeable sequence, since objects of one image are dependent (Definition 1 and the
paragraph that follows it), and Figure 6 shows the resulting undercoverage by category.
Inner scores (Section 4.7): over all matched objects with an inner proposal, the share with
r(K, P_in) = 0 (P_in inside K), and the median and 90th percentile of r(K, P_in).

Usage:  python 14_pooled_and_inner.py    (reads results/riskgrid.json and results/inner2.json)
"""
import json, collections
import numpy as np

G = json.load(open('results/riskgrid.json'))
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
O5 = [o for o in G['objects'] if all(m in o for m in ME)]
allimgs = sorted({o['img'] for o in O5})
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))   # development subset
O = [o for o in O5 if o['img'] not in dev]                                                # evaluation set
imgs = sorted({o['img'] for o in O}); cls = np.array([o['cls'] for o in O]); classes = sorted(set(cls))
byimg = collections.defaultdict(list)
for i, o in enumerate(O): byimg[o['img']].append(i)


def cq(x, a):
    """q_hat = s_(k), k = ceil((n + 1)(1 - alpha)); +infinity when k > n (Equation (30))."""
    n = len(x); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if k > n else float(np.sort(x)[k - 1])


rng = np.random.default_rng(2026)
splits = []
for _ in range(200):
    p = rng.permutation(len(imgs)); h = len(imgs) // 2
    splits.append(([imgs[i] for i in p[:h]], [imgs[i] for i in p[h:]]))

print('== Pooled (object-level) calibration, alpha = 0.10')
for m in ('fuse', 'det'):
    s = np.array([o[m]['s'] for o in O])
    for mode in ('marginal', 'per class'):
        ci, co = [], []; pc = collections.defaultdict(list)
        for cal, test in splits:
            C = np.concatenate([byimg[j] for j in cal])            # all calibration objects pooled
            if mode == 'marginal':
                q = np.full(len(s), cq(s[C], 0.10))
            else:
                q = np.zeros(len(s))
                for c in classes: q[cls == c] = cq(s[C][cls[C] == c], 0.10)
            cov = s <= q                                            # K in R^(q_hat) iff s(K, R) <= q_hat (Lemma 4)
            T = np.concatenate([byimg[j] for j in test])
            co.append(cov[T].mean()); ci.append(np.mean([cov[byimg[j]].mean() for j in test]))
            for c in classes:
                per = [cov[[i for i in byimg[j] if cls[i] == c]].mean() for j in test if any(cls[i] == c for i in byimg[j])]
                if per: pc[c].append(np.mean(per))
        worst = min(pc, key=lambda c: np.mean(pc[c]))
        print('%s, %s: image-weighted %.1f%%, object-weighted %.1f%%, smallest category %s %.1f%%'
              % (m, mode, 100 * np.mean(ci), 100 * np.mean(co), worst, 100 * np.mean(pc[worst])))
        if mode == 'marginal' and m == 'fuse':
            print('  baseball diamond under the pooled marginal margin: %.1f%%' % (100 * np.mean(pc['baseball diamond'])))

print('\n== Inner scores r(K, P_in) over all matched objects with an inner proposal')
r = np.array([a['r'] for a in json.load(open('results/inner2.json'))])
print('objects %d; r = 0: %.1f%%; median %.2f; 90th percentile %.2f' % (len(r), 100 * np.mean(r == 0), np.median(r), np.percentile(r, 90)))
