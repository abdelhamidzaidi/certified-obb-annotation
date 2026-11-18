"""Step 7 (Sections 3.6, 3.9, 4.2, 4.5, 4.7 and 4.8): further analyses on the evaluation set.

A. Training-conditional coverage (Proposition 3, Section 4.5, Table 7). For the fused proposal and
   the detected rectangle at alpha = 0.10 and 0.05: the distribution over splits of the coverage
   C(q_hat) of the marginal margin q_hat = s_(k), compared with Beta(k, n + 1 - k); and the
   training-conditional margin q_hat^delta = s_(k_delta), where k_delta is the smallest k <= n with
   P(Beta(k, n + 1 - k) < 1 - alpha) <= delta (Equation (34)), for delta = 0.10 and 0.05.
B. Selective certification (Section 3.9, Corollary 3, Section 4.7). The agreement statistic g is the
   IoU between the detected rectangle and MBR(P) (step 5). The threshold t is the 20% (or 10%)
   quantile of g on the development subset, so that 80% (90%) of its proposals are accepted.
   Margins are calibrated on one accepted object per calibration image (Corollary 3) and, for
   comparison, separately within the accepted and deferred strata.
C. Sensitivity to matching (Section 4.8): the evaluation restricted to objects whose detection has
   IoU >= 0.7 with MBR(K).
D. Matched precision (Section 4.2): share of the detections matched to a retained reference object.

Usage:  python 07_extra_analyses.py    (reads results/riskgrid.json, features.json, scores.json,
                                         det_m/; writes results/extra.json)
"""
import json, numpy as np, collections, glob
from scipy import stats
B = ''
G = json.load(open(B + 'results/riskgrid.json')); Q = np.array(G['qgrid'])
F = {f['id']: f['f'] for f in json.load(open(B + 'results/features.json'))}
S = {s['id']: s for s in json.load(open(B + 'results/scores.json'))}
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
Oall = [o for o in G['objects'] if all(m in o for m in ME)]
allimgs = sorted({o['img'] for o in G['objects']})
# development subset (Section 3.13), excluded from the evaluation
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))
Odev = [o for o in Oall if o['img'] in dev]
O = [o for o in Oall if o['img'] not in dev]
res = {}

# q_hat = s_(k), k = ceil((n + 1)(1 - alpha)); +infinity when k > n (Equation (30))
def cq(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if k > n else float(np.sort(x)[k - 1])

# 200 image-level splits; within each stratum, one object per calibration image (Theorem 3, Corollaries 1 and 3)
def run_splits(O, key, strata_fn=None, a=0.10, nsplit=200, seed=2026, pac=None):
    """Image-level split conformal; strata_fn(o) -> stratum label or None (not certified)."""
    rng = np.random.default_rng(seed)
    imgs = sorted({o['img'] for o in O})
    byimg = collections.defaultdict(list)
    for i, o in enumerate(O): byimg[o['img']].append(i)
    s = np.array([o[key]['s'] for o in O]); raw = np.array([o[key]['raw'] for o in O])
    st = np.array([strata_fn(o) if strata_fn else 'all' for o in O], dtype=object)
    labels = sorted({x for x in st if x is not None})
    out = collections.defaultdict(list)
    for _ in range(nsplit):
        p = rng.permutation(len(imgs)); h = len(imgs) // 2
        cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
        q = np.full(len(s), np.nan)
        for lab in labels:
            sc = []
            for j in cal:
                ids = [i for i in byimg[j] if st[i] == lab]
                if ids: sc.append(s[rng.choice(ids)])
            sc = np.array(sc)
            if pac is None: qq = cq(sc, a)
            else:
                n = len(sc); d = pac
                # smallest k with P(Beta(k, n+1-k) >= 1-a) >= 1-d
                # k_delta: smallest k with P(Beta(k, n + 1 - k) < 1 - alpha) <= delta (Equation (34))
                ks = [k for k in range(1, n + 1) if stats.beta.sf(1 - a, k, n + 1 - k) >= 1 - d]
                qq = float(np.sort(sc)[ks[0] - 1]) if ks else np.inf
            q[st == lab] = qq
            out['n_' + str(lab)].append(len(sc))
        for lab in labels:
            per = []; ar = []
            for j in test:
                ids = [i for i in byimg[j] if st[i] == lab]
                if ids: per.append(np.mean(s[ids] <= q[ids])); ar += list((1 + 2 * q[ids]) ** 2 * raw[ids])
            ar = np.array(ar)
            out['cov_' + str(lab)].append(np.mean(per)); out['area_' + str(lab)].append(np.median(ar[np.isfinite(ar)]))
            out['q_' + str(lab)].append(q[st == lab][0])
        acc = np.array([x is not None for x in st])
        T = np.concatenate([byimg[j] for j in test])
        out['accept_rate'].append(acc[T].mean())
        # image-weighted coverage over all certified objects
        per = [np.mean(s[[i for i in byimg[j] if acc[i]]] <= q[[i for i in byimg[j] if acc[i]]]) for j in test if any(acc[i] for i in byimg[j])]
        out['cov_all'].append(np.mean(per))
        ar = (1 + 2 * q[T][acc[T]]) ** 2 * raw[T][acc[T]]; out['area_all'].append(np.median(ar[np.isfinite(ar)]))
    return {k: v for k, v in out.items()}

def summ(d):
    r = {}
    for k, v in d.items():
        v = np.array(v, float)
        r[k] = dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), sd=float(np.std(v)))
    return r

# Table 7: observed spread of the coverage against the Beta(k, n + 1 - k) law, and the margin q_hat^delta
# ---------------- A. training-conditional coverage
A = {}
for m in ('fuse', 'det'):
    for a in (0.10, 0.05):
        d = run_splits(O, m, a=a)
        n = int(np.median(d['n_all'])); k = int(np.ceil((n + 1) * (1 - a)))
        A[f'{m}|{a}'] = dict(emp=summ(d), n=n, k=k, beta_mean=k / (n + 1),
                             beta_p5=float(stats.beta.ppf(0.05, k, n + 1 - k)), beta_p95=float(stats.beta.ppf(0.95, k, n + 1 - k)),
                             beta_sd=float(stats.beta.std(k, n + 1 - k)))
        for dl in (0.10, 0.05):
            dp = run_splits(O, m, a=a, pac=dl)
            A[f'{m}|{a}|pac{dl}'] = summ(dp)
            A[f'{m}|{a}|pac{dl}']['share_below'] = float(np.mean(np.array(dp['cov_all']) < 1 - a))
        A[f'{m}|{a}']['share_below'] = float(np.mean(np.array(d['cov_all']) < 1 - a))
        print('A', m, a, 'n', n, 'k', k, 'emp cov', A[f'{m}|{a}']['emp']['cov_all'], 'beta', A[f'{m}|{a}']['beta_p5'], A[f'{m}|{a}']['beta_p95'],
              'share<1-a', A[f'{m}|{a}']['share_below'], 'PAC0.1', A[f'{m}|{a}|pac0.1']['cov_all'], A[f'{m}|{a}|pac0.1']['area_all'], A[f'{m}|{a}|pac0.1']['share_below'], flush=True)
res['A'] = A

# ---------------- B. selective certification by agreement
# agreement statistic g of Section 3.9
agree = lambda o: F[o['id']]['iou_ds']
dev_ag = np.array([agree(o) for o in Odev]); dev_s = np.array([o['fuse']['s'] for o in Odev])
Bres = {}
for target in (0.8, 0.9):
    # threshold t fixed on the development subset (0.600 for 80% acceptance, 0.529 for 90%)
    t = float(np.quantile(dev_ag, 1 - target))            # accept the target share of dev proposals
    acc_dev = dev_ag >= t
    Bres[f'thr{target}'] = dict(threshold=t, dev_accept=float(acc_dev.mean()),
                               dev_s_p90_acc=float(np.quantile(dev_s[acc_dev], 0.9)), dev_s_p90_def=float(np.quantile(dev_s[~acc_dev], 0.9)))
    for m in ('fuse', 'det'):
        sel = run_splits(O, m, strata_fn=lambda o: 'acc' if agree(o) >= t else None)
        two = run_splits(O, m, strata_fn=lambda o: 'acc' if agree(o) >= t else 'def')
        Bres[f'thr{target}|{m}|select'] = summ(sel); Bres[f'thr{target}|{m}|two'] = summ(two)
        print('B', target, m, 't', round(t, 3), 'accept', summ(sel)['accept_rate']['mean'], 'sel cov', summ(sel)['cov_acc']['mean'], 'area', summ(sel)['area_acc']['mean'],
              '| two-strata cov acc/def', summ(two)['cov_acc']['mean'], summ(two)['cov_def']['mean'], 'area acc/def/all', summ(two)['area_acc']['mean'], summ(two)['area_def']['mean'], summ(two)['area_all']['mean'], flush=True)
base = {m: summ(run_splits(O, m)) for m in ('fuse', 'det')}
Bres['base'] = base
print('B base', {m: (base[m]['cov_all']['mean'], base[m]['area_all']['mean']) for m in base})
res['B'] = Bres

# ---------------- C. matching sensitivity
# Section 4.8: stricter matching, IoU(detection, MBR(K)) >= 0.7
O7 = [o for o in O if S[o['id']].get('det_iou', 0) >= 0.7]
C = dict(n=len(O7), share=len(O7) / len(O))
for m in ('fuse', 'det'):
    d = summ(run_splits(O7, m)); C[m] = d
    s = np.array([o[m]['s'] for o in O7]); L0 = np.array([o[m]['L'][0] for o in O7])
    C[m + '_raw'] = dict(exact=float(np.mean(s == 0)), unc=float(L0.mean()))
    print('C', m, len(O7), d['cov_all']['mean'], d['area_all']['mean'], C[m + '_raw'], flush=True)
res['C'] = C

# ---------------- D. detection precision
# Section 4.2: matched precision = matched detections / all detections
ndet = sum(len(json.load(open(f))['dets']) for f in glob.glob(B + 'results/det_m/*.json'))
nmatch = sum('det' in s for s in S.values())
conf = np.array([s['score'] for s in S.values() if 'det' in s])
res['D'] = dict(detections=ndet, matched=nmatch, precision=nmatch / ndet)
print('D', res['D'])
json.dump(res, open(B + 'results/extra.json', 'w'), indent=1)
