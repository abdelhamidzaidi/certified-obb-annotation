"""Step 6 (Sections 3.6, 3.7, 3.13, 4.4 and 4.6): image-level calibration and risk control.

Evaluation set (Section 3.13, Table 2). Objects with all five proposals; the development subset of
112 images, drawn with numpy.random.default_rng(7) from the 449 images containing such objects, is
excluded, leaving 337 images and 23,429 objects. 200 random splits into a calibration half
I_1, ..., I_n (n = 168) and a test half are drawn with numpy.random.default_rng(2026).

Coverage (Section 3.6, Definition 1, Theorem 3 and Corollary 1). The images, each with all its
objects and proposals, are assumed exchangeable (Definition 1); the image, not the object, is the
unit of calibration. In each calibration image I_j one eligible object is drawn uniformly at random and its outer score s_j = s(K, R) is kept. With
k = ceil((n + 1)(1 - alpha)), the margin is q_hat = s_(k), the k-th smallest score, and
q_hat = +infinity when k > n (Equation (30)). The class-conditional margin q_hat_y applies the same
rule within each category, using the calibration images that contain that category. The calibrated
label of a test object is R^(q_hat), which contains K iff s(K, R) <= q_hat (Lemma 4).
Reported (Table 6): image-weighted coverage (mean over test images of the share of their objects
covered, the quantity of Theorem 3), object-weighted coverage, smallest per-category coverage,
median area ratio area(R^(q_hat)) / A*(K) = (1 + 2 q_hat)^2 area(R) / A*(K) over objects with a
finite margin, median margin, and the share of objects with an infinite margin.

Risk control (Section 3.7, Theorem 4). For a calibration image I_j, L_j(q) is the mean uncovered
fraction L(q) of its eligible objects. q_hat_beta is the smallest q in Q with
(1/(n+1)) sum_j L_j(q) + 1/(n+1) <= beta (Equation (37)); when no grid value qualifies, the formal
margin is +infinity, and the summaries replace it by the largest grid value, 0.4 (capped output,
marked with a dagger in Table 8; "feasible" is the share of splits with a finite margin).
Reported (Table 8): realised image-weighted uncovered fraction (mean and 95th percentile over
splits), object-weighted fraction, median area ratio, margin, share fully covered.

The coverage and risk curves for Figure 8 use the first 50 splits.

Usage:  python 06_calibrate.py results/riskgrid.json results/calib2.json 200
"""
import sys, json, numpy as np, collections
G = json.load(open(sys.argv[1])); OUT = sys.argv[2]; NSPLIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
Q = np.array(G['qgrid']); O = G['objects']
METHODS = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
O = [o for o in O if all(m in o for m in METHODS)]
allimgs = sorted({o['img'] for o in O})
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))
O = [o for o in O if o['img'] not in dev]
imgs = sorted({o['img'] for o in O}); classes = sorted({o['cls'] for o in O})
img = np.array([o['img'] for o in O]); cls = np.array([o['cls'] for o in O])
byimg = collections.defaultdict(list)
for i, o in enumerate(O): byimg[o['img']].append(i)
print('evaluation objects', len(O), 'images', len(imgs), 'excluded dev images', len(dev), flush=True)

# q_hat = s_(k) with k = ceil((n + 1)(1 - alpha)), or +infinity when k > n (Equation (30))
def cq(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if k > n else float(np.sort(x)[k - 1])

res = dict(n_objects=len(O), n_images=len(imgs), n_dev=len(dev), qgrid=Q.tolist(), classes=classes)
# splits and one-object-per-image selections (Appendix A)
rng = np.random.default_rng(2026)
splits = []
for _ in range(NSPLIT):
    # calibration half (n = 168 images) and test half (169 images)
    p = rng.permutation(len(imgs)); h = len(imgs) // 2
    splits.append(([imgs[i] for i in p[:h]], [imgs[i] for i in p[h:]]))

for m in METHODS:
    s = np.array([o[m]['s'] for o in O]); raw = np.array([o[m]['raw'] for o in O]); L = np.array([o[m]['L'] for o in O])
    res[f'{m}|raw'] = dict(exact=float(np.mean(s == 0)), uncovered_mean=float(L[:, 0].mean()),
                           uncovered_median=float(np.median(L[:, 0])), area_median=float(np.median(raw)),
                           area_mean=float(raw.mean()), area_p5=float(np.percentile(raw, 5)), area_p95=float(np.percentile(raw, 95)),
                           s_median=float(np.median(s)), s_p90=float(np.percentile(s, 90)), s_p95=float(np.percentile(s, 95)))
    # ---- coverage
    # coverage at alpha = 0.10 and 0.05, marginal (Theorem 3) and class-conditional (Corollary 1)
    for a in (0.10, 0.05):
        for mode in ('marginal', 'mondrian'):
            cov_img, cov_obj, area_med, area_mean, qs, ninf = [], [], [], [], [], []
            pc = collections.defaultdict(list)
            for cal, test in splits:
                if mode == 'marginal':
                    # one eligible object drawn uniformly at random in each calibration image
                    sc = [s[rng.choice(byimg[j])] for j in cal]
                    q = np.full(len(s), cq(np.array(sc), a))
                else:
                    # class-conditional margins q_hat_y
                    q = np.zeros(len(s))
                    for c in classes:
                        sc = []
                        for j in cal:
                            ids = [i for i in byimg[j] if cls[i] == c]
                            if ids: sc.append(s[rng.choice(ids)])
                        q[cls == c] = cq(np.array(sc), a)
                T = np.concatenate([byimg[j] for j in test])
                # K contained in R^(q_hat) iff s(K, R) <= q_hat (Lemma 4)
                cov = s <= q
                cov_obj.append(cov[T].mean())
                # image-weighted coverage: mean over test images of the share of their objects covered
                cov_img.append(np.mean([cov[byimg[j]].mean() for j in test]))
                # area ratio area(R^(q_hat)) / A*(K) = (1 + 2 q_hat)^2 area(R) / A*(K)
                ar = (1 + 2 * q[T]) ** 2 * raw[T]
                fin = np.isfinite(ar); ninf.append(1 - fin.mean())
                area_med.append(np.median(ar[fin])); area_mean.append(ar[fin].mean()); qs.append(np.median(q[T][fin]))
                for c in classes:
                    per = [cov[[i for i in byimg[j] if cls[i] == c]].mean() for j in test if any(cls[i] == c for i in byimg[j])]
                    if per: pc[c].append(np.mean(per))
            res[f'{m}|cov|{mode}|{a}'] = dict(cov_img=float(np.mean(cov_img)), cov_img_p5=float(np.percentile(cov_img, 5)),
                cov_img_p95=float(np.percentile(cov_img, 95)), cov_obj=float(np.mean(cov_obj)),
                cov_obj_p5=float(np.percentile(cov_obj, 5)), cov_obj_p95=float(np.percentile(cov_obj, 95)),
                area_median=float(np.mean(area_med)), area_mean=float(np.mean(area_mean)), q_median=float(np.mean(qs)),
                share_infinite=float(np.mean(ninf)), per_class={c: float(np.mean(v)) for c, v in pc.items()},
                per_class_min=float(min(np.mean(v) for v in pc.values())))
            print(m, 'cov', mode, a, {k: round(v, 4) for k, v in res[f'{m}|cov|{mode}|{a}'].items() if isinstance(v, float)}, flush=True)
    # ---- risk control
    # risk control at beta = 0.05, 0.02, 0.01 (Theorem 4)
    for beta in (0.01, 0.02, 0.05):
        risk_img, risk_obj, area_med, area_mean, qs, feas = [], [], [], [], [], []
        for cal, test in splits:
            # L_j(q): mean uncovered fraction of the eligible objects of calibration image I_j
            Lj = np.array([L[byimg[j]].mean(0) for j in cal]); n = len(cal)
            # Equation (37): (1/(n+1)) sum_j L_j(q) + 1/(n+1) <= beta
            ok = np.flatnonzero(n / (n + 1) * Lj.mean(0) + 1 / (n + 1) <= beta)
            if len(ok): k = ok[0]; feas.append(1)
            # no admissible grid value: formal margin +infinity, capped at 0.4 in the summaries
            else: k = len(Q) - 1; feas.append(0)
            T = np.concatenate([byimg[j] for j in test])
            risk_obj.append(L[T, k].mean()); risk_img.append(np.mean([L[byimg[j], k].mean() for j in test]))
            ar = (1 + 2 * Q[k]) ** 2 * raw[T]; area_med.append(np.median(ar)); area_mean.append(ar.mean()); qs.append(Q[k])
        res[f'{m}|risk|{beta}'] = dict(risk_img=float(np.mean(risk_img)), risk_img_p95=float(np.percentile(risk_img, 95)),
            risk_obj=float(np.mean(risk_obj)), area_median=float(np.mean(area_med)), area_mean=float(np.mean(area_mean)),
            q_median=float(np.median(qs)), q_p5=float(np.percentile(qs, 5)), q_p95=float(np.percentile(qs, 95)),
            feasible=float(np.mean(feas)), cov_exact=float(np.mean([np.mean(s[np.concatenate([byimg[j] for j in t])] <= qq) for (c_, t), qq in zip(splits, qs)])))
        print(m, 'risk', beta, {k: round(v, 4) for k, v in res[f'{m}|risk|{beta}'].items()}, flush=True)
    # ---- trade-off curve (marginal coverage, image-level), for the figure
    # Figure 8a: coverage against median area, marginal margins at several alpha (first 50 splits)
    curve = []
    for a in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05):
        cv, am = [], []
        for cal, test in splits[:50]:
            q = cq(np.array([s[rng.choice(byimg[j])] for j in cal]), a)
            T = np.concatenate([byimg[j] for j in test])
            cv.append(np.mean([np.mean(s[byimg[j]] <= q) for j in test])); am.append(np.median((1 + 2 * q) ** 2 * raw[T]))
        curve.append(dict(alpha=a, cov=float(np.mean(cv)), area=float(np.mean(am))))
    # Figure 8b: realised uncovered fraction against median area, risk control at several beta
    rcurve = []
    for beta in (0.10, 0.05, 0.03, 0.02, 0.01):
        rk, am = [], []
        for cal, test in splits[:50]:
            Lj = np.array([L[byimg[j]].mean(0) for j in cal]); n = len(cal)
            ok = np.flatnonzero(n / (n + 1) * Lj.mean(0) + 1 / (n + 1) <= beta); k = ok[0] if len(ok) else len(Q) - 1
            T = np.concatenate([byimg[j] for j in test])
            rk.append(np.mean([L[byimg[j], k].mean() for j in test])); am.append(np.median((1 + 2 * Q[k]) ** 2 * raw[T]))
        rcurve.append(dict(beta=beta, risk=float(np.mean(rk)), area=float(np.mean(am))))
    res[f'{m}|curve'] = curve; res[f'{m}|rcurve'] = rcurve
json.dump(res, open(OUT, 'w'), indent=1)
