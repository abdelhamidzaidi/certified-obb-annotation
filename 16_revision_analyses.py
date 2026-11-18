"""Step 16: additional analyses of the revised manuscript.

Each block prints the numbers quoted in one place of the manuscript (in brackets):
  A  expert labels and proposals on one common subset of objects            [Table 5, Section 4.3]
  B  omission measured on the instance region O as well as on its hull K     [Section 4.3]
  C  latent (Beta) and test-sample components of the coverage spread       [Table 7, Section 4.5]
  D  image-weighted and object-weighted class-conditional coverage           [Table 9, Section 4.9]
  E  informativeness of the statistical sandwich interval (Theorem 5)        [Section 4.7]
  F  calibration under a change of acquisition source (Google Earth -> GF-2/JL-1)  [Section 4.8]
  G  an eligibility rule observable at deployment (detection confidence)     [Section 4.8]
  H  matched recall as a function of the orientation of MBR(K)               [Section 4.2]
  I  horizontal-or-oriented decision computed from the proposal only         [Section 4.11]
  J  identifiers and coordinates of the objects of Figure 10                 [Table A3]
  K  risk control at beta = 0.01: valid procedure and capped output, separately [Table 8]
  L  class-conditional margins: share of finite labels and coverage among them  [Table 6]
Notation as in the manuscript: K reference hull, O instance region, R proposal rectangle, R^(q) its
proportional enlargement, s = s(K, R) the outer score (Equation (29)), q_hat the margin of
Equation (30), k = ceil((n+1)(1-alpha)).

Usage:  python 16_revision_analyses.py ISAID_JSON DOTA_LABEL_DIR OUT_JSON
"""
import sys, json, collections, math, os
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.stats import beta as Beta
import geometry as E

ISAID, LBL, OUT = sys.argv[1:4]
G = json.load(open('results/riskgrid.json')); Q = np.array(G['qgrid'])
M = {r['id']: r for r in json.load(open('results/match.json'))}
S = json.load(open('results/scores.json'))
I2 = {a['id']: a for a in json.load(open('results/inner2.json'))}
C2 = json.load(open('results/calib2.json'))
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
NAME = dict(det='Detector rectangle', sam='SAM mask rectangle', fuse='Fused proposal',
            sam_pca='Principal axes of the mask', sam_range='RANGE-MBR on mask points')
res = {}

# development subset and evaluation set, exactly as in steps 6, 10, 13 and 15
O5 = [o for o in G['objects'] if all(m in o for m in ME)]
allimgs = sorted({o['img'] for o in O5})
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))
O = [o for o in O5 if o['img'] not in dev]
imgs = sorted({o['img'] for o in O})
byimg = collections.defaultdict(list)
for i, o in enumerate(O): byimg[o['img']].append(i)
cls = np.array([o['cls'] for o in O]); classes = sorted(set(cls))


def cq(x, a):
    """Margin of Equation (30): the k-th smallest score, k = ceil((n+1)(1-alpha)), or +inf if k > n."""
    n = len(x); k = int(np.ceil((n + 1) * (1 - a)))
    return np.inf if k > n else float(np.sort(x)[k - 1])


def splits(images, nsplit=200, seed=2026):
    rng = np.random.default_rng(seed); out = []
    for _ in range(nsplit):
        p = rng.permutation(len(images)); h = len(images) // 2
        out.append(([images[i] for i in p[:h]], [images[i] for i in p[h:]]))
    return out


s = {m: np.array([o[m]['s'] for o in O]) for m in ME}
raw = {m: np.array([o[m]['raw'] for o in O]) for m in ME}
L0 = {m: np.array([o[m]['L'][0] for o in O]) for m in ME}

# ----------------------------------------------------------------------------- A: common subset
print('== A: expert labels and uncalibrated proposals on the objects that have an expert label')
withd = np.array(['dota' in M[o['id']] for o in O])
du, da = [], []
for o in O:
    r = M[o['id']]
    if 'dota' in r:
        K = Polygon(r['hull']); q = Polygon(r['dota']['quad']).buffer(0)
        du.append(max(0.0, 1 - K.intersection(q).area / K.area)); da.append(q.area / r['Astar'])
du, da = np.array(du), np.array(da)
A = dict(n=int(withd.sum()), n_all=len(O))
A['expert'] = dict(contained=float(np.mean(du <= 1e-9)), unc_mean=float(du.mean()), unc_median=float(np.median(du)),
                   area_median=float(np.median(da)), area_p5=float(np.percentile(da, 5)), area_p95=float(np.percentile(da, 95)))
for m in ME:
    x = s[m][withd]; l = L0[m][withd]; a = raw[m][withd]
    A[m] = dict(contained=float(np.mean(x == 0)), unc_mean=float(l.mean()), unc_median=float(np.median(l)),
                area_median=float(np.median(a)), area_p5=float(np.percentile(a, 5)), area_p95=float(np.percentile(a, 95)))
    # same quantities on the objects without an expert label
    A[m + '|nodota'] = dict(contained=float(np.mean(s[m][~withd] == 0)), unc_mean=float(L0[m][~withd].mean()))
for k, v in A.items():
    print(' ', k, v)
res['A'] = A

# ----------------------------------------------------------------------------- B: O versus K
print('\n== B: uncovered fraction of the instance region O and of its hull K')
isaid = json.load(open(ISAID))
want = {o['id'] for o in O}
Opoly = {}
for a in isaid['annotations']:
    if a['id'] not in want: continue
    parts = [Polygon(np.array(p, float).reshape(-1, 2)).buffer(0) for p in a['segmentation'] if len(p) >= 6]
    parts = [p for p in parts if p.area > 0]
    if parts: Opoly[a['id']] = unary_union(parts)
qa = C2['fuse|cov|marginal|0.1']['q_median']
B = collections.defaultdict(list)
for o in O:
    r = M[o['id']]
    if o['id'] not in Opoly: continue
    Oo = Opoly[o['id']]; K = Polygon(r['hull'])
    p = np.array(r['det']['poly']); H = E.hull_pts(np.vstack([np.array(z) for z in r['sam']]))
    bd = E.mbr(p); bf = E.mbr(np.vstack([E.corners(bd), H]))
    labs = dict(det=Polygon(E.corners(bd)), fuse=Polygon(E.corners(bf)), fuse_cal=Polygon(E.corners(bf, qa)))
    if 'dota' in r: labs['expert'] = Polygon(r['dota']['quad']).buffer(0)
    B['ratio_area_O_K'].append(Oo.area / K.area)
    for k, lab in labs.items():
        B[k + '|O'].append(max(0.0, 1 - Oo.intersection(lab).area / Oo.area))
        B[k + '|K'].append(max(0.0, 1 - K.intersection(lab).area / K.area))
Bs = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v)) for k, v in B.items()}
Bs['q'] = qa
for k, v in Bs.items(): print(' ', k, v)
res['B'] = Bs

# ----------------------------------------------------------------------------- C: Beta and test noise
print('\n== C: spread of the image-weighted coverage over splits (fused, marginal margins)')
Cres = {}
for a in (0.10, 0.05):
    rng = np.random.default_rng(2026)            # same interleaved sequence as step 15
    cov, se = [], []
    for _ in range(200):
        p = rng.permutation(len(imgs)); h = len(imgs) // 2
        cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
        q = cq(np.array([s['fuse'][rng.choice(byimg[j])] for j in cal]), a)
        c = np.array([np.mean(s['fuse'][byimg[j]] <= q) for j in test])
        cov.append(c.mean()); se.append(c.std(ddof=1) / np.sqrt(len(c)))      # standard error of the test-half estimate
    n = h; k = int(np.ceil((n + 1) * (1 - a)))
    sd_beta = math.sqrt(k * (n + 1 - k) / ((n + 1) ** 2 * (n + 2)))
    Cres[a] = dict(n=n, k=k, sd_beta=sd_beta, se_test=float(np.mean(se)), sd_pred=math.sqrt(sd_beta ** 2 + np.mean(np.square(se))),
                   sd_obs=float(np.std(cov, ddof=1)), mean_obs=float(np.mean(cov)),
                   beta_mean=k / (n + 1))
    print(' ', a, Cres[a])
res['C'] = Cres

# ----------------------------------------------------------------------------- D: class-conditional targets
print('\n== D: class-conditional margins, image-weighted and object-weighted coverage (fused, alpha 0.10)')
rng = np.random.default_rng(2026)
Dimg, Dobj, Dfin, Nimg = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), {}
for c in classes: Nimg[c] = sum(any(cls[i] == c for i in byimg[j]) for j in imgs)
for _ in range(200):
    p = rng.permutation(len(imgs)); h = len(imgs) // 2; cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
    for c in classes:
        sc = [s['fuse'][rng.choice([i for i in byimg[j] if cls[i] == c])] for j in cal if any(cls[i] == c for i in byimg[j])]
        q = cq(np.array(sc), 0.10)
        per, allc = [], []
        for j in test:
            ids = [i for i in byimg[j] if cls[i] == c]
            if ids: per.append(np.mean(s['fuse'][ids] <= q)); allc += list(s['fuse'][ids] <= q)
        if per:
            Dimg[c].append(np.mean(per)); Dobj[c].append(np.mean(allc)); Dfin[c].append(np.isfinite(q))
D = {}
for c in classes:
    fin = np.array(Dfin[c]); im = np.array(Dimg[c]); ob = np.array(Dobj[c])
    D[c] = dict(images=Nimg[c], objects=int((cls == c).sum()), cov_img=float(im.mean()), cov_obj=float(ob.mean()),
                cov_img_finite=float(im[fin].mean()) if fin.any() else None, cov_obj_finite=float(ob[fin].mean()) if fin.any() else None,
                finite=float(fin.mean()))
    print(' ', c, D[c])
res['D'] = D

# ----------------------------------------------------------------------------- E: sandwich informativeness
print('\n== E: statistical sandwich (fused outer proposal, inner proposal of Section 3.12)')
OI = [i for i, o in enumerate(O) if o['id'] in I2]
idx = collections.defaultdict(list)
for i in OI: idx[O[i]['img']].append(i)
imgsI = sorted(idx)
r_in = np.zeros(len(O)); ain = np.zeros(len(O))
for i in OI:
    a_ = I2[O[i]['id']]; r_in[i] = a_['r']; ain[i] = a_['AH'] / a_['Astar']
rng = np.random.default_rng(2026)
Er = collections.defaultdict(list)
for a1, a2 in ((0.05, 0.05), (0.10, 0.10), (0.05, 0.15)):
    zero, ratio_med, inf2, inf15, lo_med = [], [], [], [], []
    allratio, alllo, qs_, rs_, shape = [], [], [], [], []
    for _ in range(200):
        p = rng.permutation(len(imgsI)); h = len(imgsI) // 2; cal = [imgsI[i] for i in p[:h]]; test = [imgsI[i] for i in p[h:]]
        pick = [rng.choice(idx[j]) for j in cal]
        q = cq(s['fuse'][pick], a1); rh = cq(r_in[pick], a2)
        T = np.concatenate([idx[j] for j in test])
        lo = (1 - min(rh, 1)) ** 2 * ain[T]; up = (1 + 2 * q) ** 2 * raw['fuse'][T]
        zero.append(rh >= 1)
        ratio = np.where(lo > 0, up / np.maximum(lo, 1e-12), np.inf)
        ratio_med.append(np.median(ratio)); lo_med.append(np.median(lo))
        inf2.append(np.mean(ratio <= 2)); inf15.append(np.mean(ratio <= 1.5))
        allratio.append(ratio); alllo.append(lo); qs_.append(q); rs_.append(rh)
        shape.append(raw['fuse'][T] / np.maximum(ain[T], 1e-12))      # area(R) / A*(P_in), the third factor of the ratio
    Er[f'{a1}|{a2}'] = dict(share_zero_lower=float(np.mean(zero)), ratio_median=float(np.median(ratio_med)),
                           lower_median=float(np.median(lo_med)), share_ratio_le2=float(np.mean(inf2)),
                           share_ratio_le15=float(np.mean(inf15)))
    R_ = np.concatenate(allratio); Lo_ = np.concatenate(alllo); Sh_ = np.concatenate(shape)
    # distribution of the relative width up/lo over all test objects and splits, and of the lower end in units of A*(K)
    Er[f'{a1}|{a2}'].update(ratio_q=[float(np.percentile(R_, p)) for p in (10, 25, 50, 75, 90)],
                           share_ratio_ge10=float(np.mean(R_ >= 10)), share_ratio_le3=float(np.mean(R_ <= 3)),
                           share_lower_le01=float(np.mean(Lo_ <= 0.1)), share_lower_le025=float(np.mean(Lo_ <= 0.25)),
                           share_lower_ge05=float(np.mean(Lo_ >= 0.5)),
                           q_median=float(np.median(qs_)), r_hat_median=float(np.median(rs_)),
                           factor_q=float(np.median((1 + 2 * np.array(qs_)) ** 2)), factor_r=float(np.median(1 / (1 - np.minimum(rs_, 0.999)) ** 2)),
                           factor_shape=float(np.median(Sh_)))
    print(' ', a1, a2, Er[f'{a1}|{a2}'])
r_all = r_in[OI]
Er['r_share_eq1'] = float(np.mean(r_all >= 1)); Er['r_median'] = float(np.median(r_all))
print('  inner scores: share r = 1 (centroid outside K or no contraction inside)', Er['r_share_eq1'], 'median', Er['r_median'])
res['E'] = dict(Er)

# ----------------------------------------------------------------------------- F: acquisition source
print('\n== F: calibration on Google Earth images, test on Google Earth or on GF-2 and JL-1 images')
# acquisition source of each image, read from the 'imagesource' line of the DOTA label files; the result
# is also written to results/image_sources.json, which is used when the label files are not available
src = {}
if os.path.isdir(LBL):
    for f in os.listdir(LBL):
        for line in open(os.path.join(LBL, f)):
            if line.startswith('imagesource'): src[f[:-4]] = line.split(':')[1].strip(); break
    json.dump(src, open('results/image_sources.json', 'w'), indent=0, sort_keys=True)
else:
    src = json.load(open('results/image_sources.json'))
ge = [j for j in imgs if src.get(j) == 'GoogleEarth']; sat = [j for j in imgs if src.get(j) in ('GF', 'JL')]
F = dict(n_ge=len(ge), n_sat=len(sat), n_obj_ge=int(sum(len(byimg[j]) for j in ge)), n_obj_sat=int(sum(len(byimg[j]) for j in sat)))
rng = np.random.default_rng(2026)
for m in ('fuse', 'det'):
    cin, cout, qs, ain_, aout = [], [], [], [], []
    for _ in range(200):
        p = rng.permutation(len(ge)); cal = [ge[i] for i in p[:168]]; test = [ge[i] for i in p[168:]]
        q = cq(np.array([s[m][rng.choice(byimg[j])] for j in cal]), 0.10)
        cin.append(np.mean([np.mean(s[m][byimg[j]] <= q) for j in test]))
        cout.append(np.mean([np.mean(s[m][byimg[j]] <= q) for j in sat]))
        Tin = np.concatenate([byimg[j] for j in test]); Tout = np.concatenate([byimg[j] for j in sat])
        ain_.append(np.median((1 + 2 * q) ** 2 * raw[m][Tin])); aout.append(np.median((1 + 2 * q) ** 2 * raw[m][Tout])); qs.append(q)
    F[m] = dict(cov_in=float(np.mean(cin)), cov_in_p5=float(np.percentile(cin, 5)), cov_out=float(np.mean(cout)),
                cov_out_p5=float(np.percentile(cout, 5)), cov_out_p95=float(np.percentile(cout, 95)),
                q=float(np.median(qs)), area_in=float(np.mean(ain_)), area_out=float(np.mean(aout)),
                share_out_below=float(np.mean(np.array(cout) < 0.9)), share_in_below=float(np.mean(np.array(cin) < 0.9)))
    # score distribution in each domain
    sg = np.concatenate([s[m][byimg[j]] for j in ge]); ss = np.concatenate([s[m][byimg[j]] for j in sat])
    F[m]['p90_ge'] = float(np.percentile(sg, 90)); F[m]['p90_sat'] = float(np.percentile(ss, 90))
    # calibration within the satellite images themselves
    cs = []
    for _ in range(200):
        p = rng.permutation(len(sat)); h = len(sat) // 2; cal = [sat[i] for i in p[:h]]; test = [sat[i] for i in p[h:]]
        q = cq(np.array([s[m][rng.choice(byimg[j])] for j in cal]), 0.10)
        cs.append(np.mean([np.mean(s[m][byimg[j]] <= q) for j in test]))
    F[m]['cov_sat_self'] = float(np.mean(cs)); F[m]['n_sat_cal'] = len(sat) // 2
print(' ', F)
res['F'] = F

# ----------------------------------------------------------------------------- G: observable eligibility
print('\n== G: eligibility = detection with confidence >= c (observable at deployment); unmatched detections get s = inf')
det_dir = 'results/det_m'
key = lambda p: tuple(np.round(np.array(p, float).ravel(), 2))
sdet = {}    # matched detection -> score of its detected rectangle
for r in M.values():
    if 'det' in r and r['img'] not in dev:
        K = np.array(r['hull']); bd = E.mbr(np.array(r['det']['poly']))
        th = bd[0]; c_, s_ = np.cos(th), np.sin(th)
        a_ = K[:, 0] * c_ + K[:, 1] * s_; t_ = -K[:, 0] * s_ + K[:, 1] * c_
        ea = max((a_ - bd[2]).max(), (bd[1] - a_).max(), 0); eb = max((t_ - bd[4]).max(), (bd[3] - t_).max(), 0)
        sdet[(r['img'], key(r['det']['poly']))] = max(ea / (bd[2] - bd[1]), eb / (bd[4] - bd[3]))
evimgs = sorted({r['img'] for r in M.values() if r['img'] not in dev})
dets = {}
for j in evimgs:
    f = os.path.join(det_dir, j + '.json')
    if not os.path.exists(f): continue
    dets[j] = [(d['score'], sdet.get((j, key(d['poly'])), np.inf)) for d in json.load(open(f))['dets']]
nref = collections.Counter(r['img'] for r in M.values() if r['img'] not in dev)
Gres = {}
for c in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9):
    D_ = {j: [x for x in v if x[0] >= c] for j, v in dets.items()}
    D_ = {j: v for j, v in D_.items() if v}
    ims = sorted(D_)
    allx = np.array([x[1] for v in D_.values() for x in v])
    prec = float(np.mean(np.isfinite(allx)))
    rec = float(np.sum(np.isfinite(allx)) / sum(nref[j] for j in evimgs))
    rng = np.random.default_rng(2026); cov, qs, fin, succ, yld = [], [], [], [], []
    for _ in range(200):
        p = rng.permutation(len(ims)); h = len(ims) // 2; cal = [ims[i] for i in p[:h]]; test = [ims[i] for i in p[h:]]
        q = cq(np.array([D_[j][rng.integers(len(D_[j]))][1] for j in cal]), 0.10)
        # event of Corollary 4: s~ <= q_hat (true for every detection when q_hat = +inf)
        cov.append(np.mean([np.mean([x[1] <= q for x in D_[j]]) for j in test])); qs.append(q); fin.append(np.isfinite(q))
        # success: a finite label that contains a matched object (Definition 3)
        ok = np.isfinite(q)
        succ.append(np.mean([np.mean([ok and x[1] <= q for x in D_[j]]) for j in test]))
        if ok: yld.append(q)
    Gres[c] = dict(images=len(ims), detections=int(len(allx)), precision=prec, recall=rec, cov=float(np.mean(cov)),
                   finite=float(np.mean(fin)), q_median=float(np.median(qs)), success=float(np.mean(succ)), abstention=float(1 - np.mean(fin)))
    # yield: share of all reference objects outside the development subset whose eligible detection succeeds at
    # the median finite margin (0 when the margin is never finite)
    if yld:
        qm = float(np.median(yld)); Gres[c]['yield'] = float(sum(x[1] <= qm for v in D_.values() for x in v) / sum(nref[j] for j in evimgs))
    else:
        Gres[c]['yield'] = 0.0
    print(' ', c, Gres[c])
res['G'] = {str(k): v for k, v in Gres.items()}

# ----------------------------------------------------------------------------- H: recall by orientation
print('\n== H: matched recall by orientation of the long side of MBR(K) (objects outside the development subset)')
H_ = collections.defaultdict(lambda: [0, 0]); H30 = collections.defaultdict(lambda: [0, 0])
for r in M.values():
    if r['img'] in dev: continue
    c4 = np.array(r['mbr']); e1 = c4[1] - c4[0]; e2 = c4[2] - c4[1]
    e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    ang = math.degrees(math.atan2(e[1], e[0])) % 180; ang = min(ang, 180 - ang)          # 0 = horizontal, 90 = vertical
    b = min(int(ang // 15), 5)
    H_[b][0] += 'det' in r; H_[b][1] += 1
    if min(np.linalg.norm(e1), np.linalg.norm(e2)) >= 30: H30[b][0] += 'det' in r; H30[b][1] += 1
Hres = {f'{15*b}-{15*b+15}': dict(recall=H_[b][0] / H_[b][1], n=H_[b][1], recall30=H30[b][0] / max(H30[b][1], 1), n30=H30[b][1]) for b in sorted(H_)}
for k, v in Hres.items(): print(' ', k, v)
res['H'] = Hres

# ----------------------------------------------------------------------------- I: proposal-based orientation rule
print('\n== I: horizontal-or-oriented decision (tau = 0.05) computed from the fused proposal hull')
tau = 0.05
agree, exc_h, nh, nh_ref, ang_h, ang_exc, ang_n = 0, [], 0, 0, 0, [], 0
for o in O:
    r = M[o['id']]
    K = np.array(r['hull']); AK0 = E.area(E.box_at(K, 0.0)); AKs = r['Astar']
    p = np.array(r['det']['poly']); Hm = E.hull_pts(np.vstack([np.array(z) for z in r['sam']]))
    P = E.hull_pts(np.vstack([E.corners(E.mbr(p)), Hm]))
    bP = E.mbr(P); AP0 = E.area(E.box_at(P, 0.0)); APs = E.area(bP)
    hP = AP0 <= (1 + tau) * APs; hK = AK0 <= (1 + tau) * AKs
    agree += hP == hK; nh_ref += hK
    if hP: nh += 1; exc_h.append(AK0 / AKs - 1)
    th = math.degrees(bP[0]) % 90; th = min(th, 90 - th)
    if th <= 10: ang_n += 1; ang_exc.append(AK0 / AKs - 1)
exc_h = np.array(exc_h); ang_exc = np.array(ang_exc)
Ires = dict(n=len(O), agreement=agree / len(O), horizontal_P=nh / len(O), horizontal_K=nh_ref / len(O),
            share_h_excess_gt_tau=float(np.mean(exc_h > tau)), mean_excess_h=float(exc_h.mean()), share_h_excess_gt_10=float(np.mean(exc_h > 0.10)),
            angle10_horizontal=ang_n / len(O), angle10_share_excess_gt_tau=float(np.mean(ang_exc > tau)), angle10_mean_excess=float(ang_exc.mean()))
print(' ', Ires)
res['I'] = Ires

# ----------------------------------------------------------------------------- J: Figure 10 objects
print('\n== J: objects of Figure 10 (same selection as step 11)')
du_ = {}
for o in O:
    r = M[o['id']]
    if 'dota' in r:
        K = Polygon(r['hull']); q = Polygon(r['dota']['quad']).buffer(0); du_[o['id']] = 1 - K.intersection(q).area / K.area
rng = np.random.default_rng(5); J = []
for c in ['plane', 'ship', 'large vehicle', 'storage tank', 'harbor', 'tennis court']:
    cand = [o for o in O if o['cls'] == c and o['id'] in du_ and 40 <= max(o['fuse']['la'], o['fuse']['lb']) <= 220 and o['fuse']['s'] > 0]
    o = cand[int(rng.integers(len(cand)))]; r = M[o['id']]
    K = np.array(r['hull']); cen = K.mean(0)
    J.append(dict(cls=c, img=o['img'], isaid_id=o['id'], centre=[round(float(cen[0]), 1), round(float(cen[1]), 1)],
                  s_fuse=round(o['fuse']['s'], 4), unc_expert=round(100 * du_[o['id']], 1), unc_fuse=round(100 * o['fuse']['L'][0], 1),
                  hull=r['hull'], dota=r['dota']['quad'], det=r['det']['poly']))
    print('  %s | %s | iSAID id %d | centre (%.1f, %.1f) | s = %.3f' % (c, o['img'], o['id'], cen[0], cen[1], o['fuse']['s']))
res['J'] = J
json.dump(res, open(OUT, 'w'), indent=1, default=float)

# ----------------------------------------------------------------------------- K: valid versus capped risk control
print('\n== K: risk control at beta = 0.01, theoretically valid procedure (infinite margin = no label) and capped output')
Lm = {m: np.array([o[m]['L'] for o in O]) for m in ME}
Kres = {}
for m in ('sam', 'sam_pca', 'sam_range'):
    fin, rv, rp, ar, qq, ce, cap_r, cap_a, cap_c = [], [], [], [], [], [], [], [], []
    for cal, test in splits(imgs):
        Lj = np.array([Lm[m][byimg[j]].mean(0) for j in cal]); n = len(cal)
        ok = np.flatnonzero(n / (n + 1) * Lj.mean(0) + 1 / (n + 1) <= 0.01)      # Equation (37)
        T = np.concatenate([byimg[j] for j in test])
        k = ok[0] if len(ok) else len(Q) - 1
        r_ = np.mean([Lm[m][byimg[j], k].mean() for j in test]); a_ = np.median((1 + 2 * Q[k]) ** 2 * raw[m][T]); c_ = np.mean(s[m][T] <= Q[k])
        cap_r.append(r_); cap_a.append(a_); cap_c.append(c_)
        if len(ok): fin.append(1); rv.append(r_); ar.append(a_); qq.append(Q[k]); ce.append(c_)
        else: fin.append(0)
    Kres[m] = dict(valid=float(np.mean(fin)), valid_risk=float(np.mean(rv)), valid_risk_p95=float(np.percentile(rv, 95)),
                   valid_area=float(np.mean(ar)), valid_q=float(np.median(qq)), valid_cov=float(np.mean(ce)),
                   capped_risk=float(np.mean(cap_r)), capped_area=float(np.mean(cap_a)), capped_cov=float(np.mean(cap_c)))
    print(' ', m, Kres[m])
res['K'] = Kres

# ----------------------------------------------------------------------------- L: usable class-conditional labels
print('\n== L: class-conditional margins, share of test objects with a finite margin and coverage among them')
Lres = {}
for m in ME:
    for a in (0.10, 0.05):
        rng = np.random.default_rng(2026); usable, cov_f, cov_all = [], [], []
        for _ in range(200):
            p = rng.permutation(len(imgs)); h = len(imgs) // 2; cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
            q = np.zeros(len(O))
            for c in classes:
                sc = [s[m][rng.choice([i for i in byimg[j] if cls[i] == c])] for j in cal if any(cls[i] == c for i in byimg[j])]
                q[cls == c] = cq(np.array(sc), a)
            T = np.concatenate([byimg[j] for j in test]); f = np.isfinite(q[T])
            usable.append(f.mean())
            per = [np.mean((s[m][byimg[j]] <= q[byimg[j]])[np.isfinite(q[byimg[j]])]) for j in test if np.isfinite(q[byimg[j]]).any()]
            cov_f.append(np.mean(per))
        Lres[f'{m}|{a}'] = dict(usable=float(np.mean(usable)), cov_finite_img=float(np.mean(cov_f)))
        print(' ', m, a, Lres[f'{m}|{a}'])
res['L'] = Lres
json.dump(res, open(OUT, 'w'), indent=1, default=float)
json.dump(J, open(os.path.join(os.path.dirname(OUT), 'figure10_objects.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------- split definitions
# Development images and the 200 calibration/test partitions of steps 6 and 13 (Tables 6 and 8), which
# draw all 200 permutations first. Steps 7, 10, 14, 15 and 16 draw each permutation followed by the
# selections of that split from the same generator; their partitions are regenerated by the code.
rng = np.random.default_rng(2026); part = []
for _ in range(200):
    p = rng.permutation(len(imgs)); h = len(imgs) // 2
    part.append(dict(calibration=[imgs[i] for i in p[:h]], test=[imgs[i] for i in p[h:]]))
json.dump(dict(development=sorted(dev), evaluation=imgs, splits_step06=part), open('results/splits.json', 'w'))
print('\nwrote results/splits.json (development list, evaluation list, 200 partitions of step 6)')
