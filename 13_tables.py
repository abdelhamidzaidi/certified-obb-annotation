"""Step 13: Tables 2, 5, 6, 7 and 8 and the figures quoted in Sections 4.2, 4.4 and 4.6.

All quantities are read from the outputs of steps 2-7; nothing is recomputed here except the
summaries restricted to valid splits in Section 4.6. Notation as in the manuscript: K is the
reference hull, R a proposal rectangle, s = s(K, R) its outer score (Equation (29)), L(q) the
uncovered fraction of K outside R^(q) (Section 3.7), A*(K) the minimum enclosing area.

Usage:  python 13_tables.py    (reads results/match.json, scores.json, riskgrid.json, calib2.json,
                                extra.json)
"""
import json, collections
import numpy as np
from shapely.geometry import Polygon

R = json.load(open('results/match.json'))
S = json.load(open('results/scores.json'))
G = json.load(open('results/riskgrid.json')); Q = np.array(G['qgrid'])
C = json.load(open('results/calib2.json'))
X = json.load(open('results/extra.json'))
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
NAME = dict(det='Detector rectangle', sam='SAM mask rectangle', fuse='Fused proposal',
            sam_pca='Principal axes of the mask', sam_range='RANGE-MBR on mask points')

# development subset and evaluation set (Section 3.13)
O5 = [o for o in G['objects'] if all(m in o for m in ME)]
allimgs = sorted({o['img'] for o in O5})
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))
E = [o for o in O5 if o['img'] not in dev]
eval_ids = {o['id'] for o in E}
Rnd = [r for r in R if r['img'] not in dev]                  # retained objects outside the development subset
Snd = [s for s in S if s['img'] not in dev]

print('== Table 2: images and objects at each stage')
D = X['D']
print('retained iSAID objects: %d images, %d objects' % (len({r['img'] for r in R}), len(R)))
print('detections: %d, matched %d (precision %.1f%%)' % (D['detections'], D['matched'], 100 * D['precision']))
print('development subset: %d images, %d objects with all five proposals' % (len(dev), sum(o['img'] in dev for o in O5)))
print('outside the development subset: %d images, %d objects, %d matched'
      % (len({r['img'] for r in Rnd}), len(Rnd), sum('det' in r for r in Rnd)))
print('evaluation set: %d images, %d objects; with an expert label %d; matched with IoU >= 0.7 %d'
      % (len({o['img'] for o in E}), len(E), sum('dota' in r for r in R if r['id'] in eval_ids), X['C']['n']))

print('\n== Section 4.2: matched-proposal recall')
rec = np.mean(['det' in s for s in Snd])
print('recall %.1f%% of %d objects' % (100 * rec, len(Snd)))
for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 1e9)):
    L = [s for s in Snd if lo <= s['sK'] < hi]
    print('  shorter side of MBR(K) in [%g, %g): recall %.1f%%' % (lo, hi, 100 * np.mean(['det' in s for s in L])))
sv = [s for s in Snd if s['cls'] == 'small vehicle']
print('  small vehicles: %.0f%% of the objects, recall %.1f%%' % (100 * len(sv) / len(Snd), 100 * np.mean(['det' in s for s in sv])))
print('  objects with an expert DOTA label: %.1f%%' % (100 * np.mean(['dota' in s for s in Snd])))

print('\n== Table 5: expert labels and uncalibrated proposals (evaluation objects)')
print('label | fully covered % | uncovered mean % | uncovered median % | area ratio median [p5, p95]')
M = {r['id']: r for r in R}
du, da = [], []
for o in E:
    r = M[o['id']]
    if 'dota' in r:
        K = Polygon(r['hull']); q = Polygon(r['dota']['quad']).buffer(0)
        du.append(max(0.0, 1 - K.intersection(q).area / K.area)); da.append(q.area / r['Astar'])
du, da = np.array(du), np.array(da)
print('DOTA expert label | %.1f | %.2f | %.2f | %.3f [%.3f, %.3f]' % (100 * np.mean(du <= 1e-9), 100 * du.mean(), 100 * np.median(du),
      np.median(da), np.percentile(da, 5), np.percentile(da, 95)))
for m in ME:
    r = C[f'{m}|raw']
    print('%s | %.1f | %.2f | %.2f | %.3f [%.3f, %.3f]' % (NAME[m], 100 * r['exact'], 100 * r['uncovered_mean'], 100 * r['uncovered_median'],
          r['area_median'], r['area_p5'], r['area_p95']))

print('\n== Table 6: coverage of the calibrated labels (200 splits)')
print('proposal | alpha | margin | image-weighted mean [p5, p95] | object-weighted | worst category | area ratio | q_hat')
for m in ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']:
    for a in (0.1, 0.05):
        for mode, lab in (('marginal', 'marginal'), ('mondrian', 'per class')):
            r = C[f'{m}|cov|{mode}|{a}']
            print('%s | %.2f | %s | %.1f [%.1f, %.1f] | %.1f | %.1f | %.2f | %.3f' % (NAME[m], a, lab, 100 * r['cov_img'], 100 * r['cov_img_p5'],
                  100 * r['cov_img_p95'], 100 * r['cov_obj'], 100 * r['per_class_min'], r['area_median'], r['q_median']))
for a in (0.1, 0.05):
    print('share of test objects with an infinite class-conditional margin, fused, alpha=%.2f: %.1f%%' % (a, 100 * C[f'fuse|cov|mondrian|{a}']['share_infinite']))
print('90th percentile of s: fused %.3f, detector %.3f' % (C['fuse|raw']['s_p90'], C['det|raw']['s_p90']))

print('\n== Table 7: training-conditional behaviour (marginal margins, image-weighted)')
A = X['A']
for m in ('fuse', 'det'):
    for a in (0.1, 0.05):
        e = A[f'{m}|{a}']; p = A[f'{m}|{a}|pac0.1']; p5 = A[f'{m}|{a}|pac0.05']
        print('%s | alpha %.2f | n %d k %d | coverage %.1f [%.1f, %.1f] | Beta [%.1f, %.1f] | below %.1f%% | area %.2f || delta 0.10: %.1f [%.1f, %.1f], below %.1f%%, area %.2f || delta 0.05: below %.1f%%'
              % (NAME[m], a, e['n'], e['k'], 100 * e['emp']['cov_all']['mean'], 100 * e['emp']['cov_all']['p5'], 100 * e['emp']['cov_all']['p95'],
                 100 * e['beta_p5'], 100 * e['beta_p95'], 100 * e['share_below'], e['emp']['area_all']['mean'],
                 100 * p['cov_all']['mean'], 100 * p['cov_all']['p5'], 100 * p['cov_all']['p95'], 100 * p['share_below'], p['area_all']['mean'],
                 100 * p5['share_below']))

print('\n== Table 8: risk-controlled labels (200 splits); daggered rows use the capped margin 0.4')
print('proposal | beta | uncovered mean % | P95 % | area ratio | q_hat_beta | valid % | fully covered %')
for m in ME:
    for b in (0.05, 0.02, 0.01):
        r = C[f'{m}|risk|{b}']
        dag = '' if r['feasible'] == 1 else ' (dagger)'
        print('%s | %.2f%s | %.2f | %.2f | %.2f | %.3f | %.0f | %.1f' % (NAME[m], b, dag, 100 * r['risk_img'], 100 * r['risk_img_p95'],
              r['area_median'], r['q_median'], 100 * r['feasible'], 100 * r['cov_exact']))

# Section 4.6: summaries restricted to the splits with a finite margin
imgs = sorted({o['img'] for o in E}); byimg = collections.defaultdict(list)
for i, o in enumerate(E): byimg[o['img']].append(i)
rng = np.random.default_rng(2026); splits = []
for _ in range(200):
    p = rng.permutation(len(imgs)); h = len(imgs) // 2
    splits.append(([imgs[i] for i in p[:h]], [imgs[i] for i in p[h:]]))
print('\n== Section 4.6: beta = 0.01, splits with a finite margin only')
for m in ME:
    L = np.array([o[m]['L'] for o in E]); raw = np.array([o[m]['raw'] for o in E])
    rv, ar, qq, ninf = [], [], [], 0
    for cal, test in splits:
        Lj = np.array([L[byimg[j]].mean(0) for j in cal]); n = len(cal)
        ok = np.flatnonzero(n / (n + 1) * Lj.mean(0) + 1 / (n + 1) <= 0.01)     # Equation (37)
        if not len(ok): ninf += 1; continue
        k = ok[0]; T = np.concatenate([byimg[j] for j in test])
        rv.append(np.mean([L[byimg[j], k].mean() for j in test])); ar.append(np.median((1 + 2 * Q[k]) ** 2 * raw[T])); qq.append(Q[k])
    print('%s: infinite margin in %d of 200 splits; finite splits: uncovered %.2f%%, area ratio %.2f, median margin %.2f'
          % (NAME[m], ninf, 100 * np.mean(rv), np.mean(ar), np.median(qq)))

print('\n== Section 4.6, Figure 8: trade-off curves of the fused proposal (first 50 splits)')
for c in C['fuse|curve']:
    print('  marginal margin, alpha %.2f: coverage %.1f%%, median area ratio %.3f' % (c['alpha'], 100 * c['cov'], c['area']))
for c in C['fuse|rcurve']:
    print('  risk control, beta %.2f: uncovered %.2f%%, median area ratio %.3f' % (c['beta'], 100 * c['risk'], c['area']))

print('\n== Section 4.7: selective certification (agreement g, thresholds fixed on the development subset)')
Bx = X['B']
for tg in ('0.8', '0.9'):
    b = Bx['thr' + tg]
    print('  accept %s of dev proposals: t = %.3f; dev 90th percentile of s: accepted %.3f, deferred %.3f'
          % (tg, b['threshold'], b['dev_s_p90_acc'], b['dev_s_p90_def']))
    sel = Bx[f'thr{tg}|fuse|select']; two = Bx[f'thr{tg}|fuse|two']
    print('  fused: accepted %.1f%%, coverage %.1f%%, median area %.2f; two strata: coverage %.1f%% / %.1f%%, combined area %.2f'
          % (100 * sel['accept_rate']['mean'], 100 * sel['cov_acc']['mean'], sel['area_acc']['mean'],
             100 * two['cov_acc']['mean'], 100 * two['cov_def']['mean'], two['area_all']['mean']))
print('  without selection: fused coverage %.1f%%, median area %.2f' % (100 * Bx['base']['fuse']['cov_all']['mean'], Bx['base']['fuse']['area_all']['mean']))

print('\n== Section 4.8: matching threshold IoU >= 0.7 (alpha = 0.10, marginal)')
Cx = X['C']
print('  objects %d (%.1f%% of the evaluation set)' % (Cx['n'], 100 * Cx['share']))
for m in ('fuse', 'det'):
    print('  %s: uncalibrated uncovered %.2f%%; margin %.3f (IoU >= 0.5: %.3f); median area %.2f (IoU >= 0.5: %.2f); coverage %.1f%%'
          % (NAME[m], 100 * Cx[m + '_raw']['unc'], Cx[m]['q_all']['mean'], Bx['base'][m]['q_all']['mean'],
             Cx[m]['area_all']['mean'], Bx['base'][m]['area_all']['mean'], 100 * Cx[m]['cov_all']['mean']))
