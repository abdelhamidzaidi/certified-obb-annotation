"""Step 11 (Sections 4.3, 4.6 and 4.10): Figures 8 and 10.

Figure 8 (Section 4.6). (a) Image-weighted share of objects contained in their label, against the
median area ratio area(R^(q_hat)) / A*(K), for marginal margins at alpha from 0.5 to 0.05; (b) the
realised image-weighted uncovered fraction L(q_hat_beta), against the median area ratio, for risk
control at beta from 0.10 to 0.01 (curves computed in step 6, first 50 splits). Open squares: the
uncalibrated proposals (q = 0, Table 5); diamond: the expert DOTA labels.
Figure 10 (Section 4.10). Six held-out objects of moderate size whose fused proposal does not already
contain K (s(K, R) > 0), drawn with numpy.random.default_rng(5): reference hull K (white), expert
label (yellow, dotted), fused proposal R (orange, dashed) and calibrated label R^(q_hat) at
alpha = 0.10 with the median marginal margin q_hat (blue).

Usage:  python 11_figures.py OUT_DIR    (reads results/calib2.json, riskgrid.json, match.json and the
                                         DOTA images under dota/DOTAv1/images/val/)
"""
import json, sys, numpy as np, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
import geometry as E
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
BLUE, ORANGE, AQUA, INK, MUTED, GRID, YEL = '#2a78d6', '#eb6834', '#1baf7a', '#1f1f1e', '#6b6a64', '#dcdad3', '#eda100'
B = ''
C = json.load(open(B + 'results/calib2.json')); G = json.load(open(B + 'results/riskgrid.json')); OUT = sys.argv[1]
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']
O = G['objects']; allimgs = sorted({o['img'] for o in O})
dev = set(np.random.default_rng(7).choice(allimgs, len(allimgs) // 4, replace=False))
O = [o for o in O if o['img'] not in dev and all(m in o for m in ME)]
byimg = collections.defaultdict(list)
for i, o in enumerate(O): byimg[o['img']].append(i)
M = {r['id']: r for r in json.load(open(B + 'results/match.json'))}
# image-weighted mean: mean over images of the mean over their objects
def imgmean(v): return float(np.mean([np.mean([v[i] for i in ids]) for ids in byimg.values()]))
raw = {}
for m in ('det', 'sam', 'fuse'):
    s = [o[m]['s'] for o in O]; L0 = [o[m]['L'][0] for o in O]
    # uncalibrated proposal (q = 0): share contained (s = 0), uncovered fraction L(0), area ratio
    raw[m] = dict(cov=imgmean([x == 0 for x in s]), risk=imgmean(L0), area=float(np.median([o[m]['raw'] for o in O])))
du = {}; da = {}; dc = {}
for o in O:
    r = M[o['id']]
    if 'dota' in r:
        K = Polygon(r['hull']); q = Polygon(r['dota']['quad']).buffer(0)
        # expert label: uncovered fraction of K, area ratio to A*(K), full containment
        du[o['id']] = 1 - K.intersection(q).area / K.area; da[o['id']] = q.area / r['Astar']; dc[o['id']] = du[o['id']] <= 1e-9
ids_d = [i for i, o in enumerate(O) if o['id'] in du]
bi = collections.defaultdict(list)
for i in ids_d: bi[O[i]['img']].append(i)
dota = dict(cov=float(np.mean([np.mean([dc[O[i]['id']] for i in v]) for v in bi.values()])),
            risk=float(np.mean([np.mean([du[O[i]['id']] for i in v]) for v in bi.values()])),
            area=float(np.median([da[O[i]['id']] for i in ids_d])))
print('raw', raw, 'dota', dota)
# Figure 8
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
col = dict(det=ORANGE, sam=AQUA, fuse=BLUE); lab = dict(det='Detector rectangle', sam='SAM mask rectangle', fuse='Fused proposal')
for m in ('det', 'sam', 'fuse'):
    cv = C[f'{m}|curve']; x = [c['area'] for c in cv]; y = [100 * c['cov'] for c in cv]
    axs[0].plot(x, y, '-o', color=col[m], ms=3.5, lw=1.6, label=lab[m] + ' (certified)')
    axs[0].plot(raw[m]['area'], 100 * raw[m]['cov'], 's', color=col[m], mfc='white', mew=1.4, ms=6)
    rc = C[f'{m}|rcurve']; x = [c['area'] for c in rc]; y = [100 * c['risk'] for c in rc]
    axs[1].plot(x, y, '-o', color=col[m], ms=3.5, lw=1.6)
    axs[1].plot(raw[m]['area'], 100 * raw[m]['risk'], 's', color=col[m], mfc='white', mew=1.4, ms=6)
for ax, key in ((axs[0], 'cov'), (axs[1], 'risk')):
    ax.plot(dota['area'], 100 * dota[key], 'D', color=INK, ms=5.5, label='DOTA expert label')
    ax.grid(color=GRID, lw=0.6); ax.tick_params(colors=MUTED); ax.set_xlabel(r'Median area ratio to $A^*(K)$')
axs[0].set_ylabel('Objects fully covered (%)'); axs[0].set_title('(a) Coverage, conformal prediction', fontsize=9, loc='left')
axs[1].set_ylabel('Mean uncovered area (%)'); axs[1].set_title('(b) Uncovered area, risk control', fontsize=9, loc='left')
for a, (x, y) in zip((0.10, 0.05), [(c['area'], 100 * c['cov']) for c in C['fuse|curve'] if c['alpha'] in (0.10, 0.05)]):
    axs[0].annotate(rf'$\alpha={a}$', (x, y), textcoords='offset points', xytext=(4, -10), fontsize=7, color=MUTED)
for c in C['fuse|rcurve']:
    if c['beta'] in (0.02, 0.01):
        axs[1].annotate(rf'$\beta={c["beta"]}$', (c['area'], 100 * c['risk']), textcoords='offset points', xytext=(4, 4), fontsize=7, color=MUTED)
h, l = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], ls='none', marker='s', mfc='white', mec=MUTED, mew=1.4, ms=6)); l.append('Uncertified proposal')
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(OUT + '/fig_tradeoff.pdf', bbox_inches='tight'); fig.savefig(OUT + '/fig_tradeoff.png', dpi=200, bbox_inches='tight')

# ---------------- examples on real images
# median marginal margin q_hat at alpha = 0.10, and median q_hat_beta at beta = 0.02
qa = C['fuse|cov|marginal|0.1']['q_median']; qr = C['fuse|risk|0.02']['q_median']
print('q coverage', qa, 'q risk', qr)
# closed outline of R^(q) for R = b = (theta, a0, a1, b0, b1)
def corners(b, q=0.0):
    th = b[0]; u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    la, lb = b[2] - b[1], b[4] - b[3]
    a0, a1, b0, b1 = b[1] - q * la, b[2] + q * la, b[3] - q * lb, b[4] + q * lb
    return np.array([a0*u + b0*v, a1*u + b0*v, a1*u + b1*v, a0*u + b1*v, a0*u + b0*v])
want = ['plane', 'ship', 'large vehicle', 'storage tank', 'harbor', 'tennis court']
# Figure 10: one random object per category among those not already contained
rng = np.random.default_rng(5)
fig, axs = plt.subplots(2, 3, figsize=(6.8, 4.9))
for ax, c in zip(axs.ravel(), want):
    cand = [o for o in O if o['cls'] == c and o['id'] in du and 40 <= max(o['fuse']['la'], o['fuse']['lb']) <= 220 and o['fuse']['s'] > 0]
    o = cand[int(rng.integers(len(cand)))]; r = M[o['id']]
    K = np.array(r['hull']); p = np.array(r['det']['poly']); H = E.hull_pts(np.vstack([np.array(z) for z in r['sam']]))
    # fused proposal R = MBR(detected rectangle U P)
    bf = E.mbr(np.vstack([p, H]))
    rf = corners(bf); rc = corners(bf, qa); quad = np.vstack([r['dota']['quad'], r['dota']['quad'][:1]])
    allp = np.vstack([rc, K]); x0, y0 = allp.min(0) - 12; x1, y1 = allp.max(0) + 12
    im = cv2.imread(B + 'dota/DOTAv1/images/val/' + r['img'] + '.jpg')[:, :, ::-1]
    X0, Y0 = int(max(x0, 0)), int(max(y0, 0)); X1, Y1 = int(min(x1, im.shape[1])), int(min(y1, im.shape[0]))
    ax.imshow(im[Y0:Y1, X0:X1], extent=(X0, X1, Y1, Y0))
    Kc = np.vstack([K, K[:1]])
    import matplotlib.patheffects as pe
    ax.plot(Kc[:, 0], Kc[:, 1], color='white', lw=1.0, ls='-', path_effects=[pe.Stroke(linewidth=2.2, foreground='black'), pe.Normal()])
    ax.plot(quad[:, 0], quad[:, 1], color=YEL, lw=1.2, ls=':')
    ax.plot(rf[:, 0], rf[:, 1], color=ORANGE, lw=1.2, ls='--')
    ax.plot(rc[:, 0], rc[:, 1], color=BLUE, lw=1.5)
    unc_f = o['fuse']['L'][0]; unc_d = du[o['id']]
    ax.set_title(c.capitalize(), fontsize=9, pad=20)
    # uncovered fraction of the calibrated label, interpolated on the grid Q
    uc = 0.0 if o['fuse']['s'] <= qa else 100*np.interp(qa, G["qgrid"], o["fuse"]["L"])
    ax.text(0.5, 1.0, f'uncovered area: expert {100*unc_d:.1f}%\nproposal {100*unc_f:.1f}%, certified {uc:.1f}%',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.3, color=MUTED, linespacing=1.15)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('equal'); ax.axis('off')
import matplotlib.patheffects as pe
hs = [Line2D([], [], color='white', lw=1.0, path_effects=[pe.Stroke(linewidth=2.2, foreground='black'), pe.Normal()], label='iSAID outline (convex hull)'), Line2D([], [], color=YEL, lw=1.2, ls=':', label='DOTA expert label'),
      Line2D([], [], color=ORANGE, lw=1.2, ls='--', label='Fused proposal'), Line2D([], [], color=BLUE, lw=1.5, label=r'Certified label ($\alpha=0.1$)')]
fig.legend(handles=hs, loc='lower center', ncol=2, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.5)
fig.savefig(OUT + '/fig_examples_auto.pdf', bbox_inches='tight'); fig.savefig(OUT + '/fig_examples_auto.png', dpi=200, bbox_inches='tight')
json.dump(dict(raw=raw, dota=dota, qa=qa, qr=qr), open(OUT + '/fig_numbers.json', 'w'), indent=1)
