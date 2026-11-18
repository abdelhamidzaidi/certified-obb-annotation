"""Step 15: Figures 5, 6, 7 and 9 (score distributions, coverage against the Beta law, image-level against
pooled calibration, recall and class-conditional margins).

Figure 5 (Section 4.4). Survival function P(s(K, R) > q) of the outer score over the evaluation
objects, for the five proposals. By Lemma 4, an object is contained in R^(q) iff s(K, R) <= q, so the
curve gives the share of objects that a margin q leaves uncovered; it crosses alpha near the
calibrated margin q_hat, and its value at q = 0 is the share not already contained.
Figure 6 (Section 4.4). Per-category image-weighted coverage of the fused proposal at alpha = 0.10
with class-conditional margins q_hat_y calibrated on one object per image (Corollary 1) and on all
calibration objects pooled.
Figure 7 (Section 4.5, Proposition 3). Histogram over the 200 splits of the image-weighted coverage
C(q_hat) of the marginal margin q_hat = s_(k) and of the training-conditional margin
q_hat^delta = s_(k_delta) (delta = 0.10), for the fused proposal, with the density of
Beta(k, n + 1 - k) and the target 1 - alpha.
Figure 9 (Sections 4.2 and 4.9). (a) Matched-proposal recall against the shorter side of MBR(K),
outside the development subset; (b) class-conditional margins q_hat_y of the fused proposal at
alpha = 0.10 (median over the splits with a finite margin).

The splits, selections and margins are those of steps 6, 7, 10 and 14 (same seeds), so the summary
statistics printed here equal those of Tables 6, 7 and 9 and Section 4.4.

Usage:  python 15_extra_figures.py OUT_DIR    (reads results/riskgrid.json and results/scores.json)
"""
import sys, json, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
# fixed colour per proposal, as in Figure 8 (reference categorical order: blue, orange, aqua, yellow, magenta)
COL = dict(fuse='#2a78d6', det='#eb6834', sam='#1baf7a', sam_pca='#eda100', sam_range='#e87ba4')
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#dcdad3'
NAME = dict(det='Detector rectangle', sam='SAM mask rectangle', fuse='Fused proposal',
            sam_pca='Principal axes of the mask', sam_range='RANGE-MBR on mask points')
CNAME = {'plane': 'Plane', 'ship': 'Ship', 'storage tank': 'Storage tank', 'baseball diamond': 'Baseball diamond',
         'tennis court': 'Tennis court', 'basketball court': 'Basketball court', 'ground track field': 'Ground track field',
         'harbor': 'Harbor', 'bridge': 'Bridge', 'large vehicle': 'Large vehicle', 'small vehicle': 'Small vehicle',
         'helicopter': 'Helicopter', 'roundabout': 'Roundabout', 'soccer ball field': 'Soccer field', 'swimming pool': 'Swimming pool'}
OUT = sys.argv[1]
ME = ['det', 'sam', 'fuse', 'sam_pca', 'sam_range']

G = json.load(open('results/riskgrid.json'))
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


def kdelta(n, a, d):
    """k_delta: smallest k <= n with P(Beta(k, n + 1 - k) < 1 - alpha) <= delta (Equation (34))."""
    ks = [k for k in range(1, n + 1) if stats.beta.sf(1 - a, k, n + 1 - k) >= 1 - d]
    return ks[0] if ks else None


def style(ax):
    ax.grid(color=GRID, lw=0.6); ax.tick_params(colors=MUTED)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(MUTED)


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', dpi=200, bbox_inches='tight')


# ---------------------------------------------------------------- Figure 5: survival function of s(K, R)
fig, ax = plt.subplots(figsize=(6.8, 3.0))
qq = np.linspace(0, 0.45, 451)
for m in ['fuse', 'det', 'sam', 'sam_pca', 'sam_range']:
    s = np.array([o[m]['s'] for o in O])
    surv = [(s > q).mean() for q in qq]                         # share of objects not contained in R^(q)
    ax.plot(qq, 100 * np.array(surv), color=COL[m], lw=1.8, label=NAME[m])
for a in (0.10, 0.05):
    ax.axhline(100 * a, color=MUTED, lw=0.8, ls='--')
    ax.text(0.452, 100 * a, rf'$\alpha={a:.2f}$', color=MUTED, fontsize=7.5, va='center', ha='left')
ax.set_yscale('log'); ax.set_ylim(2, 100); ax.set_xlim(0, 0.45)
ax.set_yticks([2, 5, 10, 20, 50, 100]); ax.set_yticklabels(['2', '5', '10', '20', '50', '100'])
ax.set_xlabel(r'Margin $q$'); ax.set_ylabel(r'Objects with $s(K,R)>q$ (%)')
style(ax); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout(); save(fig, 'fig_scores')

# ---------------------------------------------------------------- Figure 7: coverage across splits and the Beta law
# The splits below, and those of Figures 6 and 9, replicate the sequence of default_rng(2026) used in
# steps 6, 7 and 10: each split permutes the 337 images, takes the first 168 as calibration images
# (Table A1), then draws one eligible object per calibration image with the same generator.
# replicates step 7 (run_splits with seed 2026): marginal margin and training-conditional margin, fused proposal
s = np.array([o['fuse']['s'] for o in O]); a = 0.10


def coverage_over_splits(pac):
    rng = np.random.default_rng(2026); cov = []
    for _ in range(200):
        p = rng.permutation(len(imgs)); h = len(imgs) // 2
        cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
        sc = np.sort(np.array([s[rng.choice(byimg[j])] for j in cal]))   # one object per calibration image
        n = len(sc)
        if pac is None:
            q = cq(sc, a)
        else:
            kd = kdelta(n, a, pac); q = float(sc[kd - 1]) if kd else np.inf
        cov.append(np.mean([np.mean(s[byimg[j]] <= q) for j in test]))  # image-weighted coverage C(q_hat)
    return np.array(cov), n


cov_m, n = coverage_over_splits(None); cov_d, _ = coverage_over_splits(0.10)
k = int(np.ceil((n + 1) * (1 - a))); kd = kdelta(n, a, 0.10)
print('Figure 7: n %d, k %d, k_delta %d; marginal coverage mean %.3f (share below %.3f); training-conditional mean %.3f (share below %.3f)'
      % (n, k, kd, cov_m.mean(), np.mean(cov_m < 1 - a), cov_d.mean(), np.mean(cov_d < 1 - a)))
fig, ax = plt.subplots(figsize=(6.8, 2.9))
bins = np.linspace(0.80, 1.0, 41)
ax.hist(cov_m, bins=bins, density=True, color=COL['fuse'], alpha=0.55, edgecolor='white', lw=0.6, label=rf'Marginal margin $\hat q=s_{{({k})}}$')
ax.hist(cov_d, bins=bins, density=True, color=COL['det'], alpha=0.45, edgecolor='white', lw=0.6, label=rf'Training-conditional margin $\hat q^{{\,\delta}}=s_{{({kd})}}$, $\delta=0.10$')
xs = np.linspace(0.80, 1.0, 400)
ax.plot(xs, stats.beta.pdf(xs, k, n + 1 - k), color=INK, lw=1.4, label=rf'Beta$({k},{n + 1 - k})$')
ax.plot(xs, stats.beta.pdf(xs, kd, n + 1 - kd), color=INK, lw=1.2, ls=':', label=rf'Beta$({kd},{n + 1 - kd})$')
ax.axvline(1 - a, color=MUTED, lw=0.9, ls='--'); ax.text(1 - a - 0.002, ax.get_ylim()[1] * 0.95, r'$1-\alpha$', color=MUTED, fontsize=7.5, ha='right', va='top')
ax.set_xlim(0.80, 1.0); ax.set_xlabel('Image-weighted coverage of a calibration set'); ax.set_ylabel('Density')
style(ax); ax.legend(frameon=False, fontsize=7.2, loc='upper left')
fig.tight_layout(); save(fig, 'fig_beta')

# ---------------------------------------------------------------- Figure 6: image-level against pooled calibration
# image-level class-conditional margins replicate step 6 (Corollary 1); pooled margins replicate step 14
def per_class_cov(pooled):
    rng = np.random.default_rng(2026); pc = collections.defaultdict(list)
    for _ in range(200):
        p = rng.permutation(len(imgs)); h = len(imgs) // 2
        cal = [imgs[i] for i in p[:h]]; test = [imgs[i] for i in p[h:]]
        q = np.zeros(len(s))
        if pooled:
            C = np.concatenate([byimg[j] for j in cal])
            for c in classes: q[cls == c] = cq(s[C][cls[C] == c], 0.10)          # all calibration objects pooled
        else:
            for c in classes:
                sc = [s[rng.choice([i for i in byimg[j] if cls[i] == c])] for j in cal if any(cls[i] == c for i in byimg[j])]
                q[cls == c] = cq(np.array(sc), 0.10)                               # one object per calibration image
        cov = s <= q
        for c in classes:
            per = [cov[[i for i in byimg[j] if cls[i] == c]].mean() for j in test if any(cls[i] == c for i in byimg[j])]
            if per: pc[c].append(np.mean(per))
    return {c: 100 * np.mean(v) for c, v in pc.items()}


img_lvl = per_class_cov(False); pooled = per_class_cov(True)
order = sorted([c for c in classes if c != 'helicopter'], key=lambda c: pooled[c])
print('Figure 6: smallest image-level %.1f (%s); smallest pooled %.1f (%s)'
      % (min(img_lvl[c] for c in order), min(order, key=lambda c: img_lvl[c]), pooled[order[0]], order[0]))
fig, ax = plt.subplots(figsize=(6.8, 3.6))
y = np.arange(len(order))
for i, c in enumerate(order):
    ax.plot([pooled[c], img_lvl[c]], [i, i], color=GRID, lw=2, zorder=1)
ax.scatter([pooled[c] for c in order], y, s=34, color=COL['det'], zorder=3, edgecolor='white', lw=0.8, label='All calibration objects pooled')
ax.scatter([img_lvl[c] for c in order], y, s=34, color=COL['fuse'], zorder=3, edgecolor='white', lw=0.8, label='One object per calibration image')
ax.axvline(90, color=MUTED, lw=0.9, ls='--'); ax.text(90.15, len(order) - 0.6, r'$1-\alpha$', color=MUTED, fontsize=7.5, va='bottom')
ax.set_yticks(y); ax.set_yticklabels([CNAME[c] for c in order]); ax.set_ylim(-0.7, len(order) - 0.2)
ax.set_xlabel('Image-weighted coverage, class-conditional margins, fused proposal (%)')
style(ax); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout(); save(fig, 'fig_pooled')

# ---------------------------------------------------------------- Figure 9: recall against size; class-conditional margins
S = json.load(open('results/scores.json'))
Snd = [x for x in S if x['img'] not in dev]                                     # 346 images outside the development subset
sK = np.array([x['sK'] for x in Snd]); det = np.array(['det' in x for x in Snd])
edges = np.array([3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 300, 2000])
mid, rec, cnt = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (sK >= lo) & (sK < hi)
    if sel.sum() >= 20: mid.append(np.sqrt(lo * min(hi, 600))); rec.append(100 * det[sel].mean()); cnt.append(int(sel.sum()))
# class-conditional margins of the fused proposal (as in step 10)
qc = collections.defaultdict(list); rng = np.random.default_rng(2026)
for _ in range(200):
    p = rng.permutation(len(imgs)); h = len(imgs) // 2; cal = [imgs[i] for i in p[:h]]
    for c in classes:
        sc = [s[rng.choice([i for i in byimg[j] if cls[i] == c])] for j in cal if any(cls[i] == c for i in byimg[j])]
        qc[c].append(cq(np.array(sc), 0.10))
qmed = {c: float(np.median([x for x in v if np.isfinite(x)])) for c, v in qc.items() if any(np.isfinite(v)) and np.mean(np.isfinite(v)) > 0.5}
print('Figure 9: recall by size', list(zip([round(m_) for m_ in mid], [round(r_, 1) for r_ in rec], cnt)))
print('Figure 9: class-conditional margins', {CNAME[c]: round(v, 3) for c, v in sorted(qmed.items(), key=lambda t: t[1])})
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
ax.plot(mid, rec, '-o', color=COL['fuse'], lw=1.8, ms=4.5, mec='white', mew=0.8)
ax.set_xscale('log'); ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(['3', '10', '30', '100', '300'])
ax.set_ylim(0, 100); ax.set_xlabel(r'Shorter side of $\mathrm{MBR}(K)$ (pixels)'); ax.set_ylabel('Matched-proposal recall (%)')
ax.set_title('(a) Recall against object size', fontsize=9, loc='left'); style(ax)
ax = axs[1]
cs = sorted(qmed, key=lambda c: qmed[c]); yy = np.arange(len(cs))
ax.barh(yy, [qmed[c] for c in cs], height=0.62, color=COL['fuse'])
for i, c in enumerate(cs): ax.text(qmed[c] + 0.006, i, f'{qmed[c]:.3f}', va='center', fontsize=6.8, color=MUTED)
ax.set_yticks(yy); ax.set_yticklabels([CNAME[c] for c in cs], fontsize=7.5); ax.set_xlim(0, 0.45)
ax.set_xlabel(r'Class-conditional margin $\hat q_y$, $\alpha=0.10$'); ax.set_title(r'(b) Margins of the fused proposal', fontsize=9, loc='left')
style(ax); ax.grid(axis='y', visible=False)
fig.tight_layout(w_pad=1.2); save(fig, 'fig_recall_margins')
print('done')
