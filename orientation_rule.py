"""Section 4.11, Table 10: area-based rule against the threshold-angle rule of Zaidi (2022).

For every retained reference object K (103,524 objects), the relative excess of the horizontal
rectangle, A_K(0) / A*(K) - 1, and the angle between MBR(K) and the image axes (modulo 90 degrees)
are computed. The area rule of Section 3.10 assigns a horizontal label when 0 is in Theta_tau, that
is, when A_K(0) <= (1 + tau) A*(K), with tau = 0.05. The threshold-angle rule assigns a horizontal
label when the angle of MBR(K) is at most theta_0, for theta_0 = 5, 10, 15 and 20 degrees.
Printed for each theta_0: share of horizontal labels, share of horizontal labels with an excess
above 5%, their mean and maximum excess, and share of oriented labels although horizontal is
admissible.

Usage:  python orientation_rule.py results/match.json
"""
import sys, json, numpy as np
import geometry as G

R = json.load(open(sys.argv[1]))
ex, ang, asp, cl = [], [], [], []
for r in R:
    K = np.array(r['hull']); b = G.mbr(K)
    ex.append(G.area(G.box_at(K, 0.0)) / G.area(b) - 1)
    t = np.degrees(np.mod(b[0], np.pi / 2)); ang.append(min(t, 90 - t))
    w, h = sorted((b[2] - b[1], b[4] - b[3])); asp.append(h / w); cl.append(r['cls'])   # aspect ratio h / w of MBR(K)
ex = np.array(ex); ang = np.array(ang); asp = np.array(asp); cl = np.array(cl); adm = ex <= 0.05   # adm: 0 in Theta_tau, tau = 0.05
print('objects', len(ex), ' area rule: horizontal %.1f%%' % (100 * adm.mean()))
for t0 in (5, 10, 15, 20):
    h = ang <= t0; bad = h & ~adm
    print('theta0=%2d  horizontal %.1f%%  excess>5%% %.1f%%  mean excess %.1f%%  max %.1f%%  oriented although admissible %.1f%%' % (
        t0, 100 * h.mean(), 100 * bad.mean(), 100 * ex[bad].mean(), 100 * ex[bad].max(), 100 * (~h & adm).mean()))
h10 = ang <= 10; bad10 = h10 & ~adm; miss10 = ~h10 & adm
for c in ('large vehicle', 'harbor'):
    print('theta0=10: %s with a horizontal label exceeding the optimum by more than 5%%: %.0f%%' % (c, 100 * bad10[cl == c].mean()))
print('theta0=10: median aspect ratio of the objects labelled oriented although horizontal is admissible: %.2f' % np.median(asp[miss10]))
