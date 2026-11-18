"""Coordinates of Figure 1 (Section 3.1), printed as TikZ paths.

Panel (a): an instance region O (non-convex polygon), its hull K = conv(O), a proposal P and its
proposal rectangle R = MBR(P). Panel (b): the rectangle with orientation theta (here theta*(K) + 32
degrees) whose sides are the widths w_K(theta) and w_K(theta + pi/2) of Equation (1), of area
A_K(theta), and MBR(K), of area A*(K). Panel (c): R with sides w <= h, R_delta for delta = 0.15 and
R^(q) for q = 0.15. Panel (d): the contraction P^[r] = c + (1 - r)(P - c) for r = 0.4.
The minimum-area rectangles are found by a search over 180,001 orientations.
Output: fig2.json with the TikZ coordinate strings.
"""
import numpy as np
from shapely.geometry import Polygon, MultiPoint
O0=[(0,0.5),(0.8,0),(3.6,0.15),(4.7,0.75),(3.8,1.5),(2.4,1.05),(1.3,1.8),(0.2,1.3)]
P0=[(0.15,0.5),(0.85,0.12),(3.45,0.3),(3.85,0.85),(3.55,1.35),(1.35,1.55),(0.35,1.2)]
phi=np.radians(28); Rm=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
def T(p): return (np.array(p)-[2.35,0.9])@Rm.T
O=T(O0); P=T(P0)
K=np.array(MultiPoint([tuple(x) for x in O]).convex_hull.exterior.coords)[:-1]
f=lambda A:' -- '.join(f'({x:.3f},{y:.3f})' for x,y in A)+' -- cycle'
pt=lambda x:f'({x[0]:.3f},{x[1]:.3f})'
out={}
out['O']=f(O); out['K']=f(K); out['P']=f(P)
def rect(S,th):
    u=np.array([np.cos(th),np.sin(th)]); v=np.array([-np.sin(th),np.cos(th)])
    a=S@u; b=S@v
    C=[a.min()*u+b.min()*v, a.max()*u+b.min()*v, a.max()*u+b.max()*v, a.min()*u+b.max()*v]
    return np.array(C), a.max()-a.min(), b.max()-b.min()
def mbr(S):
    best=None
    for th in np.linspace(0,np.pi,180001):
        C,w1,w2=rect(S,th)
        if best is None or w1*w2<best[0]: best=(w1*w2,th,C,w1,w2)
    return best
# proposal rectangle R = MBR(P), panel (a)
Ap,thp,Cp,w1,w2=mbr(P)
print('P mbr theta',np.degrees(thp),w1,w2)
out['R']=f(Cp)
# frame of R with w <= h: unit vectors along the short (us) and long (ul) sides
cR=Cp.mean(0)
e1=Cp[1]-Cp[0]; e2=Cp[3]-Cp[0]
if np.linalg.norm(e1)>np.linalg.norm(e2): e1,e2=e2,e1
w=np.linalg.norm(e1); h=np.linalg.norm(e2); us=e1/w; ul=e2/h
print('w',w,'h',h)
def rr(cx,a,b): return f([cx-a*us-b*ul,cx+a*us-b*ul,cx+a*us+b*ul,cx-a*us+b*ul])
d=0.15; q=0.15          # delta for R_delta and q for R^(q), panel (c)
out['Rd']=rr(cR,w/2+d,h/2+d); out['Rq']=rr(cR,(1+2*q)*w/2,(1+2*q)*h/2)
out['cR']=pt(cR)
# key points for margins
out['us']=pt(us); out['ul']=pt(ul); out['w']=w; out['h']=h
# MBR(K) and the rectangle of area A_K(theta) at a generic theta, panel (b)
AK,thk,Ck,k1,k2=mbr(K)
print('K mbr theta',np.degrees(thk),k1,k2,AK)
out['MK']=f(Ck)
th=np.radians(np.degrees(thk)%180+32)
C,a1,a2=rect(K,th)
print('generic theta',np.degrees(th),a1,a2,a1*a2, 'ratio',a1*a2/AK)
out['RK']=f(C)
for i,c in enumerate(C): out[f'C{i}']=pt(c)
out['th']=np.degrees(th)
# contraction P^[r] about the centroid c, panel (d)
Pp=Polygon(P); c=np.array(Pp.centroid.coords[0]); r=0.4
Pr=c+(1-r)*(P-c)
out['Pr']=f(Pr); out['c']=pt(c)
i=np.argmax(np.linalg.norm(P-c,axis=1)); out['x']=pt(P[i]); out['xr']=pt(Pr[i])
import json; json.dump({k:(v if isinstance(v,str) else float(v)) for k,v in out.items()},open('fig2.json','w'),indent=0)
for k,v in out.items(): print(k,v)
