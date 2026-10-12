#!/usr/bin/env python3
from fractions import Fraction as Q

def fq(x): return x if isinstance(x,Q) else Q(x)
def eye(n): return [[Q(int(i==j)) for j in range(n)] for i in range(n)]
def tr(a): return [list(x) for x in zip(*a)]
def mm(a,b):
    bt=tr(b)
    return [[sum((fq(x)*fq(y) for x,y in zip(row,col)),Q(0)) for col in bt] for row in a]
def add(a,b): return [[fq(x)+fq(y) for x,y in zip(rx,ry)] for rx,ry in zip(a,b)]
def sub(a,b): return [[fq(x)-fq(y) for x,y in zip(rx,ry)] for rx,ry in zip(a,b)]
def scale(c,a): return [[fq(c)*fq(x) for x in row] for row in a]
def inv(a):
    n=len(a); w=[list(map(fq,row))+eye(n)[i] for i,row in enumerate(a)]
    for c in range(n):
        p=next((r for r in range(c,n) if w[r][c]),None)
        if p is None: raise ValueError('singular')
        w[c],w[p]=w[p],w[c]; z=w[c][c]; w[c]=[x/z for x in w[c]]
        for r in range(n):
            if r!=c and w[r][c]:
                z=w[r][c];w[r]=[x-z*y for x,y in zip(w[r],w[c])]
    return [row[n:] for row in w]
def rank(a):
    w=[list(map(fq,row)) for row in a];m=len(w);n=len(w[0]) if m else 0;r=0
    for c in range(n):
        p=next((i for i in range(r,m) if w[i][c]),None)
        if p is None: continue
        w[r],w[p]=w[p],w[r];z=w[r][c];w[r]=[x/z for x in w[r]]
        for i in range(m):
            if i!=r and w[i][c]:
                z=w[i][c];w[i]=[x-z*y for x,y in zip(w[i],w[r])]
        r+=1
    return r
def cross(a,b): return mm(tr(a),b)
def curvature(carriers,mask,left,right):
    a=mask^left;b=mask^right
    cm,ca,cb=carriers[mask],carriers[a],carriers[b]
    gm,ga,gb=map(lambda x:inv(cross(x,x)),(cm,ca,cb))
    ab=mm(mm(mm(mm(mm(gm,cross(cm,ca)),ga),cross(ca,cb)),gb),cross(cb,cm))
    ba=mm(mm(mm(mm(mm(gm,cross(cm,cb)),gb),cross(cb,ca)),ga),cross(ca,cm))
    return sub(ab,ba)
def lorentz_completion(h):
    ht=[row[0] for row in h];d=ht[0]
    if not d: raise ValueError('zero clock norm')
    return sub(h,scale(Q(2,d),[[x*y for y in ht] for x in ht]))
def hchi(eta,f): return mm([[Q(1),Q(0),Q(0),Q(0)]],mm(eta,f))[0]
def modq(x,p): x=fq(x);return x.numerator*pow(x.denominator,-1,p)%p
def ray_mod(v,p):
    w=[modq(x,p) for x in v];z=next((x for x in w if x),None)
    return None if z is None else [x*pow(z,-1,p)%p for x in w]

