"""
homework_cga.py -- line-by-line transcriptions of the CLUCalc homework scripts
(Hausaufgabe1-3) into the NumPy CGA engine of cga.py, in their ORIGINAL form
and in the REVISED form proposed in the paper (Appendix B).

Every function returns Euclidean coordinates so that the results can be
compared with the closed-form reference solutions in geometry.py.
"""
from __future__ import annotations

import numpy as np

from cga import (VecN3, SphereN3, einf, meet, meet3, euclid, normalise_point,
                 pointpair_extract_homework, classify_pointpair)

U = VecN3(0, 0, 0)


class Imaginary(Exception):
    pass


def _pair(P):
    if classify_pointpair(P) == "imaginary":
        raise Imaginary
    return pointpair_extract_homework(P)[2:]          # (N_Pp1, N_Pp2)


# ------------------------------------------------------------------ task 1
def tripod_original():
    A, B, C = VecN3(-1, 0, 0), VecN3(1, 0, 0), VecN3(0, 0, 2)
    Boden = 4 * (A ^ B ^ C ^ einf)
    S = meet3(SphereN3(A, 2.69), SphereN3(B, 1.8), SphereN3(C, 1.5))
    n1, n2 = _pair(S)
    spitze = n1 if n1.c[2] > 0 else n2
    KS = SphereN3(n2, 1.1)                       # original: centred at N_Pp2, not Spitze
    circ = meet(Boden, KS)
    F = meet(A ^ n1 ^ n2 ^ einf, circ)
    n3, n4 = _pair(F)
    d1, d2 = (np.sqrt(-2 * (A | n).scalar()) for n in (n3, n4))
    aussen = n3 if d1 > d2 else n4
    return {"legs_drawn_to": euclid(n2), "spitze": euclid(spitze),
            "aussen": euclid(aussen), "fourth_leg_drawn_to": euclid(n4)}


def tripod_revised():
    A, B, C = VecN3(-1, 0, 0), VecN3(1, 0, 0), VecN3(0, 0, 2)
    Boden = A ^ B ^ C ^ einf
    S = meet3(SphereN3(A, 2.69), SphereN3(B, 1.8), SphereN3(C, 1.5))
    n1, n2 = _pair(S)
    spitze, other = (n1, n2) if euclid(n1)[1] > euclid(n2)[1] else (n2, n1)
    circ = meet(Boden, SphereN3(spitze, 1.1))
    F = meet(A ^ spitze ^ other ^ einf, circ)
    n3, n4 = (normalise_point(n) for n in _pair(F))
    d3, d4 = (np.sqrt(-2 * (A | n).scalar()) for n in (n3, n4))
    aussen = n3 if d3 > d4 else n4
    return {"spitze": euclid(spitze), "aussen": euclid(aussen)}


# ------------------------------------------------------------------ task 2
def two_link_original(a, b, c):
    A = VecN3(a, b, c)
    Z = VecN3(0, 1, 0)
    E = 8 * (U ^ Z ^ A ^ einf)
    P = meet(meet(SphereN3(U, 1), SphereN3(A, 1)), E)
    n1, n2 = _pair(P)
    return euclid(n1 if n1.c[2] > n2.c[2] else n2)


def two_link_revised(a, b, c):
    A = VecN3(a, b, c)
    helper = VecN3(0, 1, 0) if a * a + c * c > 1e-12 else VecN3(1, 0, 0)
    E = U ^ helper ^ A ^ einf
    P = meet(meet(SphereN3(U, 1), SphereN3(A, 1)), E)
    n1, n2 = (normalise_point(n) for n in _pair(P))
    return euclid(n1 if euclid(n1)[1] > euclid(n2)[1] else n2)


# ------------------------------------------------------------------ task 3
def three_link_original(a, b, c, swap=False):
    """Literal transcription.  Note: which of N_Pp1/N_Pp2 is the point nearer to the
    shoulder depends on the ordering produced by the extraction formula (i.e. on the
    sign conventions of dual and inner product).  swap=True emulates the opposite
    ordering."""
    A = VecN3(a, b, c)
    K1, K2 = SphereN3(U, 1), SphereN3(A, 1)
    HE = 10 * (U ^ VecN3(0, 1, 0) ^ A ^ einf)
    GKA = meet(4 * (U ^ A ^ einf), K2)
    n1, n2 = _pair(GKA)
    if swap:
        n1, n2 = n2, n1
    GE1, GE2 = (U - n1).dual(), (U - n2).dual()
    elbows = []
    for GE, K in ((GE1, K1), (GE2, K2)):
        p, q = _pair(meet(meet(GE, K), HE))
        elbows.append(euclid(p if p.c[2] > q.c[2] else q))
    return elbows[0], elbows[1], (euclid(n1), euclid(n2))


def three_link_revised(a, b, c):
    """Revised construction: W+ chosen explicitly (farther from the shoulder, never
    coincides with it), E2 from one circle-plane meet, E1 by reflecting E2 in the
    bisector plane of shoulder and target  (E1 = -m E2 m^{-1},  m = U - A)."""
    A = VecN3(a, b, c)
    K2 = SphereN3(A, 1)
    helper = VecN3(0, 1, 0) if a * a + c * c > 1e-12 else VecN3(1, 0, 0)
    HE = U ^ helper ^ A ^ einf
    n1, n2 = (normalise_point(n) for n in _pair(meet(U ^ A ^ einf, K2)))
    Wp = n1 if -(U | n1).scalar() > -(U | n2).scalar() else n2
    p, q = (normalise_point(n) for n in _pair(meet(meet((U - Wp).dual(), K2), HE)))
    E2 = p if euclid(p)[1] > euclid(q)[1] else q
    m = U - A                                           # IPNS bisector plane of U and A
    E1 = -(m * E2 * m.inverse())
    return euclid(E1), euclid(E2)
