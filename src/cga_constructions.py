"""
cga_constructions.py -- line-by-line transcriptions of the CLUCalc homework scripts
into the NumPy CGA engine of cga.py, in their ORIGINAL form 
and in the REVISED form proposed in the paper (Appendix B).

Every function returns Euclidean coordinates so that the results can be
compared with the closed-form reference solutions in geometry.py.
"""
from __future__ import annotations

import numpy as np

from cga import (VecN3, SphereN3, einf, meet, meet3, euclid, normalise_point,
                 pointpair_extract_homework, classify_pointpair)

U = VecN3(0, 0, 0)


class ConstructionError(ValueError):
    """Base class for invalid geometric configurations."""


class Imaginary(ConstructionError):
    """The requested intersection has no real point pair."""


class Degenerate(ConstructionError):
    """The requested construction loses a defining geometric object."""


HELPER_AXIS_TOL_SQ = 1e-12


def _pair(P, *, literal: bool = False):
    """Extract a point pair.

    Original-program audits use the literal homework formula. Revised
    constructions use tolerant classification and clamp round-off at tangency.
    """
    status = classify_pointpair(P)
    if status == "imaginary":
        raise Imaginary("imaginary point pair")
    if status == "degenerate":
        raise Degenerate("degenerate point-pair blade")
    if literal:
        return pointpair_extract_homework(P)[2:]

    s = np.sqrt(max((P | P).scalar(), 0.0))
    divisor = einf | P
    p1 = normalise_point((s + P) / divisor)
    if status == "tangent":
        return p1, p1
    p2 = normalise_point((-s + P) / divisor)
    return p1, p2


def _helper_direction(a: float, b: float, c: float):
    """Preferred vertical helper, with a deterministic x-axis fallback."""
    d2 = a * a + b * b + c * c
    if d2 == 0.0:
        raise Degenerate("target coincides with the shoulder")
    radial2 = a * a + c * c
    on_vertical_axis = radial2 <= HELPER_AXIS_TOL_SQ * d2
    helper = VecN3(1, 0, 0) if on_vertical_axis else VecN3(0, 1, 0)
    branch_axis = 0 if on_vertical_axis else 1
    return helper, branch_axis


def _select_branch(n1, n2, branch_axis: int):
    """Select +y normally; use +x when the vertical-axis fallback is active."""
    p1, p2 = euclid(n1), euclid(n2)
    return n1 if p1[branch_axis] >= p2[branch_axis] else n2


# ------------------------------------------------------------------ task 1
def tripod_original():
    A, B, C = VecN3(-1, 0, 0), VecN3(1, 0, 0), VecN3(0, 0, 2)
    Boden = 4 * (A ^ B ^ C ^ einf)
    S = meet3(SphereN3(A, 2.69), SphereN3(B, 1.8), SphereN3(C, 1.5))
    n1, n2 = _pair(S, literal=True)
    spitze = n1 if n1.c[2] > 0 else n2
    KS = SphereN3(n2, 1.1)                       # original: centred at N_Pp2, not Spitze
    circ = meet(Boden, KS)
    F = meet(A ^ n1 ^ n2 ^ einf, circ)
    n3, n4 = _pair(F, literal=True)
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
    n3, n4 = _pair(F)
    d3, d4 = (np.sqrt(-2 * (A | n).scalar()) for n in (n3, n4))
    aussen = n3 if d3 > d4 else n4
    return {"spitze": euclid(spitze), "aussen": euclid(aussen)}


# ------------------------------------------------------------------ task 2
def two_link_original(a, b, c):
    A = VecN3(a, b, c)
    Z = VecN3(0, 1, 0)
    E = 8 * (U ^ Z ^ A ^ einf)
    P = meet(meet(SphereN3(U, 1), SphereN3(A, 1)), E)
    n1, n2 = _pair(P, literal=True)
    return euclid(n1 if n1.c[2] > n2.c[2] else n2)


def two_link_revised(a, b, c):
    A = VecN3(a, b, c)
    helper, branch_axis = _helper_direction(a, b, c)
    E = U ^ helper ^ A ^ einf
    P = meet(meet(SphereN3(U, 1), SphereN3(A, 1)), E)
    n1, n2 = _pair(P)
    return euclid(_select_branch(n1, n2, branch_axis))


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
    n1, n2 = _pair(GKA, literal=True)
    if swap:
        n1, n2 = n2, n1
    GE1, GE2 = (U - n1).dual(), (U - n2).dual()
    elbows = []
    for GE, K in ((GE1, K1), (GE2, K2)):
        p, q = _pair(meet(meet(GE, K), HE), literal=True)
        elbows.append(euclid(p if p.c[2] > q.c[2] else q))
    return elbows[0], elbows[1], (euclid(n1), euclid(n2))


def three_link_revised(a, b, c):
    """Robust three-link construction with explicit auxiliary-point selection.

    W+ is chosen as the auxiliary point farther from the shoulder, E2 is obtained
    from one circle-plane meet, and E1 is reflected in the shoulder-target
    bisector plane.  A deterministic +x branch is used when the target lies on
    the vertical axis, where "upper" alone cannot distinguish the two branches.
    """
    A = VecN3(a, b, c)
    K2 = SphereN3(A, 1)
    helper, branch_axis = _helper_direction(a, b, c)
    HE = U ^ helper ^ A ^ einf
    n1, n2 = _pair(meet(U ^ A ^ einf, K2))
    Wp = n1 if -(U | n1).scalar() > -(U | n2).scalar() else n2
    p, q = _pair(meet(meet((U - Wp).dual(), K2), HE))
    E2 = _select_branch(p, q, branch_axis)
    m = U - A                                           # IPNS bisector plane of U and A
    E1 = -(m * E2 * m.inverse())
    return euclid(E1), euclid(E2)
