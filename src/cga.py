"""
cga.py -- a minimal, dependency-free (NumPy only) implementation of the
conformal geometric algebra G(4,1) used by CLUCalc's N3 mode.

The goal is not speed but transparency: every CLUCalc expression used in the
homework scripts (VecN3, SphereN3, ^, ., *, geometric product, division by a
vector) has a literal counterpart here, so that the scripts can be executed
line by line outside CLUCalc and their results checked against independent
Euclidean formulas.

Basis (bitmask order): e1, e2, e3, e+, e-  with  e_i^2 = +1 (i=1,2,3,+),
e-^2 = -1.  Null vectors:  einf = e- + e+,  e0 = (e- - e+)/2,
so that  einf^2 = e0^2 = 0  and  e0 . einf = -1.
"""
from __future__ import annotations

import itertools
import numpy as np

N = 5                      # dimension of the vector space R^{4,1}
DIM = 1 << N               # 32 blades
METRIC = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
GRADE = np.array([bin(b).count("1") for b in range(DIM)])


def _reorder_sign(a: int, b: int) -> int:
    """Sign obtained by reordering the basis product e_a e_b into canonical order."""
    a >>= 1
    s = 0
    while a:
        s += bin(a & b).count("1")
        a >>= 1
    return -1 if (s & 1) else 1


# Pre-computed multiplication table:  e_a e_b = SIGN[a,b] * e_{a xor b}
SIGN = np.zeros((DIM, DIM))
for _a, _b in itertools.product(range(DIM), repeat=2):
    s = _reorder_sign(_a, _b)
    common = _a & _b
    for i in range(N):
        if common & (1 << i):
            s *= METRIC[i]
    SIGN[_a, _b] = s
XOR = np.bitwise_xor.outer(np.arange(DIM), np.arange(DIM))


class MV:
    """A multivector of G(4,1) stored as 32 real coefficients."""

    __slots__ = ("c",)
    __array_priority__ = 1000

    def __init__(self, coeffs=None):
        if coeffs is None:
            self.c = np.zeros(DIM)
            return
        values = np.asarray(coeffs, float)
        if values.shape != (DIM,):
            raise ValueError(f"expected {DIM} multivector coefficients, got shape {values.shape}")
        self.c = values.copy()

    # ----------------------------------------------------------------- basics
    @staticmethod
    def blade(mask: int, value: float = 1.0) -> "MV":
        if not 0 <= mask < DIM:
            raise ValueError(f"blade mask must be in [0, {DIM - 1}]")
        m = MV()
        m.c[mask] = value
        return m

    def grade(self, k: int) -> "MV":
        return MV(np.where(GRADE == k, self.c, 0.0))

    def scalar(self) -> float:
        return float(self.c[0])

    def __add__(self, o):
        o = _as_mv(o)
        return MV(self.c + o.c)

    __radd__ = __add__

    def __sub__(self, o):
        return MV(self.c - _as_mv(o).c)

    def __rsub__(self, o):
        return MV(_as_mv(o).c - self.c)

    def __neg__(self):
        return MV(-self.c)

    def __mul__(self, o):
        """Geometric product (scalars are promoted)."""
        if np.isscalar(o):
            return MV(self.c * o)
        return _product(self, o, lambda ga, gb, gr: True)

    def __rmul__(self, o):
        return MV(self.c * o)

    def __truediv__(self, o):
        if np.isscalar(o):
            return MV(self.c / o)
        return self * o.inverse()

    def __xor__(self, o):
        """Outer product  A ^ B  (CLUCalc '^')."""
        return _product(self, o, lambda ga, gb, gr: gr == ga + gb)

    def __or__(self, o):
        """Inner product  A . B  (Hestenes inner product as in CLUCalc '.')."""
        return _product(self, o, lambda ga, gb, gr: ga > 0 and gb > 0 and gr == abs(ga - gb))

    def lc(self, o):
        """Left contraction A _| B."""
        return _product(self, o, lambda ga, gb, gr: gr == gb - ga)

    def reverse(self) -> "MV":
        k = GRADE
        return MV(self.c * np.where((k * (k - 1) // 2) % 2 == 0, 1.0, -1.0))

    def inverse(self) -> "MV":
        """Inverse for non-null blades/versors: A^{-1} = ~A / (A ~A)."""
        r = self.reverse()
        d = self * r
        if np.max(np.abs(d.c[1:])) > 1e-9 * max(1.0, abs(d.c[0])):
            raise ValueError("inverse() only implemented for blades/versors")
        if d.c[0] == 0.0 or not np.isfinite(d.c[0]):
            raise ZeroDivisionError("multivector is non-invertible")
        return r / d.c[0]

    def dual(self) -> "MV":
        """CLUCalc '*A' = A I^{-1}, with I = e1 e2 e3 e+ e-."""
        return self * I5_INV

    def norm2(self) -> float:
        return (self * self.reverse()).scalar()

    def __repr__(self):
        names = ["1"] + ["".join(BASIS_NAMES[i] for i in range(N) if b & (1 << i))
                         for b in range(1, DIM)]
        terms = [f"{v:+.6g}*{n}" for v, n in zip(self.c, names) if abs(v) > 1e-12]
        return " ".join(terms) if terms else "0"


BASIS_NAMES = ["e1", "e2", "e3", "ep", "em"]


def _as_mv(o) -> MV:
    return o if isinstance(o, MV) else MV.blade(0, float(o))


def _product(A: MV, B: MV, keep) -> MV:
    A, B = _as_mv(A), _as_mv(B)
    out = np.zeros(DIM)
    ia = np.nonzero(A.c)[0]
    ib = np.nonzero(B.c)[0]
    for a in ia:
        ga = GRADE[a]
        for b in ib:
            r = XOR[a, b]
            if keep(ga, GRADE[b], GRADE[r]):
                out[r] += SIGN[a, b] * A.c[a] * B.c[b]
    return MV(out)


# ---------------------------------------------------------------- constants
e1, e2, e3, ep, em = (MV.blade(1 << i) for i in range(N))
einf = em + ep
e0 = 0.5 * (em - ep)
I5 = e1 * e2 * e3 * ep * em
I5_INV = I5.inverse()


# ------------------------------------------------------ CLUCalc counterparts
def VecN3(x, y, z) -> MV:
    """Conformal embedding  X = x + 1/2 |x|^2 einf + e0  (CLUCalc VecN3)."""
    v = np.array([x, y, z], float)
    return v[0] * e1 + v[1] * e2 + v[2] * e3 + 0.5 * float(v @ v) * einf + e0


def SphereN3_ipns(center: MV, r: float) -> MV:
    """IPNS sphere vector  s = C - 1/2 r^2 einf  (C normalised)."""
    return normalise_point(center) - 0.5 * r * r * einf


def SphereN3(center: MV, r: float) -> MV:
    """OPNS sphere (as returned by CLUCalc SphereN3 in N3_OPNS mode)."""
    return SphereN3_ipns(center, r).dual()


def meet(A: MV, B: MV) -> MV:
    """CLUCalc idiom  *(*A ^ *B)  for OPNS blades A, B."""
    return (A.dual() ^ B.dual()).dual()


def meet3(A: MV, B: MV, C: MV) -> MV:
    return (A.dual() ^ B.dual() ^ C.dual()).dual()


def euclid(X: MV) -> np.ndarray:
    """Euclidean coordinates of a (not necessarily normalised) conformal point."""
    w = -(X | einf).scalar()
    if abs(w) < 1e-14:
        raise ZeroDivisionError("point at infinity / flat object")
    return np.array([X.c[1], X.c[2], X.c[4]]) / w


def normalise_point(X: MV) -> MV:
    """Normalise a finite conformal point to weight one."""
    weight = -(X | einf).scalar()
    if abs(weight) < 1e-14 or not np.isfinite(weight):
        raise ZeroDivisionError("cannot normalise point with zero/invalid conformal weight")
    return X / weight


def pointpair_extract_homework(P: MV):
    """Literal transcription of the homework formula

        Pp1 = ( sqrt(P.P) + P) / (einf . P)
        Pp2 = (-sqrt(P.P) + P) / (einf . P)
        N_Ppk = Ppk * -Ppk.einf

    Returns (Pp1, Pp2, N_Pp1, N_Pp2) as multivectors."""
    s2 = (P | P).scalar()
    if s2 < 0:
        raise ValueError("imaginary point pair (P.P < 0)")
    s = np.sqrt(s2)
    v = einf | P
    pp1 = (s + P) / v
    pp2 = (-s + P) / v
    n1 = pp1 * (-(pp1 | einf).scalar())
    n2 = pp2 * (-(pp2 | einf).scalar())
    return pp1, pp2, n1, n2


def classify_pointpair(P: MV, rel_tol: float = 1e-10) -> str:
    """Classify an OPNS point pair as real, tangent, imaginary, or degenerate."""
    s2 = (P | P).scalar()
    scale = float(np.sum(P.c ** 2))
    if scale == 0.0 or not np.isfinite(scale):
        return "degenerate"
    if abs(s2) <= rel_tol * scale:
        return "tangent"
    return "real" if s2 > 0 else "imaginary"


def pointpair_points(P: MV):
    """Extract Euclidean points from a non-degenerate real/tangent point pair."""
    status = classify_pointpair(P)
    if status in {"imaginary", "degenerate"}:
        return status, []

    s = np.sqrt(max((P | P).scalar(), 0.0))
    v = einf | P
    signs = (+1.0,) if status == "tangent" else (+1.0, -1.0)
    pts = []
    for sign in signs:
        X = (sign * s + P) / v
        pts.append(euclid(X))
    return status, pts
