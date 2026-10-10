"""
geometry.py -- reference implementations of the six constructions studied in
the paper, written against plain NumPy.

Every routine follows the same contract (Section 6 of the paper):

    encode -> construct -> intersect -> classify -> (all candidates) -> select -> validate

and returns a ConstructionResult so that status, *all* candidates, the selected
candidate and the residuals travel together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

EPS = np.finfo(float).eps


# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------
@dataclass
class ConstructionResult:
    status: str                                   # 'regular', 'tangent', 'ideal', 'empty', 'degenerate'
    candidates: list = field(default_factory=list)
    selected: Optional[np.ndarray] = None
    residual: float = float("nan")
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Projective plane
# --------------------------------------------------------------------------
def hpoint(x, y, w=1.0) -> np.ndarray:
    return np.array([x, y, w], float)


def join(p, q) -> np.ndarray:
    """Line through two homogeneous points (or point and ideal point)."""
    return np.cross(p, q)


def meet(l, m) -> np.ndarray:
    """Intersection point of two homogeneous lines."""
    return np.cross(l, m)


def is_ideal(p, tau: float = 1e3 * EPS) -> bool:
    """Scale-aware test  |w| <= tau * ||p||  (eq. 'homogeneous threshold')."""
    return abs(p[2]) <= tau * np.linalg.norm(p)


def dehom(p) -> np.ndarray:
    return p[:2] / p[2]


def rho_inc(l, p) -> float:
    """Scale-invariant incidence residual |l^T p| / (||l|| ||p||)."""
    return abs(l @ p) / (np.linalg.norm(l) * np.linalg.norm(p))


def orient2d(a, b, c) -> float:
    """Twice the signed area of triangle abc."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def collinear(a, b, c, tau: float = 1e3 * EPS) -> bool:
    """Scale-aware collinearity predicate: |2A| <= tau * (product of two edge lengths)."""
    a, b, c = map(np.asarray, (a, b, c))
    scale = np.linalg.norm(b - a) * np.linalg.norm(c - a)
    return scale == 0.0 or abs(orient2d(a, b, c)) <= tau * scale


# ---- Task P1: circumcircle ------------------------------------------------
def circumcircle(P0, P1, P2) -> ConstructionResult:
    P = [np.asarray(v, float) for v in (P0, P1, P2)]
    if collinear(*P):
        # bisectors are parallel: their meet is the ideal point of their direction
        return ConstructionResult("ideal", info={"reason": "collinear input"})

    def bisector(a, b):
        m = hpoint(*(0.5 * (a + b)))
        d = b - a
        v_perp = np.array([-d[1], d[0], 0.0])      # ideal point: perpendicular direction
        return join(m, v_perp)

    b01, b12 = bisector(P[0], P[1]), bisector(P[1], P[2])
    c_h = meet(b01, b12)
    C = dehom(c_h)
    r = np.linalg.norm(C - P[0])
    res = max(abs(np.linalg.norm(C - Pi) - r) for Pi in P) / max(r, 1.0)
    res = max(res, rho_inc(b01, c_h), rho_inc(b12, c_h))
    return ConstructionResult("regular", [C], C, res,
                              {"radius": r, "bisectors": (b01, b12), "c_h": c_h})


# ---- Task P2: perpendicular foot through an ideal point --------------------
def perpendicular_foot(P0, P1, P2) -> ConstructionResult:
    p0, p1, p2 = (hpoint(*v) for v in (P0, P1, P2))
    L = join(p0, p1)
    I_perp = np.array([L[0], L[1], 0.0])            # normal direction of L as ideal point
    n0, n1 = join(p0, I_perp), join(p1, I_perp)
    S1 = meet(n0, n1)                                # = I_perp up to scale (ideal!)
    n2 = join(S1, p2)
    f_h = meet(n2, L)
    F = dehom(f_h)
    d = np.asarray(P1, float) - np.asarray(P0, float)
    res = max(rho_inc(L, f_h), rho_inc(n2, f_h),
              abs(d @ (F - np.asarray(P2, float))) / (np.linalg.norm(d) * max(1.0, np.linalg.norm(F - P2))))
    return ConstructionResult("regular", [F], F, res,
                              {"L": L, "S1": S1, "n0": n0, "n1": n1, "n2": n2,
                               "S1_is_ideal": is_ideal(S1)})


# ---- Task P4: isosceles apices on the parallel ------------------------------
def isosceles_candidates(P0, P1, P2, tau: float = 1e-12) -> ConstructionResult:
    """All apices Q on the parallel to P1P2 through P0 with an isosceles triangle P1 P2 Q.

    Returns labelled candidates  (label, Q)  where the label records the locus
    ('bisector', 'circle P1', 'circle P2').  Coincident candidates are merged."""
    P0, P1, P2 = (np.asarray(v, float) for v in (P0, P1, P2))
    u = P2 - P1
    d = np.linalg.norm(u)
    u_hat = u / d
    n_hat = np.array([-u_hat[1], u_hat[0]])
    h = (P0 - P1) @ n_hat                              # signed height of the parallel
    if abs(h) <= tau * d:
        return ConstructionResult("degenerate", info={"reason": "P0 on the base line", "h": h, "d": d})

    # parametrise the parallel as Q(t) = P1 + t*u_hat + h*n_hat
    cands = [("bisector", 0.5 * d)]                  # always exactly one intersection
    for label, centre_t in (("circle P1", 0.0), ("circle P2", d)):
        disc = d * d - h * h
        if disc > tau * d * d:
            s = np.sqrt(disc)
            cands += [(label, centre_t - s), (label, centre_t + s)]
        elif disc >= -tau * d * d:
            cands += [(label + " (tangent)", centre_t)]
    merged: list[tuple[str, float]] = []
    for lab, t in sorted(cands, key=lambda c: c[1]):
        if merged and abs(t - merged[-1][1]) <= 1e-9 * d:
            merged[-1] = (merged[-1][0] + " = " + lab, merged[-1][1])
        else:
            merged.append((lab, t))
    pts = [(lab, P1 + t * u_hat + h * n_hat) for lab, t in merged]

    def iso_res(Q):
        a, b = np.linalg.norm(Q - P1), np.linalg.norm(Q - P2)
        return min(abs(a - b), abs(a - d), abs(b - d)) / d

    res = max(iso_res(Q) for _, Q in pts)
    status = "regular" if len(pts) == 5 else "reduced"
    return ConstructionResult(status, pts, None, res, {"h": h, "d": d})


def isosceles_count(
    h_over_d: float,
    tau: float = 1e-12,
    merge_tol: float = 1e-9,
) -> int:
    """Number of distinct candidates as a function of |h|/d.

    The tolerances mirror isosceles_candidates: tau controls the
    degenerate/tangent classification, while merge_tol controls merging
    of the equilateral coincidence.
    """
    x = abs(h_over_d)
    if x <= tau:
        return 0

    disc = 1.0 - x * x
    if disc < -tau:
        return 1
    if abs(disc) <= tau:
        return 3

    s = np.sqrt(max(disc, 0.0))
    if abs(s - 0.5) <= merge_tol:
        return 3

    return 5


# --------------------------------------------------------------------------
# Distance geometry in R^3
# --------------------------------------------------------------------------
def trilaterate(c, r) -> ConstructionResult:
    """Intersection of three spheres (centres c[i], radii r[i]) via linear reduction."""
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    A = 2 * (c[1:] - c[0])
    b = r[0] ** 2 - r[1:] ** 2 + np.sum(c[1:] ** 2, 1) - np.sum(c[0] ** 2)
    # line of the radical axis: x = x0 + s*n
    n = np.cross(A[0], A[1])
    if np.linalg.norm(n) <= 1e3 * EPS * np.linalg.norm(A[0]) * np.linalg.norm(A[1]):
        return ConstructionResult("degenerate", info={"reason": "collinear centres"})
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    n = n / np.linalg.norm(n)
    w = x0 - c[0]
    B = w @ n
    Cq = w @ w - r[0] ** 2
    disc = B * B - Cq
    if disc < -1e-12 * max(1.0, r[0] ** 2):
        return ConstructionResult("empty", info={"disc": disc})
    s = np.sqrt(max(disc, 0.0))
    X = [x0 + (-B + s) * n, x0 + (-B - s) * n]
    res = max(abs(np.linalg.norm(x - ci) - ri) for x in X for ci, ri in zip(c, r))
    return ConstructionResult("tangent" if s == 0 else "regular", X, None, res)


def select_max(cands, key: Callable) -> np.ndarray:
    return max(cands, key=key)


def sphere_plane_circle(centre, radius, plane_point, plane_normal):
    """Circle = sphere  cap  plane:  returns (centre, radius, normal) or None."""
    n = np.asarray(plane_normal, float)
    n = n / np.linalg.norm(n)
    dist = (np.asarray(centre) - plane_point) @ n
    if abs(dist) > radius:
        return None
    return np.asarray(centre) - dist * n, np.sqrt(radius ** 2 - dist ** 2), n


def circle_plane_points(circ, plane_point, plane_normal):
    """Circle cap plane (the two feet), assuming the planes are not parallel."""
    cc, rr, nc = circ
    m = np.asarray(plane_normal, float)
    m = m / np.linalg.norm(m)
    # direction of the line where the two planes meet
    t = np.cross(nc, m)
    t = t / np.linalg.norm(t)
    # point on that line closest to cc, inside circle plane
    v = np.cross(nc, t)                     # in circle plane, perpendicular to t
    s = ((plane_point - cc) @ m) / (v @ m)
    base = cc + s * v
    q = rr ** 2 - s ** 2
    if q < 0:
        return []
    return [base + np.sqrt(q) * t, base - np.sqrt(q) * t]


def convex_hull_2d(P):
    """Andrew monotone chain; returns hull vertices counter-clockwise."""
    P = sorted(map(tuple, np.asarray(P, float)))
    if len(P) <= 2:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def stability_margin(support_xz, com_xz) -> float:
    """Signed distance of the projected centre of mass to the support-polygon boundary
    (positive = inside = statically stable)."""
    H = convex_hull_2d(support_xz)
    q = np.asarray(com_xz, float)
    dmin = np.inf
    inside = True
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        e = b - a
        # signed distance to the edge line (left = inside for CCW order)
        sd = (e[0] * (q[1] - a[1]) - e[1] * (q[0] - a[0])) / np.linalg.norm(e)
        inside &= sd >= 0
        dmin = min(dmin, abs(sd) if sd >= 0 else np.inf)
    if inside:
        return dmin
    # outside: distance to the polygon (negative)
    best = np.inf
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        t = np.clip((q - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        best = min(best, np.linalg.norm(q - (a + t * (b - a))))
    return -best


def tripod(A, B, C, rA, rB, rC, r4, up=np.array([0.0, 1.0, 0.0])):
    """Apex from three spheres + fourth support (the CLUCalc task 1)."""
    tri = trilaterate([A, B, C], [rA, rB, rC])
    if tri.status in ("empty", "degenerate"):
        return tri
    S_plus = select_max(tri.candidates, key=lambda x: x @ up)          # height policy
    circ = sphere_plane_circle(S_plus, r4, np.asarray(A, float), up)
    # helper plane tau: vertical plane through A and the apex (contains S+ and S-)
    S_minus = [x for x in tri.candidates if x is not S_plus][0]
    tau_n = np.cross(S_plus - np.asarray(A, float), S_minus - np.asarray(A, float))
    feet = circle_plane_points(circ, np.asarray(A, float), tau_n)
    Q_out = select_max(feet, key=lambda q: np.linalg.norm(q - A))      # 'outer' policy
    legs = [(A, rA), (B, rB), (C, rC), (Q_out, r4)]
    res = max(abs(np.linalg.norm(S_plus - np.asarray(p)) - r) for p, r in legs)
    res = max(res, abs((Q_out - A) @ up))
    return ConstructionResult("regular", tri.candidates, S_plus, res,
                              {"S_minus": S_minus, "feet": feet, "Q_out": Q_out,
                               "circle": circ})


# --------------------------------------------------------------------------
# Kinematic chains (in the plane of the helper chart)
# --------------------------------------------------------------------------
def _in_plane_perp(u, up, tau: float = 1e-12):
    """Unit vector perpendicular to u inside the plane spanned by u and `up`,
    oriented towards `up`.  Built from exact rotations / cross products so that
    orthogonality holds to machine precision."""
    if u.size == 2:
        p = np.array([-u[1], u[0]])
        return p if p @ up >= 0 else -p
    n = np.cross(u, up)
    nn = np.linalg.norm(n)
    if nn <= tau * np.linalg.norm(up):
        return None
    return np.cross(n / nn, u)

def two_link(O, T, l1=1.0, l2=1.0, up=None, tau: float = 1e-12) -> ConstructionResult:
    """Elbow of a planar/spatial two-link arm in the motion plane spanned by T-O and `up`."""
    O, T = np.asarray(O, float), np.asarray(T, float)
    dim = O.size
    if up is None:
        up = np.eye(dim)[1]
    v = T - O
    d = np.linalg.norm(v)
    if d <= tau:
        return ConstructionResult("degenerate", info={"reason": "target at shoulder"})
    u = v / d
    u_perp = _in_plane_perp(u, up)
    if u_perp is None:
        return ConstructionResult("degenerate", info={"reason": "helper plane undefined"})
    a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    q = l1 ** 2 - a ** 2
    if q < -tau * l1 ** 2:
        return ConstructionResult("empty", info={"d": d})
    h = np.sqrt(max(q, 0.0))
    E = [O + a * u + h * u_perp, O + a * u - h * u_perp]
    sel = select_max(E, key=lambda e: e @ up)
    res = max(abs(np.linalg.norm(sel - O) - l1), abs(np.linalg.norm(T - sel) - l2))
    return ConstructionResult("tangent" if q <= tau * l1 ** 2 else "regular", E, sel, res,
                              {"d": d, "h": h})


def three_link_trapezoid(O, T, l=1.0, up=None) -> ConstructionResult:
    """Three equal links as an isosceles trapezoid O-E1-E2-T with E1E2 || OT.

    This is the closed form of what the CLUCalc task 3 actually constructs:
    E2 on the bisector plane of O and W+ = T + l*u, at distance l from T;
    E1 is the mirror image of E2 in the bisector plane of O and T."""
    O, T = np.asarray(O, float), np.asarray(T, float)
    dim = O.size
    if up is None:
        up = np.eye(dim)[1]
    v = T - O
    d = np.linalg.norm(v)
    if d == 0:
        return ConstructionResult("degenerate", info={"reason": "target at shoulder"})
    u = v / d
    u_perp = _in_plane_perp(u, up)
    if u_perp is None:
        return ConstructionResult("degenerate", info={"reason": "helper plane undefined"})
    a = 0.5 * (d - l)
    q = l * l - a * a
    if q < 0:
        return ConstructionResult("empty", info={"d": d})
    h = np.sqrt(q)
    E1 = O + a * u + h * u_perp
    E2 = E1 + l * u
    res = max(abs(np.linalg.norm(E1 - O) - l), abs(np.linalg.norm(E2 - E1) - l),
              abs(np.linalg.norm(T - E2) - l))
    return ConstructionResult("tangent" if q == 0 else "regular",
                              [(E1, E2), (O + a * u - h * u_perp, O + a * u - h * u_perp + l * u)],
                              (E1, E2), res, {"d": d, "h": h})
