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
    denominator = np.linalg.norm(l) * np.linalg.norm(p)
    if denominator == 0.0:
        return float("inf")
    return abs(l @ p) / denominator


def orient2d(a, b, c) -> float:
    """Twice the signed area of triangle abc."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def collinear(a, b, c, tau: float = 1e3 * EPS) -> bool:
    """Scale-aware collinearity predicate: |2A| <= tau * (product of two edge lengths)."""
    a, b, c = map(np.asarray, (a, b, c))
    scale = np.linalg.norm(b - a) * np.linalg.norm(c - a)
    return scale == 0.0 or abs(orient2d(a, b, c)) <= tau * scale


# ---- Task P1: circumcircle ------------------------------------------------
def circumcircle(P0, P1, P2, tau: float = 1e-12) -> ConstructionResult:
    P = [np.asarray(v, float) for v in (P0, P1, P2)]
    pair_distances = [np.linalg.norm(P[i] - P[j]) for i in range(3) for j in range(i)]
    span = max(pair_distances)
    if span == 0.0 or any(distance <= tau * span for distance in pair_distances):
        return ConstructionResult("degenerate", info={"reason": "repeated input point"})
    if collinear(*P, tau=tau):
        # distinct collinear points have their circumcentre at infinity
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
def perpendicular_foot(P0, P1, P2, tau: float = 1e-12) -> ConstructionResult:
    P0, P1, P2 = (np.asarray(v, float) for v in (P0, P1, P2))
    base_length = np.linalg.norm(P1 - P0)
    scale = max(base_length, np.linalg.norm(P2 - P0), np.linalg.norm(P2 - P1))
    if base_length == 0.0 or base_length <= tau * scale:
        return ConstructionResult("degenerate", info={"reason": "P0 and P1 coincide"})

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
    base_scale = max(d, np.linalg.norm(P0 - P1), np.linalg.norm(P0 - P2))
    if d == 0.0 or d <= tau * base_scale:
        return ConstructionResult("degenerate", info={"reason": "P1 and P2 coincide"})

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
        if merged and abs(t - merged[-1][1]) <= tau * d:
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


def isosceles_count(h_over_d: float, tau: float = 1e-12) -> int:
    """Number of distinct candidates as a function of |h|/d.

    The tolerance mirrors isosceles_candidates so that only numerically
    indistinguishable geometric coincidences are merged.
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
    if abs(s - 0.5) <= tau:
        return 3

    return 5


# --------------------------------------------------------------------------
# Distance geometry in R^3
# --------------------------------------------------------------------------
def trilaterate(c, r, tau: float = 1e-12) -> ConstructionResult:
    """Intersection of three spheres in R^3 via linear reduction."""
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    if c.shape != (3, 3) or r.shape != (3,):
        raise ValueError("trilaterate expects three 3D centres and three radii")
    if np.any(r < 0):
        raise ValueError("sphere radii must be non-negative")

    A = 2 * (c[1:] - c[0])
    b = r[0] ** 2 - r[1:] ** 2 + np.sum(c[1:] ** 2, 1) - np.sum(c[0] ** 2)
    n = np.cross(A[0], A[1])
    nn = np.linalg.norm(n)
    if nn <= 1e3 * EPS * np.linalg.norm(A[0]) * np.linalg.norm(A[1]):
        return ConstructionResult("degenerate", info={"reason": "collinear centres"})

    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    n = n / nn
    w = x0 - c[0]
    B = w @ n
    Cq = w @ w - r[0] ** 2
    disc = B * B - Cq
    disc_scale = max(r[0] ** 2, float(w @ w), np.finfo(float).tiny)
    disc_tol = tau * disc_scale
    if disc < -disc_tol:
        return ConstructionResult("empty", info={"disc": disc})
    if abs(disc) <= disc_tol:
        disc = 0.0

    s = np.sqrt(max(disc, 0.0))
    X = [x0 + (-B + s) * n, x0 + (-B - s) * n]
    res = max(abs(np.linalg.norm(x - ci) - ri) for x in X for ci, ri in zip(c, r))
    return ConstructionResult("tangent" if disc == 0.0 else "regular", X, None, res)


def select_max(cands, key: Callable) -> np.ndarray:
    return max(cands, key=key)


def sphere_plane_circle(centre, radius, plane_point, plane_normal, tau: float = 1e-12):
    """Circle = sphere cap plane; return (centre, radius, normal) or None."""
    if radius < 0:
        raise ValueError("sphere radius must be non-negative")
    n = np.asarray(plane_normal, float)
    nn = np.linalg.norm(n)
    if nn == 0.0:
        raise ValueError("plane normal must be non-zero")
    n = n / nn
    dist = (np.asarray(centre, float) - np.asarray(plane_point, float)) @ n
    scale = max(radius, abs(dist), np.finfo(float).tiny)
    tol = tau * scale
    if abs(dist) > radius + tol:
        return None
    r2 = radius ** 2 - dist ** 2
    return np.asarray(centre, float) - dist * n, np.sqrt(max(r2, 0.0)), n


def circle_plane_points(circ, plane_point, plane_normal, tau: float = 1e-12):
    """Intersect a circle with a non-parallel plane."""
    if circ is None:
        return []
    cc, rr, nc = circ
    m = np.asarray(plane_normal, float)
    mm = np.linalg.norm(m)
    if mm == 0.0:
        return []
    m = m / mm

    t = np.cross(nc, m)
    tt = np.linalg.norm(t)
    if tt <= tau:
        return []
    t = t / tt

    v = np.cross(nc, t)
    denom = v @ m
    if abs(denom) <= tau:
        return []
    s = ((np.asarray(plane_point, float) - cc) @ m) / denom
    base = cc + s * v
    q = rr ** 2 - s ** 2
    q_scale = max(rr ** 2, s ** 2, np.finfo(float).tiny)
    tol = tau * q_scale
    if q < -tol:
        return []
    root = np.sqrt(max(q, 0.0))
    if root == 0.0:
        return [base]
    return [base + root * t, base - root * t]


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
    """Signed distance from projected load point to a non-degenerate support polygon."""
    H = convex_hull_2d(support_xz)
    if len(H) < 3:
        raise ValueError("support polygon requires at least three non-collinear points")
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


def tripod(A, B, C, rA, rB, rC, r4, up=None):
    """Apex from three spheres plus a fourth support candidate."""
    A, B, C = (np.asarray(p, float) for p in (A, B, C))
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    up = np.asarray(up, float)
    if up.shape != (3,) or np.linalg.norm(up) == 0.0:
        raise ValueError("up must be a non-zero 3D direction")
    up = up / np.linalg.norm(up)

    tri = trilaterate([A, B, C], [rA, rB, rC])
    if tri.status in ("empty", "degenerate"):
        return tri

    S_plus = select_max(tri.candidates, key=lambda x: x @ up)
    circ = sphere_plane_circle(S_plus, r4, A, up)
    if circ is None:
        return ConstructionResult("empty", tri.candidates, S_plus,
                                  info={"reason": "fourth-leg sphere misses ground plane"})

    # Helper plane tau: vertical plane through A and the two apex branches.
    S_minus = min(tri.candidates, key=lambda x: x @ up)
    va, vb = S_plus - A, S_minus - A
    tau_n = np.cross(va, vb)
    if np.linalg.norm(tau_n) <= 1e-12 * np.linalg.norm(va) * np.linalg.norm(vb):
        return ConstructionResult("degenerate", tri.candidates, S_plus,
                                  info={"reason": "helper plane undefined at tangent apex"})

    feet = circle_plane_points(circ, A, tau_n)
    if not feet:
        return ConstructionResult("empty", tri.candidates, S_plus,
                                  info={"reason": "no real fourth-support foot"})

    Q_out = select_max(feet, key=lambda q: np.linalg.norm(q - A))
    legs = [(A, rA), (B, rB), (C, rC), (Q_out, r4)]
    res = max(abs(np.linalg.norm(S_plus - p) - r) for p, r in legs)
    res = max(res, abs((Q_out - A) @ up))
    return ConstructionResult("regular", tri.candidates, S_plus, res,
                              {"S_minus": S_minus, "feet": feet, "Q_out": Q_out,
                               "circle": circ})


# --------------------------------------------------------------------------
# Kinematic chains (in the plane of the helper chart)
# --------------------------------------------------------------------------
def _in_plane_perp(u, up, projection_tol_sq: float = 1e-12):
    """Preferred transverse direction for a motion plane.

    The preferred `up` direction is projected orthogonally to the target
    direction.  If that projection is too small (target parallel to `up`), a
    deterministic coordinate axis is used instead.  This mirrors the fallback
    helper plane used by the revised CGA construction.
    """
    u = np.asarray(u, float)
    up = np.asarray(up, float)
    if u.ndim != 1 or up.shape != u.shape:
        raise ValueError("u and up must be vectors of the same dimension")
    up_norm = np.linalg.norm(up)
    if up_norm == 0.0:
        raise ValueError("up direction must be non-zero")
    up_hat = up / up_norm

    p = up_hat - (up_hat @ u) * u
    if p @ p <= projection_tol_sq:
        axis = np.eye(u.size)[int(np.argmin(np.abs(u)))]
        p = axis - (axis @ u) * u

    pn = np.linalg.norm(p)
    if pn == 0.0:
        raise ValueError("could not construct a transverse direction")
    return p / pn

def two_link(O, T, l1=1.0, l2=1.0, up=None, tau: float = 1e-12) -> ConstructionResult:
    """Elbow of a planar/spatial two-link arm in the motion plane spanned by T-O and `up`."""
    O, T = np.asarray(O, float), np.asarray(T, float)
    dim = O.size
    if up is None:
        up = np.eye(dim)[1]
    v = T - O
    d = np.linalg.norm(v)
    if l1 <= 0 or l2 <= 0:
        return ConstructionResult("degenerate", info={"reason": "link lengths must be positive"})
    if d <= tau * max(l1, l2):
        return ConstructionResult("degenerate", info={"reason": "target at shoulder"})
    u = v / d
    u_perp = _in_plane_perp(u, up)
    a = (l1 ** 2 - l2 ** 2 + d ** 2) / (2 * d)
    q = l1 ** 2 - a ** 2
    if q < -tau * l1 ** 2:
        return ConstructionResult("empty", info={"d": d})
    h = np.sqrt(max(q, 0.0))
    E = [O + a * u + h * u_perp, O + a * u - h * u_perp]
    sel = E[0]
    res = max(abs(np.linalg.norm(sel - O) - l1), abs(np.linalg.norm(T - sel) - l2))
    return ConstructionResult("tangent" if q <= tau * l1 ** 2 else "regular", E, sel, res,
                              {"d": d, "h": h})


def three_link_trapezoid(O, T, l=1.0, up=None, tau: float = 1e-12) -> ConstructionResult:
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
    if l <= 0:
        return ConstructionResult("degenerate", info={"reason": "link length must be positive"})
    if d <= tau * l:
        return ConstructionResult("degenerate", info={"reason": "target at shoulder"})
    u = v / d
    u_perp = _in_plane_perp(u, up)
    a = 0.5 * (d - l)
    q = l * l - a * a
    if q < -tau * l * l:
        return ConstructionResult("empty", info={"d": d})
    h = np.sqrt(max(q, 0.0))
    E1 = O + a * u + h * u_perp
    E2 = E1 + l * u
    res = max(abs(np.linalg.norm(E1 - O) - l), abs(np.linalg.norm(E2 - E1) - l),
              abs(np.linalg.norm(T - E2) - l))
    return ConstructionResult("tangent" if q <= tau * l * l else "regular",
                              [(E1, E2), (O + a * u - h * u_perp, O + a * u - h * u_perp + l * u)],
                              (E1, E2), res, {"d": d, "h": h})
