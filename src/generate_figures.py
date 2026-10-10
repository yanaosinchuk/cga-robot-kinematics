#!/usr/bin/env python3
"""
generate_figures.py -- reproduces every number, table row and figure of the paper
"From Ideal Points to Robot Joints" (Y. Osinchuk).

    python src/generate_figures.py            # full run
    FAST=1 python src/generate_figures.py     # reduced smoke run

Outputs:
    figures/*.pdf (vector) and figures/*.png (preview)
    results/results.tex   (LaTeX macros used by paper/main.tex)
    results/results.json  (machine-readable ledger)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, Polygon                    # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                           # noqa: E402,F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from geometry import (circumcircle, perpendicular_foot, isosceles_candidates,     # noqa: E402
                      isosceles_count, tripod, two_link, three_link_trapezoid,
                      stability_margin, convex_hull_2d, rho_inc, hpoint, join, meet)
import cga_constructions as hw                                   # noqa: E402

ROOT = HERE.parent
OUT = Path(os.environ.get("OUT_DIR", ROOT))          # redirect outputs (used by tests)
FIG = OUT / "figures"
GEN = OUT / "results"
KEEP_FIGURES = os.environ.get("KEEP_FIGURES") == "1"  # keep figure objects for inspection
FIGURES: dict = {}
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

SEED = 20260922
FAST = os.environ.get("FAST") == "1"
N_PROJ = 500 if FAST else 5000
N_KIN = 1000 if FAST else 10000
N_ISO = 2000 if FAST else 20000
N_CGA = 200 if FAST else 2000
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ style
INK, MUTED, GRID = "#20242A", "#68727E", "#D6DBE1"
RED, BLUE, TEAL, ORANGE, PURPLE, GREEN = "#D71920", "#276FBF", "#2A9D8F", "#E9A23B", "#7B61C4", "#4C9F38"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": GRID,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    if KEEP_FIGURES:
        FIGURES[name] = fig
    else:
        plt.close(fig)


def draw_hline(ax, l, xlim, **kw):
    """Draw the homogeneous line l = (a,b,c) inside the x-range xlim."""
    a, b, c = l
    if abs(b) > 1e-12 * max(abs(a), 1):
        xs = np.array(xlim)
        ax.plot(xs, -(a * xs + c) / b, **kw)
    else:
        x = -c / a
        ax.axvline(x, **kw)


def label(ax, p, text, dx=0.06, dy=0.06, color=INK, **kw):
    ax.annotate(text, p, xytext=(p[0] + dx, p[1] + dy), color=color, fontsize=9, **kw)


results: dict = {"seed": SEED, "fast": FAST}

# =========================================================================
# Experiment 1: projective covariance
# =========================================================================
def angle_err(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


errs = []
while len(errs) < N_PROJ:
    P = [hpoint(*rng.uniform(-2, 2, 2)) for _ in range(4)]
    H = np.eye(3) + 0.22 * rng.normal(size=(3, 3))
    if abs(np.linalg.det(H)) <= 0.15 or np.linalg.cond(H) >= 50:
        continue
    l, m = join(P[0], P[1]), join(P[2], P[3])
    s = meet(l, m)
    Hit = np.linalg.inv(H).T
    Pp = [H @ p for p in P]
    lp, mp = join(Pp[0], Pp[1]), join(Pp[2], Pp[3])
    e = max(rho_inc(Hit @ l, Pp[0]), rho_inc(Hit @ l, Pp[1]),
            angle_err(lp, Hit @ l), angle_err(mp, Hit @ m), angle_err(meet(lp, mp), H @ s))
    errs.append(e)
errs = np.array(errs)
results["projective"] = {"n": N_PROJ, "median": float(np.median(errs)), "max": float(errs.max())}

# =========================================================================
# Experiment 2: kinematic constraints (closed forms)
# =========================================================================
def annulus(n, r0, r1, dim=2):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(phi), r * np.sin(phi)]


O2 = np.zeros(2)
res2 = np.array([two_link(O2, t).residual for t in annulus(N_KIN, 0.05, 1.95)])
res3 = np.array([three_link_trapezoid(O2, t).residual for t in annulus(N_KIN, 0.05, 2.95)])
results["two_link"] = {"n": N_KIN, "median": float(np.median(res2)), "max": float(res2.max())}
results["three_link"] = {"n": N_KIN, "median": float(np.median(res3)), "max": float(res3.max())}

# isosceles proposition check
iso_mismatch, iso_res = 0, []
for _ in range(N_ISO):
    P1, P2 = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    if np.linalg.norm(P2 - P1) < 0.2:
        continue
    P0 = rng.uniform(-3, 3, 2)
    r = isosceles_candidates(P0, P1, P2)
    if r.status == "degenerate":
        continue
    iso_res.append(r.residual)
    if len(r.candidates) != isosceles_count(r.info["h"] / r.info["d"]):
        iso_mismatch += 1
# the two exceptional heights
for x in (1.0, np.sqrt(3) / 2):
    r = isosceles_candidates([0.3, x * 2.0], [-1.0, 0.0], [1.0, 0.0])
    iso_mismatch += len(r.candidates) != 3
results["isosceles"] = {"n": len(iso_res), "max": float(np.max(iso_res)), "mismatch": iso_mismatch}

# =========================================================================
# Experiment 3: tripod and support polygon
# =========================================================================
A3, B3, C3 = np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 0, 2])
trip = tripod(A3, B3, C3, 2.69, 1.8, 1.5, 1.1)
S_plus, Q_out = trip.selected, trip.info["Q_out"]
Q_in = [f for f in trip.info["feet"] if f is not Q_out][0]
base_xz = np.array([A3[[0, 2]], B3[[0, 2]], C3[[0, 2]]])
com = S_plus[[0, 2]]
m_tri = stability_margin(base_xz, com)
m_out = stability_margin(np.vstack([base_xz, Q_out[[0, 2]]]), com)
m_in = stability_margin(np.vstack([base_xz, Q_in[[0, 2]]]), com)
results["tripod"] = {"S_plus": S_plus.tolist(), "Q_out": Q_out.tolist(), "Q_in": Q_in.tolist(),
                     "residual": float(trip.residual), "margin_tripod": m_tri,
                     "margin_out": m_out, "margin_in": m_in}

# =========================================================================
# Experiment 4: conditioning near singularities
# =========================================================================
hs = np.logspace(-1, -11, 21)
kappas, cerr = [], []
for h in hs:
    P0, P1, P2 = np.array([-1.0, 0]), np.array([1.0, 0]), np.array([0.0, h])
    M = 2 * np.array([P1 - P0, P2 - P0])
    rhs = np.array([P1 @ P1 - P0 @ P0, P2 @ P2 - P0 @ P0])
    c_true = np.linalg.solve(M, rhs)
    kappas.append(np.linalg.cond(M))
    e = []
    for _ in range(400):
        Q = [p + 1e-10 * rng.normal(size=2) for p in (P0, P1, P2)]
        Mq = 2 * np.array([Q[1] - Q[0], Q[2] - Q[0]])
        rq = np.array([Q[1] @ Q[1] - Q[0] @ Q[0], Q[2] @ Q[2] - Q[0] @ Q[0]])
        e.append(np.linalg.norm(np.linalg.solve(Mq, rq) - c_true) / np.linalg.norm(c_true))
    cerr.append(np.median(e))
kappas, cerr = np.array(kappas), np.array(cerr)
cond_rows = [0, 6, 12, 18, 20]
results["conditioning"] = [{"h": float(hs[i]), "kappa": float(kappas[i]), "err": float(cerr[i])}
                           for i in cond_rows]

# three-link: representation singularity of the homework construction at d = 1
deltas = np.logspace(-1, -14, 27)
dirs = rng.normal(size=(8 if FAST else 20, 3))
dirs[:, 1] = np.abs(dirs[:, 1]) * 0.3            # keep away from the helper-plane degeneracy
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
err_orig, err_rev = [], []
for dl in deltas:
    eo, er = [], []
    for u in dirs:
        t = (1 + dl) * u
        ref = three_link_trapezoid(np.zeros(3), t).selected
        o = hw.three_link_original(*t)
        r = hw.three_link_revised(*t)
        eo.append(max(np.linalg.norm(o[0] - ref[0]), np.linalg.norm(o[1] - ref[1])))
        er.append(max(np.linalg.norm(r[0] - ref[0]), np.linalg.norm(r[1] - ref[1])))
    err_orig.append(np.median(eo))
    err_rev.append(np.median(er))
err_orig, err_rev = np.array(err_orig), np.array(err_rev)
results["three_link_singularity"] = {
    "delta": deltas.tolist(), "orig": err_orig.tolist(), "rev": err_rev.tolist(),
    "orig_at_1e-7": float(np.interp(-7, np.log10(deltas[::-1]), err_orig[::-1])),
    "rev_max": float(err_rev.max())}

# =========================================================================
# Experiment 5: executing the CLUCalc constructions in a CGA engine
# =========================================================================
def ball(n, r0, r1):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = np.cbrt(rng.uniform(r0 ** 3, r1 ** 3, n))
    return v * r[:, None]


cga = {}
T2 = ball(N_CGA, 0.05, 1.95)
e_o, e_r = [], []
for t in T2:
    ref = two_link(np.zeros(3), t).selected
    e_o.append(np.linalg.norm(hw.two_link_original(*t) - ref))
    e_r.append(np.linalg.norm(hw.two_link_revised(*t) - ref))
cga["two_orig"] = float(np.max(e_o))
cga["two_rev"] = float(np.max(e_r))

T3 = ball(N_CGA, 0.05, 2.95)
e_o, e_r = [], []
for t in T3:
    ref = three_link_trapezoid(np.zeros(3), t).selected
    o = hw.three_link_original(*t)
    r = hw.three_link_revised(*t)
    e_o.append(max(np.linalg.norm(o[0] - ref[0]), np.linalg.norm(o[1] - ref[1])))
    e_r.append(max(np.linalg.norm(r[0] - ref[0]), np.linalg.norm(r[1] - ref[1])))
e_o, e_r = np.array(e_o), np.array(e_r)
cga["three_orig_median"] = float(np.median(e_o))
cga["three_orig"] = float(e_o.max())
cga["three_rev"] = float(e_r.max())
cga["three_n_bad"] = int(np.sum(e_o > 1e-12))
cga["n"] = N_CGA
# opposite ordering of the extracted pair (other sign convention): count failures
n_fail = 0
for t in T3:
    try:
        hw.three_link_original(*t, swap=True)
    except hw.Imaginary:
        n_fail += 1
cga["three_swap_fail"] = n_fail / len(T3)
to, tr = hw.tripod_original(), hw.tripod_revised()
cga["tripod_rev"] = float(max(np.linalg.norm(tr["spitze"] - S_plus), np.linalg.norm(tr["aussen"] - Q_out)))
cga["tripod_orig"] = float(max(np.linalg.norm(to["spitze"] - S_plus), np.linalg.norm(to["aussen"] - Q_out)))
results["cga"] = cga

# =========================================================================
# Figures
# =========================================================================
# ---- Figure 1: projective constructions --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.05))
P = [np.array([-1.4, 0.1]), np.array([1.55, 0.35]), np.array([0.1, 2.0])]
cc = circumcircle(*P)
ax = axs[0]
xl = (-2.1, 2.3)
ax.add_patch(Circle(cc.selected, cc.info["radius"], fill=False, color=BLUE, lw=1.3))
tri = np.array(P + [P[0]])
ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1)
draw_hline(ax, cc.info["bisectors"][0], xl, color=TEAL, ls="--", lw=1, label=r"bisector $b_{01}$")
draw_hline(ax, cc.info["bisectors"][1], xl, color=ORANGE, ls="--", lw=1, label=r"bisector $b_{12}$")
for i, p in enumerate(P):
    ax.plot(*p, "o", color=INK, ms=4)
M01, M12 = 0.5 * (P[0] + P[1]), 0.5 * (P[1] + P[2])
ax.plot(*M01, "s", color=TEAL, ms=3.5)
ax.plot(*M12, "s", color=ORANGE, ms=3.5)
ax.plot(*cc.selected, "x", color=RED, ms=7, mew=1.8)
label(ax, P[0], r"$P_0$", -0.35, -0.22)
label(ax, P[1], r"$P_1$", 0.08, -0.2)
label(ax, P[2], r"$P_2$", 0.08, 0.06)
label(ax, M01, r"$M_{01}$", 0.05, -0.28, color=TEAL)
label(ax, M12, r"$M_{12}$", 0.1, 0.02, color=ORANGE)
label(ax, cc.selected, r"$C\sim b_{01}\times b_{12}$", 0.12, -0.08, color=RED)
ax.set_xlim(xl)
ax.set_ylim(-1.3, 2.6)
ax.set_aspect("equal")
ax.set_title("(a) Circumcentre as a projective meet")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc="lower right")

ax = axs[1]
Q = [np.array([-1.8, -0.3]), np.array([1.6, 0.55]), np.array([0.15, 2.1])]
pf = perpendicular_foot(*Q)
xl = (-2.4, 2.4)
draw_hline(ax, pf.info["L"], xl, color=INK, lw=1.1, label=r"$L=p_0\times p_1$")
draw_hline(ax, pf.info["n0"], xl, color=TEAL, ls="--", lw=1, label=r"$n_0,\ n_1$ (parallel)")
draw_hline(ax, pf.info["n1"], xl, color=TEAL, ls="--", lw=1)
draw_hline(ax, pf.info["n2"], xl, color=RED, ls="-", lw=1.1, label=r"$n_2=S_1\times p_2$")
for p, t, off in zip(Q, ["$P_0$", "$P_1$", "$P_2$"], [(-0.15, 0.2), (0.1, -0.3), (0.12, 0.05)]):
    ax.plot(*p, "o", color=INK, ms=4)
    label(ax, p, t, *off)
F = pf.selected
ax.plot(*F, "o", color=BLUE, ms=5)
label(ax, F, r"$F\sim n_2\times L$", -1.05, -0.42, color=BLUE)
I = pf.info["S1"]
dvec = I[:2] / np.linalg.norm(I[:2])
ax.text(0.98, 0.98, r"$S_1=n_0\times n_1\sim(a,b,0)^{\mathsf{T}}$" "\n" r"ideal point: shared direction",
        transform=ax.transAxes, ha="right", va="top", color=PURPLE, fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
ax.set_xlim(xl)
ax.set_ylim(-1.55, 2.9)
ax.set_aspect("equal")
ax.set_title("(b) A finite foot through an ideal point")
ax.set_xlabel("$x$")
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "projective_constructions")

# ---- Figure 2: isosceles loci ------------------------------------------
fig = plt.figure(figsize=(6.6, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1])
ax = fig.add_subplot(gs[0])
P1, P2, P0 = np.array([-1.5, 0.0]), np.array([1.5, 0.0]), np.array([-3.0, 1.2])
iso = isosceles_candidates(P0, P1, P2)
d = iso.info["d"]
xs = np.array([-4.9, 4.9])
ax.plot(xs, [1.2, 1.2], color=ORANGE, lw=1.4, label=r"$L_\parallel$ through $P_0$")
ax.axvline(0, color=RED, ls="--", lw=0.9, label=r"bisector of $P_1P_2$")
ax.add_patch(Circle(P1, d, fill=False, ls=":", color=BLUE, lw=1.1, label=r"$\mathcal{S}(P_1,d)$"))
ax.add_patch(Circle(P2, d, fill=False, ls=":", color=TEAL, lw=1.1, label=r"$\mathcal{S}(P_2,d)$"))
ax.plot([P1[0], P2[0]], [0, 0], color=INK, lw=2)
for p, t in ((P1, "$P_1$"), (P2, "$P_2$")):
    ax.plot(*p, "o", color=INK, ms=5, zorder=5)
    ax.annotate(t, p, xytext=(p[0], -0.5), fontsize=9, ha="center")
ax.plot(*P0, "D", color=ORANGE, ms=4, zorder=5)
ax.annotate("$P_0$", P0, xytext=(P0[0], P0[1] - 0.42), fontsize=9, color=ORANGE, ha="center")
colors = {"bisector": RED, "circle P1": BLUE, "circle P2": TEAL}
order = sorted(iso.candidates, key=lambda c: c[1][0])
for k, (lab, q) in enumerate(order, 1):
    col = colors[lab.split(" (")[0].split(" =")[0]]
    for base in (P1, P2):
        ax.plot([q[0], base[0]], [q[1], base[1]], color=col, lw=0.5, alpha=0.55)
    ax.plot(*q, "o", color=col, ms=5, zorder=6)
    ax.annotate(f"$Q_{k}$", q, xytext=(q[0], q[1] + 0.36), color=col, fontsize=9, ha="center")
ax.set_xlim(-4.9, 4.9)
ax.set_ylim(-0.9, 3.4)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("(a) Five apices as line-locus intersections")
ax.legend(loc="upper center", ncol=2, fontsize=7)
ax = fig.add_subplot(gs[1])
r3 = np.sqrt(3) / 2
for x0, x1, yv in ((0.0, r3, 5), (r3, 1.0, 5), (1.0, 1.4, 1)):
    ax.plot([x0, x1], [yv, yv], color=INK, lw=1.4)
for xv, yv in ((0.0, 5), (r3, 5), (1.0, 5), (1.0, 1)):      # excluded values
    ax.plot([xv], [yv], "o", mfc="white", mec=INK, ms=4.2, zorder=5)
for xv in (r3, 1.0):                                       # isolated values
    ax.plot([xv], [3], "o", color=RED, ms=4.2, zorder=5)
ax.set_xticks([0, 0.5, np.sqrt(3) / 2, 1.0, 1.4])
ax.set_xticklabels(["0", "0.5", r"$\frac{\sqrt{3}}{2}$", "1", "1.4"])
ax.set_yticks([1, 3, 5])
ax.set_ylim(0, 6)
ax.set_xlabel(r"height ratio $|h|/d$")
ax.set_ylabel("distinct apices")
ax.set_title("(b) Candidate count")
ax.text(0.03, 5.35, "$h=0$: degenerate", fontsize=6.5, color=MUTED)
ax.annotate("equilateral\ncoincidence", (np.sqrt(3) / 2, 3), xytext=(0.18, 1.6), fontsize=7,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
ax.annotate("tangency", (1.0, 3), xytext=(1.03, 4.0), fontsize=7,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
fig.tight_layout()
save(fig, "isosceles_locus")

# ---- Figure 3: tripod ----------------------------------------------------
fig = plt.figure(figsize=(6.8, 3.7))
ax = fig.add_subplot(1, 2, 1, projection="3d")
S_minus = trip.info["S_minus"]


def P3(v):              # plot with height (y) vertical: (x, z, y)
    v = np.asarray(v)
    return v[0], v[2], v[1]


def cap(c, r, target, half_angle=0.55, n=18):
    """Wire-frame patch of the sphere (c, r) around the direction of `target`."""
    w = (target - c) / np.linalg.norm(target - c)
    a = np.cross(w, [1.0, 0, 0]) if abs(w[0]) < 0.9 else np.cross(w, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(w, a)
    th, ph = np.meshgrid(np.linspace(0, half_angle, n // 2), np.linspace(0, 2 * np.pi, n))
    pts = (c[:, None, None] + r * (np.cos(th) * w[:, None, None]
           + np.sin(th) * (np.cos(ph) * a[:, None, None] + np.sin(ph) * b[:, None, None])))
    return pts


for c, r, col in ((A3, 2.69, RED), (B3, 1.8, BLUE), (C3, 1.5, TEAL)):
    for tgt in (S_plus, S_minus):
        X, Y, Z = cap(c, r, tgt, half_angle=0.75 / r)
        ax.plot_wireframe(X, Z, Y, color=col, lw=0.35, alpha=0.45)
gx, gz = np.meshgrid(np.linspace(-2, 2.2, 2), np.linspace(-1, 3, 2))
ax.plot_surface(gx, gz, 0 * gx, color="#E8ECF2", alpha=0.35)
tri3 = np.array([A3, B3, C3, A3])
ax.plot(*P3(tri3.T), color=INK, lw=1)
for p, col in ((A3, RED), (B3, BLUE), (C3, TEAL)):
    ax.plot(*zip(P3(p), P3(S_plus)), color=col, lw=1.8)
    ax.plot(*zip(P3(p), P3(S_minus)), color=col, lw=0.8, ls=":")
ax.scatter(*P3(S_plus), color=RED, s=22, depthshade=False)
ax.scatter(*P3(S_minus), color=PURPLE, s=18, depthshade=False)
ax.plot(*zip(P3(S_plus), P3(Q_out)), color=ORANGE, lw=1.6, ls="--")
ax.scatter(*P3(Q_out), color=ORANGE, s=14, depthshade=False)
ax.text(*P3(Q_out + [0.1, -0.1, 0.1]), r"$Q_{\mathrm{out}}$", color=ORANGE, fontsize=8)
ax.text(*P3(S_plus + [0.35, 0.3, 0]), r"$S_+$", color=RED)
ax.text(*P3(S_minus + [0.45, -0.45, 0]), r"$S_-$", color=PURPLE)
for p, t in ((A3, "$A$"), (B3, "$B$"), (C3, "$C$")):
    ax.text(*P3(p + [-0.25, -0.3, -0.15]), t)
ax.set_xlabel("$x$", labelpad=-6)
ax.set_ylabel("$z$", labelpad=-6)
ax.set_zlabel("height $y$", labelpad=-7)
ax.tick_params(pad=-2, labelsize=6.5)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-1, 0, 1, 2, 3])
ax.set_zticks([-1, 0, 1])
ax.view_init(elev=16, azim=-58)
ax.set_box_aspect((4.2, 4, 3.2))
ax.set_xlim(-2, 2.2)
ax.set_ylim(-1, 3)
ax.set_zlim(-1.4, 1.6)
ax.set_title("(a) Three-sphere meet: $S_+\\wedge S_-$", pad=0)

ax = fig.add_subplot(1, 2, 2)
circ_c, circ_r, _ = trip.info["circle"]
ax.add_patch(Circle(circ_c[[0, 2]], circ_r, fill=False, color=BLUE, lw=1, ls="--",
                    label=r"$K_4=\mathcal{S}(S_+,r_4)\cap\pi_{\mathrm{ground}}$"))
ax.add_patch(Polygon(base_xz, closed=True, fc="#F1D4D4", ec=RED, lw=1, alpha=0.6,
                     label=f"tripod support (margin ${m_tri:.3f}$)"))
hull = convex_hull_2d(np.vstack([base_xz, Q_out[[0, 2]]]))
ax.add_patch(Polygon(hull, closed=True, fill=False, ec=TEAL, lw=1.4,
                     label=f"with $Q_{{\\mathrm{{out}}}}$ (margin ${m_out:+.3f}$)"))
hull_in = convex_hull_2d(np.vstack([base_xz, Q_in[[0, 2]]]))
ax.add_patch(Polygon(hull_in, closed=True, fill=False, ec=MUTED, lw=0.9, ls=":",
                     label=f"with $Q_{{\\mathrm{{in}}}}$ (margin ${m_in:+.3f}$)"))
tline = np.array([A3[[0, 2]], A3[[0, 2]] + 1.35 * (com - A3[[0, 2]])])
ax.plot(tline[:, 0], tline[:, 1], color=PURPLE, lw=0.8, ls="-.", label=r"helper plane $\tau$ (trace)")
for p, t, off in ((A3, "$A$", (-0.25, -0.15)), (B3, "$B$", (0.06, -0.2)), (C3, "$C$", (-0.25, 0.05))):
    ax.plot(*p[[0, 2]], "o", color=INK, ms=4)
    ax.annotate(t, p[[0, 2]], xytext=(p[0] + off[0], p[2] + off[1]))
ax.plot(*com, "*", color=RED, ms=9, zorder=6)
ax.annotate(r"$\mathrm{proj}(S_+)$", com, xytext=(com[0] + 0.08, com[1] - 0.28), color=RED, fontsize=8)
ax.plot(*Q_out[[0, 2]], "o", color=TEAL, ms=5)
ax.annotate(r"$Q_{\mathrm{out}}$", Q_out[[0, 2]], xytext=(Q_out[0] + 0.06, Q_out[2] + 0.08), color=TEAL)
ax.plot(*Q_in[[0, 2]], "o", mfc="white", mec=MUTED, ms=5)
ax.annotate(r"$Q_{\mathrm{in}}$", Q_in[[0, 2]], xytext=(Q_in[0] - 0.33, Q_in[2] - 0.22), color=MUTED)
ax.set_aspect("equal")
ax.set_xlim(-1.4, 2.35)
ax.set_ylim(-0.45, 2.95)
ax.set_xlabel("$x$")
ax.set_ylabel("$z$")
ax.set_title("(b) Ground view: support polygons")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.3)
fig.subplots_adjust(left=0.0, right=0.99, top=0.93, bottom=0.2, wspace=0.12)
save(fig, "tripod_construction")

# ---- Figure 4: kinematic constructions -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.15))
ax = axs[0]
O, T = np.zeros(2), np.array([1.2, 0.55])
tl = two_link(O, T)
ax.add_patch(Circle(O, 1, fill=False, ls=":", color=BLUE, lw=1.1, label=r"$\mathcal{S}(O,\ell_1)$"))
ax.add_patch(Circle(T, 1, fill=False, ls=":", color=TEAL, lw=1.1, label=r"$\mathcal{S}(T,\ell_2)$"))
Ep, Em = sorted(tl.candidates, key=lambda e: -e[1])
ax.plot(*np.array([O, Ep, T]).T, color=RED, lw=2, label="selected (upper)")
ax.plot(*np.array([O, Em, T]).T, color=MUTED, lw=1, ls="--", label="other branch")
ax.plot(*np.array([Ep, Em]).T, color=PURPLE, lw=0.7, ls="-.", label=r"chord of $E_+\wedge E_-$")
for p, t, off, col in ((O, "$O$", (-0.18, -0.2), INK), (T, "$T$", (0.08, -0.05), INK),
                       (Ep, "$E_+$", (-0.25, 0.08), RED), (Em, "$E_-$", (-0.33, -0.34), MUTED)):
    ax.plot(*p, "o", color=col, ms=4.5, zorder=5)
    label(ax, p, t, *off, color=col)
ax.set_xlim(-1.25, 3.45)
ax.set_ylim(-1.2, 1.75)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("(a) Two-link arm: circle-circle meet")
ax.legend(loc="lower right", fontsize=6.5)

ax = axs[1]
T = np.array([2.2, 0.8])                 # d = 2.34: keeps W-, the mirror and bis(O,W-) apart
dT = np.linalg.norm(T)
u = T / dT
tk = three_link_trapezoid(O, T)
E1, E2 = tk.selected
Wp, Wm = T + u, T - u
ax.add_patch(Circle(O, 1, fill=False, ls=":", color=BLUE, lw=1))
ax.add_patch(Circle(T, 1, fill=False, ls=":", color=TEAL, lw=1))
ax.plot([O[0] - 0.8 * u[0], Wp[0] + 0.3 * u[0]], [O[1] - 0.8 * u[1], Wp[1] + 0.3 * u[1]],
        color=GRID, lw=1)
t_ = np.array([-u[1], u[0]])
for axial, col, ls, lab in (((dT + 1) / 2, ORANGE, "--", r"$\mathrm{bis}(O,W_+)\ni E_2$"),
                            (dT / 2, PURPLE, "--", r"mirror $\mathrm{bis}(O,T)$"),
                            ((dT - 1) / 2, MUTED, ":", r"$\mathrm{bis}(O,W_-)\ni E_1$ (original)")):
    mid = axial * u
    seg = np.array([mid - 1.25 * t_, mid + 1.35 * t_])
    ax.plot(*seg.T, color=col, ls=ls, lw=0.9, label=lab)
ax.plot(*np.array([O, E1, E2, T]).T, color=RED, lw=2, label="chain $O E_1 E_2 T$")
for p, t, off, col in ((O, "$O$", (-0.22, -0.24), INK), (T, "$T$", (0.1, -0.1), INK),
                       (E1, "$E_1$", (-0.36, 0.06), RED), (E2, "$E_2$", (-0.12, 0.14), RED),
                       (Wp, "$W_+$", (0.04, -0.3), ORANGE), (Wm, "$W_-$", (0.05, -0.34), MUTED)):
    ax.plot(*p, "o", color=col, ms=4.2, zorder=5)
    label(ax, p, t, *off, color=col)
ax.set_xlim(-1.1, 4.1)
ax.set_ylim(-1.1, 2.1)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_title("(b) Three-link arm: isosceles trapezoid")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3)
fig.tight_layout()
save(fig, "kinematic_constructions")

# ---- Figure 5: conditioning ---------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.55))
ax = axs[0]
ax.loglog(hs, kappas, "o-", color=BLUE, ms=2.8, lw=1, label=r"$\kappa_2(M)$")
ax.loglog(hs, cerr, "s-", color=RED, ms=2.8, lw=1, label="median rel. error")
ax.axvline(1e-10, color=MUTED, ls=":", lw=0.8)
ax.text(1.4e-10, 1e-6, "noise\nlevel", fontsize=6.5, color=MUTED)
ax.invert_xaxis()
ax.set_xlabel("triangle altitude $h$")
ax.set_ylabel(r"$\kappa_2(M)$ / relative error")
ax.set_title("(a) Circumcentre, $h\\to0$")
ax.legend(loc="upper left", fontsize=6.3)
ax = axs[1]
eps_d = np.logspace(-12, -0.3, 200)
dd = 2 - eps_d
hh = np.sqrt(1 - (dd / 2) ** 2)
ax.loglog(eps_d, dd / (4 * hh), color=RED, lw=1.2, label=r"$|\mathrm{d}h/\mathrm{d}d|$")
ax.loglog(eps_d, 2 * hh, color=TEAL, lw=1.2, label="branch separation $2h$")
ax.invert_xaxis()
ax.set_xlabel("distance to full extension $2-d$")
ax.set_ylabel("sensitivity / separation")
ax.set_title("(b) Two-link arm, $d\\to2$")
ax.legend(loc="center left", fontsize=6.3)
ax = axs[2]
ax.loglog(deltas, np.maximum(err_orig, 1e-17), "o-", color=RED, ms=2.8, lw=1, label="homework (bisector of $O,W_-$)")
ax.loglog(deltas, np.maximum(err_rev, 1e-17), "s-", color=TEAL, ms=2.8, lw=1, label="revised (reflection)")
ax.loglog(deltas, 2.2e-16 / deltas, color=MUTED, ls=":", lw=0.8, label=r"$\varepsilon_{\mathrm{mach}}/|d-1|$")
ax.invert_xaxis()
ax.set_ylim(1e-17, 1)
ax.set_xlabel("$|d-1|$")
ax.set_ylabel("joint position error")
ax.set_title("(c) Three-link arm, $d\\to1$")
ax.legend(loc="upper left", fontsize=6)
for a in axs:
    a.grid(True, which="major")
fig.tight_layout(w_pad=0.6)
save(fig, "stability_analysis")


# =========================================================================
# LaTeX macros
# =========================================================================
def sci(x, digits=2):
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.{digits}f}\\times 10^{{{e}}}$"


def num(n):
    return f"{n:,}"


def vec(v, k=4):
    return "(" + r",\,".join(f"{x:.{k}f}".replace("-0.0000", "0.0000") for x in v) + ")"


r = results
lines = ["% Generated by src/generate_figures.py -- do not edit by hand.",
         f"\\newcommand{{\\ExperimentSeed}}{{{SEED}}}",
         f"\\newcommand{{\\ProjectiveTrials}}{{{num(r['projective']['n'])}}}",
         f"\\newcommand{{\\TwoLinkTrials}}{{{num(r['two_link']['n'])}}}",
         f"\\newcommand{{\\ThreeLinkTrials}}{{{num(r['three_link']['n'])}}}",
         f"\\newcommand{{\\IsoTrials}}{{{num(r['isosceles']['n'])}}}",
         f"\\newcommand{{\\IsoMismatch}}{{{r['isosceles']['mismatch']}}}",
         f"\\newcommand{{\\IsoMaximum}}{{{sci(r['isosceles']['max'])}}}",
         f"\\newcommand{{\\ProjectiveMedian}}{{{sci(r['projective']['median'])}}}",
         f"\\newcommand{{\\ProjectiveMaximum}}{{{sci(r['projective']['max'])}}}",
         f"\\newcommand{{\\TwoLinkMedian}}{{{sci(r['two_link']['median'])}}}",
         f"\\newcommand{{\\TwoLinkMaximum}}{{{sci(r['two_link']['max'])}}}",
         f"\\newcommand{{\\ThreeLinkMedian}}{{{sci(r['three_link']['median'])}}}",
         f"\\newcommand{{\\ThreeLinkMaximum}}{{{sci(r['three_link']['max'])}}}",
         f"\\newcommand{{\\TripodMaximum}}{{{sci(r['tripod']['residual'])}}}",
         f"\\newcommand{{\\ApexCoordinates}}{{{vec(r['tripod']['S_plus'])}}}",
         f"\\newcommand{{\\FourthFootCoordinates}}{{{vec(r['tripod']['Q_out'])}}}",
         f"\\newcommand{{\\InnerFootCoordinates}}{{{vec(r['tripod']['Q_in'])}}}",
         f"\\newcommand{{\\MarginTripod}}{{{r['tripod']['margin_tripod']:.3f}}}",
         f"\\newcommand{{\\MarginOut}}{{{r['tripod']['margin_out']:.3f}}}",
         f"\\newcommand{{\\MarginIn}}{{{r['tripod']['margin_in']:.3f}}}",
         f"\\newcommand{{\\CGATrials}}{{{num(r['cga']['n'])}}}",
         f"\\newcommand{{\\CGATwoOrig}}{{{sci(r['cga']['two_orig'])}}}",
         f"\\newcommand{{\\CGATwoRev}}{{{sci(r['cga']['two_rev'])}}}",
         f"\\newcommand{{\\CGAThreeOrig}}{{{sci(r['cga']['three_orig'])}}}",
         f"\\newcommand{{\\CGAThreeOrigMedian}}{{{sci(r['cga']['three_orig_median'])}}}",
         f"\\newcommand{{\\CGAThreeRev}}{{{sci(r['cga']['three_rev'])}}}",
         f"\\newcommand{{\\CGAThreeBad}}{{{r['cga']['three_n_bad']}}}",
         f"\\newcommand{{\\CGAThreeSwapFail}}{{{100 * r['cga']['three_swap_fail']:.1f}}}",
         f"\\newcommand{{\\CGATripodOrig}}{{{sci(r['cga']['tripod_orig'])}}}",
         f"\\newcommand{{\\CGATripodRev}}{{{sci(r['cga']['tripod_rev'])}}}",
         f"\\newcommand{{\\ThreeSingOrig}}{{{sci(r['three_link_singularity']['orig_at_1e-7'])}}}",
         f"\\newcommand{{\\ThreeSingRevMax}}{{{sci(r['three_link_singularity']['rev_max'])}}}",
         ]
for k, row in zip("ABCDE", r["conditioning"]):
    e = int(round(np.log10(row["h"])))
    lines += [f"\\newcommand{{\\ConditionHeight{k}}}{{$10^{{{e}}}$}}",
              f"\\newcommand{{\\ConditionKappa{k}}}{{{sci(row['kappa'], 3)}}}",
              f"\\newcommand{{\\ConditionError{k}}}{{{sci(row['err'], 3)}}}"]
(GEN / "results.tex").write_text("\n".join(lines) + "\n")
(GEN / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({k: v for k, v in results.items() if k != "three_link_singularity"}, indent=1, default=float))
