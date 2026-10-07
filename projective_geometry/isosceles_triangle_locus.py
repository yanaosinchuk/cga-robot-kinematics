"""
All apices Q on the parallel L|| to P1P2 through P0 such that
the triangle P1 P2 Q is isosceles.

Three loci, one line:
    (a) |QP1| = |QP2|  : perpendicular bisector of P1P2     -> always exactly 1 point
    (b) |QP1| = d      : circle S(P1, d),  d = |P1P2|         -> 2 / 1 (tangent) / 0 points
    (c) |QP2| = d      : circle S(P2, d)                      -> 2 / 1 (tangent) / 0 points
With h = signed distance of L|| from the base line:
    |h| < d, |h| != sqrt(3)/2 d : 5 distinct apices
    |h| = sqrt(3)/2 d           : 3  (equilateral: the bisector point lies on both circles)
    |h| = d                     : 3  (both circles tangent to L||)
    |h| > d                     : 1
    h = 0                       : degenerate (every triangle collapses)
    
"""
import math
import sys

from libcfcg import cf

REL_TOL = 1e-12


def read_points(dat_filename):
    data = cf.readDatFilePointVector(dat_filename)
    if data.size() != 3:
        raise ValueError(f"DAT file must contain exactly 3 points, found {data.size()}")
    pts = []
    for i in range(data.size()):
        p = data.get(i)
        if abs(p.getW() - 1.0) > 1e-9:
            raise ValueError("expected homogeneous coordinate w = 1 in the DAT file")
        pts.append((p.getX(), p.getY()))
    return pts


def isosceles_apices(p0, p1, p2):
    """Return (status, list of (label, (x, y))) for the apices on L||."""
    ux, uy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(ux, uy)
    if d == 0.0:
        raise ValueError("P1 and P2 coincide: no base")
    ux, uy = ux / d, uy / d
    nx, ny = -uy, ux                                      # unit normal of the base
    h = (p0[0] - p1[0]) * nx + (p0[1] - p1[1]) * ny       # signed height of L||
    if abs(h) <= REL_TOL * d:
        return "degenerate: P0 lies on the base line", []

    # parameter t along L||:  Q(t) = P1 + t*u + h*n
    cands = [("bisector", 0.5 * d)]
    disc = d * d - h * h
    for label, t0 in (("circle P1", 0.0), ("circle P2", d)):
        if disc > REL_TOL * d * d:
            s = math.sqrt(disc)
            cands += [(label, t0 - s), (label, t0 + s)]
        elif disc >= -REL_TOL * d * d:
            cands.append((label + " (tangent)", t0))
    cands.sort(key=lambda c: c[1])
    merged = []
    for label, t in cands:                                # merge coincident apices
        if merged and abs(t - merged[-1][1]) <= 1e-9 * d:
            merged[-1] = (merged[-1][0] + " = " + label, merged[-1][1])
        else:
            merged.append((label, t))
    apices = [(lab, (p1[0] + t * ux + h * nx, p1[1] + t * uy + h * ny)) for lab, t in merged]
    return f"{len(apices)} distinct apices (|h|/d = {abs(h) / d:.4f})", apices


def isosceles_residual(q, p1, p2):
    a, b, d = math.dist(q, p1), math.dist(q, p2), math.dist(p1, p2)
    return min(abs(a - b), abs(a - d), abs(b - d)) / d


def main():
    dat_file = sys.argv[1] if len(sys.argv) > 1 else "geometry_files/UMKREIS1.DAT"
    print(f"Reading '{dat_file}' ...")
    p0, p1, p2 = read_points(dat_file)
    status, apices = isosceles_apices(p0, p1, p2)
    print(status)
    for k, (label, q) in enumerate(apices, 1):
        print(f"  Q{k} = ({q[0]:.6f}, {q[1]:.6f})   locus: {label:<28s} "
              f"residual = {isosceles_residual(q, p1, p2):.2e}")

    # ------------------------------------------------------------ drawing
    xs = [p[0] for p in (p0, p1, p2)] + [q[0] for _, q in apices]
    ys = [p[1] for p in (p0, p1, p2)] + [q[1] for _, q in apices]
    d = math.dist(p1, p2)
    window = cf.WindowCoordinateSystem(
        700,
        cf.Interval(min(xs) - 0.4 * d, max(xs) + 0.4 * d),
        cf.Interval(min(ys) - 0.4 * d, max(ys) + 0.4 * d),
        "Isosceles apices on the parallel")
    window.drawAxis(cf.Color.BLACK, 10, 10)

    P0, P1, P2 = (cf.PointVector(x, y) for (x, y) in (p0, p1, p2))
    for P in (P0, P1, P2):
        window.drawPoint(cf.Point(P.getX(), P.getY()))
    L12 = P1.crossProduct(P2)                                  # base line
    Lpar = P0.crossProduct(cf.DirectionVector(P2.sub(P1)))     # parallel through P0
    mid = cf.PointVector(0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
    bis = mid.crossProduct(cf.DirectionVector(-(p2[1] - p1[1]), p2[0] - p1[0], 0))
    window.drawLinearEquation(L12, cf.Color.BLACK)
    window.drawLinearEquation(Lpar, cf.Color.GREEN)
    window.drawLinearEquation(bis, cf.Color.MAGENTA)
    window.drawCircle(cf.Point(*p1), d, cf.Color.RED, 1)
    window.drawCircle(cf.Point(*p2), d, cf.Color.CYAN, 1)
    window.show()
    window.waitKey()

    colour = {"bisector": cf.Color.MAGENTA, "circle P1": cf.Color.RED, "circle P2": cf.Color.CYAN}
    for label, q in apices:
        key = label.split(" (")[0].split(" =")[0]
        window.drawPoint(cf.Point(q[0], q[1]), colour[key])
        window.drawLine(cf.Point(q[0], q[1]), cf.Point(*p1))
        window.drawLine(cf.Point(q[0], q[1]), cf.Point(*p2))
    window.show()
    print("All apices drawn. Press any key to exit.")
    window.waitKey()


if __name__ == "__main__":
    main()
