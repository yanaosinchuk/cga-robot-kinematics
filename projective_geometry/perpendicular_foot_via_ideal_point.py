"""
The foot of a perpendicular constructed through an ideal point.

Construction (homogeneous coordinates, libcfcg):
    L   = p0 x p1                        base line (a, b, c)
    I   = (a, b, 0)                      ideal point of the normal direction of L
    n0  = p0 x I,  n1 = p1 x I           two perpendiculars (parallel lines!)
    S1  = n0 x n1                        their meet: an ideal point, S1 ~ I
    n2  = S1 x p2                        perpendicular through P2
    F   = n2 x L                         foot of the perpendicular

"""
import math
import sys

from libcfcg import cf

REL_TOL = 1e-12


def read_points(dat_filename):
    data = cf.readDatFilePointVector(dat_filename)
    if data.size() != 3:
        raise ValueError(f"DAT file must contain exactly 3 points, found {data.size()}")
    pv = []
    for i in range(data.size()):
        p = data.get(i)
        if abs(p.getW() - 1.0) > 1e-9:
            raise ValueError("expected homogeneous coordinate w = 1 in the DAT file")
        pv.append(cf.PointVector(p.getX(), p.getY()))
    return pv


def hnorm(v):
    return math.sqrt(v.getX() ** 2 + v.getY() ** 2 + v.getW() ** 2)


def main():
    dat_file = sys.argv[1] if len(sys.argv) > 1 else "geometry_files/UMKREIS4.DAT"
    print(f"Reading points from '{dat_file}' ...")
    p0, p1, p2 = read_points(dat_file)
    if math.dist((p0.getX(), p0.getY()), (p1.getX(), p1.getY())) == 0.0:
        raise ValueError("P0 and P1 coincide: the base line is undefined")

    xs = [p.getX() for p in (p0, p1, p2)]
    ys = [p.getY() for p in (p0, p1, p2)]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    window = cf.WindowCoordinateSystem(
        800,
        cf.Interval(min(xs) - 0.3 * span, max(xs) + 0.3 * span),
        cf.Interval(min(ys) - 0.3 * span, max(ys) + 0.3 * span),
        "Perpendicular foot through an ideal point")
    window.setWindowDisplayScale(1.0)
    window.drawAxis(cf.Color.BLACK, 10, 10)
    for p in (p0, p1, p2):
        window.drawPoint(cf.Point(p.getX(), p.getY()))
    window.show()
    window.waitKey()

    # 1) base line
    L = p0.crossProduct(p1)
    window.drawLinearEquation(L, cf.Color.BLACK, cf.Window2D.LineType_DEFAULT)
    window.show()
    window.waitKey()

    # 2) perpendiculars at P0 and P1: joins with the ideal point of the normal of L
    I_perp = cf.DirectionVector(L.getX(), L.getY(), 0)
    perp0 = p0.crossProduct(I_perp)
    perp1 = p1.crossProduct(I_perp)
    window.drawLinearEquation(perp0, cf.Color.GREEN, cf.Window2D.LineType_DOT_DASH_0)
    window.drawLinearEquation(perp1, cf.Color.RED, cf.Window2D.LineType_DOT_1)
    window.show()
    window.waitKey()

    # 3) the two parallel perpendiculars meet in an ideal point
    S1 = perp0.crossProduct(perp1)
    print(f"S1 = ({S1.getX():.6g}, {S1.getY():.6g}, {S1.getW():.3g})")
    if abs(S1.getW()) > REL_TOL * hnorm(S1):
        print("Warning: S1 is not ideal -- the perpendiculars are not parallel?")
    else:
        print("S1 is an ideal point (w = 0): parallel lines meet at infinity.")

    # 4) perpendicular through P2 = join of P2 with the ideal point S1
    n2 = S1.crossProduct(p2)
    window.drawLinearEquation(n2, cf.Color.MAGENTA, cf.Window2D.LineType_DASH_1)
    window.show()
    window.waitKey()

    # 5) foot = meet of n2 with the base line
    F = n2.crossProduct(L)
    F.normalize()
    fx, fy = F.getX(), F.getY()

    # post-condition: F on L and (F - P2) orthogonal to (P1 - P0)
    dx, dy = p1.getX() - p0.getX(), p1.getY() - p0.getY()
    ex, ey = fx - p2.getX(), fy - p2.getY()
    on_line = abs(L.getX() * fx + L.getY() * fy + L.getW()) / math.hypot(L.getX(), L.getY())
    ortho = abs(dx * ex + dy * ey) / (math.hypot(dx, dy) * max(math.hypot(ex, ey), 1.0))
    print(f"Foot F = ({fx:.6f}, {fy:.6f})")
    print(f"Residuals: dist(F, L) = {on_line:.3e},  |cos angle(P2F, L)| = {ortho:.3e}")

    window.drawPoint(cf.Point(fx, fy), cf.Color.BLUE)
    window.show()
    print("Done. Press any key to exit.")
    window.waitKey()


if __name__ == "__main__":
    main()
