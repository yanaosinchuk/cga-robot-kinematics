"""
Aufgabe 1 (revised) -- circumcircle as the meet of two perpendicular bisectors.

Construction (homogeneous coordinates, libcfcg):
    m_ij   = midpoint of P_i P_j                      (finite point, w = 1)
    v_ij   = (-(y_j - y_i), x_j - x_i, 0)              (ideal point = perpendicular direction)
    b_ij   = m_ij x v_ij                              (perpendicular bisector)
    c      = b_01 x b_12                              (circumcentre, homogeneous)

Changes w.r.t. the submitted version
  * scale-aware collinearity test (relative to the edge lengths) instead of an
    absolute threshold 1e-3, which depended on the units of the DAT file
  * the homogeneous centre is classified (finite / ideal) before dehomogenising
  * a residual  max_i | |C - P_i| - r | / r  is printed as a post-condition
  * the drawing window is sized from the circumcircle, not from fixed offsets
"""
import math
import sys

from libcfcg import cf

REL_TOL = 1e-12          # relative tolerance for orientation / ideal-point tests


def read_points(dat_filename):
    """Read exactly three affine points (w = 1) from a DAT file."""
    data = cf.readDatFilePointVector(dat_filename)
    if data.size() != 3:
        raise ValueError(f"DAT file must contain exactly 3 points, found {data.size()}")
    pts = []
    for i in range(data.size()):
        pv = data.get(i)
        if abs(pv.getW() - 1.0) > 1e-9:
            raise ValueError("expected homogeneous coordinate w = 1 in the DAT file")
        pts.append((pv.getX(), pv.getY()))
    return pts


def is_collinear(p0, p1, p2, rel_tol=REL_TOL):
    """|2*area| <= rel_tol * |P1-P0| * |P2-P0|   (invariant under scaling of the data)."""
    area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    scale = math.dist(p0, p1) * math.dist(p0, p2)
    return scale == 0.0 or abs(area2) <= rel_tol * scale


def perpendicular_bisector(pi, pj):
    """Join of the midpoint with the ideal point of the perpendicular direction."""
    mid = cf.PointVector(0.5 * (pi[0] + pj[0]), 0.5 * (pi[1] + pj[1]))
    perp = cf.DirectionVector(-(pj[1] - pi[1]), pj[0] - pi[0], 0)
    return mid.crossProduct(perp)


def main():
    dat_file = sys.argv[1] if len(sys.argv) > 1 else "geometry_files/UMKREIS3.DAT"
    print(f"Reading points from '{dat_file}' ...")
    p0, p1, p2 = read_points(dat_file)

    if is_collinear(p0, p1, p2):
        print("The points are collinear: the bisectors are parallel, their meet is an")
        print("ideal point and no finite circumcircle exists.")
        return

    # --- construct ----------------------------------------------------------
    bis01 = perpendicular_bisector(p0, p1)
    bis12 = perpendicular_bisector(p1, p2)
    centre_h = bis01.crossProduct(bis12)

    # --- classify before dehomogenising ------------------------------------
    norm_h = math.sqrt(centre_h.getX() ** 2 + centre_h.getY() ** 2 + centre_h.getW() ** 2)
    if abs(centre_h.getW()) <= REL_TOL * norm_h:
        print("Centre is (numerically) an ideal point -- input nearly collinear.")
        return
    centre_h.normalize()                                  # w = 1
    cx, cy = centre_h.getX(), centre_h.getY()
    radius = math.dist((cx, cy), p0)

    # --- validate -----------------------------------------------------------
    residual = max(abs(math.dist((cx, cy), p) - radius) for p in (p0, p1, p2)) / radius
    print(f"Circumcentre C = ({cx:.6f}, {cy:.6f}),  radius r = {radius:.6f}")
    print(f"Residual max_i | |C-P_i| - r | / r = {residual:.3e}")

    # --- draw ---------------------------------------------------------------
    margin = 0.15 * radius
    window = cf.WindowCoordinateSystem(
        800,
        cf.Interval(cx - radius - margin, cx + radius + margin),
        cf.Interval(cy - radius - margin, cy + radius + margin),
        "Circumcircle as a projective meet")
    window.setWindowDisplayScale(1.0)
    window.drawAxis(cf.Color.BLACK, 10, 10)

    pts = [cf.Point(x, y) for (x, y) in (p0, p1, p2)]
    for p in pts:
        window.drawPoint(p)
    for i in range(3):
        window.drawLine(pts[i], pts[(i + 1) % 3])
    window.show()
    window.waitKey()

    window.drawLinearEquation(bis01, cf.Color.GREEN, cf.Window2D.LineType_DOT_DASH_0)
    window.drawLinearEquation(bis12, cf.Color.RED, cf.Window2D.LineType_DOT_1)
    window.show()
    window.waitKey()

    centre = cf.Point(cx, cy)
    window.drawPoint(centre, cf.Color.MAGENTA)
    window.drawCircle(centre, radius, cf.Color.BLUE, 2)
    window.show()
    print("Finished. Press any key to exit.")
    window.waitKey()


if __name__ == "__main__":
    main()
