"""
Tests for the revised course programs and the reference implementations.

    python3 -m unittest discover -s code/tests -v

* The revised libcfcg scripts are executed headlessly against a stub of the
  library and their printed results are compared with geometry.py.
* The revised CLUCalc constructions are executed in the NumPy CGA engine.
"""
import io
import math
import os
import re
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(HERE.parent))

import geometry as G           # noqa: E402
import homework_cga as hw      # noqa: E402

PY = ROOT / "source_tasks_revised" / "python"
FLOAT = r"[-+]?\d+\.\d+(?:e[-+]?\d+)?"


def run_script(name, points):
    with tempfile.NamedTemporaryFile("w", suffix=".DAT", delete=False) as fh:
        for p in points:
            fh.write(f"{p[0]} {p[1]}\n")
    out = io.StringIO()
    argv = sys.argv
    sys.argv = [name, fh.name]
    try:
        with redirect_stdout(out):
            runpy.run_path(str(PY / name), run_name="__main__")
    finally:
        sys.argv = argv
        os.unlink(fh.name)
    return out.getvalue()


class RevisedPythonPrograms(unittest.TestCase):
    def test_circumcircle(self):
        P = [(12.0, 30.5), (170.0, 42.0), (80.0, 160.0)]
        txt = run_script("aufgabe1_circumcircle.py", P)
        cx, cy, r = map(float, re.search(rf"C = \(({FLOAT}), ({FLOAT})\),\s+radius r = ({FLOAT})", txt).groups())
        ref = G.circumcircle(*P)
        self.assertAlmostEqual(cx, ref.selected[0], places=5)
        self.assertAlmostEqual(cy, ref.selected[1], places=5)
        self.assertAlmostEqual(r, ref.info["radius"], places=5)

    def test_circumcircle_collinear_is_reported(self):
        txt = run_script("aufgabe1_circumcircle.py", [(0, 0), (1e6, 1e6), (3e6, 3e6)])
        self.assertIn("collinear", txt)

    def test_perpendicular_foot(self):
        P = [(-18.0, -3.0), (160.0, 55.0), (15.0, 210.0)]
        txt = run_script("aufgabe2_perpendicular_foot.py", P)
        self.assertIn("ideal point", txt)
        fx, fy = map(float, re.search(rf"Foot F = \(({FLOAT}), ({FLOAT})\)", txt).groups())
        ref = G.perpendicular_foot(*P).selected
        self.assertAlmostEqual(fx, ref[0], places=5)
        self.assertAlmostEqual(fy, ref[1], places=5)

    def test_isosceles_counts(self):
        P1, P2 = (20.0, 10.0), (140.0, 10.0)
        d = 120.0
        for h, expected in ((50.0, 5), (math.sqrt(3) / 2 * d, 3), (d, 3), (130.0, 1)):
            txt = run_script("aufgabe4_isosceles.py", [(5.0, 10.0 + h), P1, P2])
            self.assertIn(f"{expected} distinct apices", txt, msg=f"h={h}")
            res = [float(x) for x in re.findall(rf"residual = ({FLOAT})", txt)]
            self.assertTrue(res and max(res) < 1e-12)

    def test_isosceles_degenerate(self):
        txt = run_script("aufgabe4_isosceles.py", [(0.0, 10.0), (20.0, 10.0), (140.0, 10.0)])
        self.assertIn("degenerate", txt)


class ReferenceGeometry(unittest.TestCase):
    def test_projective_foot_vertical_base(self):
        r = G.perpendicular_foot([2.0, -1.0], [2.0, 5.0], [7.0, 3.0])
        np.testing.assert_allclose(r.selected, [2.0, 3.0], atol=1e-14)

    def test_isosceles_proposition(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            P0, P1, P2 = rng.uniform(-5, 5, (3, 2))
            r = G.isosceles_candidates(P0, P1, P2)
            if r.status == "degenerate":
                continue
            self.assertEqual(len(r.candidates), G.isosceles_count(r.info["h"] / r.info["d"]))

    def test_trapezoid_links(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            t = rng.normal(size=3)
            t *= rng.uniform(0.05, 2.95) / np.linalg.norm(t)
            self.assertLess(G.three_link_trapezoid(np.zeros(3), t).residual, 1e-14)

    def test_tripod_stability(self):
        r = G.tripod([-1, 0, 0], [1, 0, 0], [0, 0, 2], 2.69, 1.8, 1.5, 1.1)
        base = np.array([[-1, 0], [1, 0], [0, 2]], float)
        com = r.selected[[0, 2]]
        self.assertLess(G.stability_margin(base, com), 0)                    # tripod tips over
        self.assertGreater(G.stability_margin(np.vstack([base, r.info["Q_out"][[0, 2]]]), com), 0)


class RevisedCLUCalcConstructions(unittest.TestCase):
    def test_tripod(self):
        ref = G.tripod([-1, 0, 0], [1, 0, 0], [0, 0, 2], 2.69, 1.8, 1.5, 1.1)
        out = hw.tripod_revised()
        np.testing.assert_allclose(out["spitze"], ref.selected, atol=1e-13)
        np.testing.assert_allclose(out["aussen"], ref.info["Q_out"], atol=1e-13)

    def test_two_link(self):
        for t in ([0.0, 0.5, -1.0], [1.2, 0.3, -0.4], [0.0, 1.5, 0.0]):   # last: on the vertical axis
            ref = G.two_link(np.zeros(3), np.array(t),
                             up=np.array([0.0, 1.0, 0.0]) if t[0] or t[2] else np.array([1.0, 0, 0]))
            e = hw.two_link_revised(*t)
            self.assertAlmostEqual(np.linalg.norm(e), 1.0, places=12)
            self.assertAlmostEqual(np.linalg.norm(e - np.array(t)), 1.0, places=12)

    def test_three_link_including_d_equal_1(self):
        u = np.array([0.6, 0.3, -0.2])
        u /= np.linalg.norm(u)
        for d in (0.3, 1.0, 1.0 + 1e-9, 2.0, 2.9):
            t = d * u
            ref = G.three_link_trapezoid(np.zeros(3), t).selected
            e1, e2 = hw.three_link_revised(*t)
            np.testing.assert_allclose(e1, ref[0], atol=1e-13)
            np.testing.assert_allclose(e2, ref[1], atol=1e-13)


if __name__ == "__main__":
    unittest.main()
