import numpy as np

import cga
import cga_constructions as constructions
import geometry as G


def test_conformal_point_embedding_is_null_and_normalized():
    X = cga.VecN3(1.2, -0.7, 2.5)

    assert abs((X * X).scalar()) < 1e-12
    assert abs((X | cga.einf).scalar() + 1.0) < 1e-12
    np.testing.assert_allclose(cga.euclid(X), [1.2, -0.7, 2.5], atol=1e-12)


def test_circumcircle_residual_is_small():
    result = G.circumcircle(
        np.array([-1.4, 0.1]),
        np.array([1.55, 0.35]),
        np.array([0.1, 2.0]),
    )

    assert result.status == "regular"
    assert result.residual < 1e-13


def test_circumcircle_reports_collinear_input_as_ideal():
    result = G.circumcircle(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([3.0, 3.0]),
    )

    assert result.status == "ideal"


def test_projective_foot_vertical_base():
    result = G.perpendicular_foot(
        np.array([2.0, -1.0]),
        np.array([2.0, 5.0]),
        np.array([7.0, 3.0]),
    )

    np.testing.assert_allclose(result.selected, [2.0, 3.0], atol=1e-14)
    assert result.residual < 1e-13


def test_isosceles_candidate_count_formula():
    rng = np.random.default_rng(0)

    for _ in range(2000):
        p0, p1, p2 = rng.uniform(-5.0, 5.0, (3, 2))
        if np.linalg.norm(p2 - p1) < 1e-8:
            continue

        result = G.isosceles_candidates(p0, p1, p2)
        if result.status == "degenerate":
            continue

        expected = G.isosceles_count(result.info["h"] / result.info["d"])
        assert len(result.candidates) == expected
        assert result.residual < 1e-12


def test_two_link_reference_constraints():
    rng = np.random.default_rng(1)

    for _ in range(500):
        target = rng.normal(size=2)
        target *= rng.uniform(0.05, 1.95) / np.linalg.norm(target)

        result = G.two_link(np.zeros(2), target)

        assert result.status in {"regular", "tangent"}
        assert result.residual < 1e-13


def test_three_link_reference_constraints():
    rng = np.random.default_rng(2)

    for _ in range(500):
        target = rng.normal(size=3)
        target[1] = abs(target[1]) + 0.1
        target *= rng.uniform(0.05, 2.95) / np.linalg.norm(target)

        result = G.three_link_trapezoid(np.zeros(3), target)

        assert result.status in {"regular", "tangent"}
        assert result.residual < 1e-13


def test_tripod_fourth_support_restores_static_stability():
    result = G.tripod(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        2.69,
        1.8,
        1.5,
        1.1,
    )

    base = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    projected_apex = result.selected[[0, 2]]

    assert G.stability_margin(base, projected_apex) < 0.0

    extended_base = np.vstack([base, result.info["Q_out"][[0, 2]]])
    assert G.stability_margin(extended_base, projected_apex) > 0.0
    assert result.residual < 1e-12


def test_revised_two_link_cga_satisfies_link_lengths():
    targets = (
        [0.0, 0.5, -1.0],
        [1.2, 0.3, -0.4],
        [0.0, 1.5, 0.0],
    )

    for target in targets:
        target = np.asarray(target, dtype=float)
        elbow = constructions.two_link_revised(*target)

        assert abs(np.linalg.norm(elbow) - 1.0) < 1e-12
        assert abs(np.linalg.norm(elbow - target) - 1.0) < 1e-12


def test_revised_three_link_cga_handles_d_equal_one():
    direction = np.array([0.6, 0.3, -0.2])
    direction /= np.linalg.norm(direction)

    for distance in (0.3, 1.0, 1.0 + 1e-9, 2.0, 2.9):
        target = distance * direction
        reference = G.three_link_trapezoid(np.zeros(3), target).selected
        e1, e2 = constructions.three_link_revised(*target)

        np.testing.assert_allclose(e1, reference[0], atol=1e-13)
        np.testing.assert_allclose(e2, reference[1], atol=1e-13)


def test_revised_tripod_cga_matches_reference_solution():
    reference = G.tripod(
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        2.69,
        1.8,
        1.5,
        1.1,
    )
    actual = constructions.tripod_revised()

    np.testing.assert_allclose(actual["spitze"], reference.selected, atol=1e-13)
    np.testing.assert_allclose(actual["aussen"], reference.info["Q_out"], atol=1e-13)

def test_isosceles_count_matches_construction_near_special_heights():
    p1 = np.array([-1.0, 0.0])
    p2 = np.array([1.0, 0.0])
    d = np.linalg.norm(p2 - p1)

    ratios = [
        0.0,
        0.5,
        np.sqrt(3.0) / 2.0,
        np.sqrt(3.0) / 2.0 + 5e-10,
        0.999999,
        1.0,
        1.000001,
        1.2,
    ]

    for ratio in ratios:
        p0 = np.array([0.3, ratio * d])
        result = G.isosceles_candidates(p0, p1, p2)
        expected = G.isosceles_count(ratio)

        if result.status == "degenerate":
            assert expected == 0
        else:
            assert len(result.candidates) == expected


def test_original_three_link_is_degenerate_at_d_equal_one_but_revised_is_not():
    direction = np.array([0.6, 0.3, -0.2])
    direction /= np.linalg.norm(direction)
    target = direction

    with np.testing.assert_raises(constructions.Degenerate):
        constructions.three_link_original(*target)

    reference = G.three_link_trapezoid(np.zeros(3), target).selected
    e1, e2 = constructions.three_link_revised(*target)
    np.testing.assert_allclose(e1, reference[0], atol=1e-13)
    np.testing.assert_allclose(e2, reference[1], atol=1e-13)


def test_perpendicular_foot_classifies_coincident_base_points():
    result = G.perpendicular_foot(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    )

    assert result.status == "degenerate"


def test_isosceles_locus_classifies_coincident_base_points():
    result = G.isosceles_candidates(
        np.array([0.0, 1.0]),
        np.array([2.0, 0.0]),
        np.array([2.0, 0.0]),
    )

    assert result.status == "degenerate"


def test_reference_kinematics_classifies_unreachable_targets():
    two = G.two_link(np.zeros(2), np.array([2.1, 0.0]))
    three = G.three_link_trapezoid(np.zeros(2), np.array([3.1, 0.0]))

    assert two.status == "empty"
    assert three.status == "empty"


def test_cga_kinematics_rejects_unreachable_targets():
    with np.testing.assert_raises(constructions.Imaginary):
        constructions.two_link_revised(2.1, 0.0, 0.0)

    with np.testing.assert_raises(constructions.Imaginary):
        constructions.three_link_revised(3.1, 0.0, 0.0)

def test_reference_two_link_vertical_axis_uses_deterministic_fallback():
    target = np.array([0.0, 1.5, 0.0])
    result = G.two_link(np.zeros(3), target)

    assert result.status == "regular"
    assert result.selected[0] > 0.0
    assert abs(result.residual) < 1e-12

    cga_elbow = constructions.two_link_revised(*target)
    np.testing.assert_allclose(cga_elbow, result.selected, atol=1e-12)


def test_reference_three_link_vertical_axis_matches_cga_fallback():
    target = np.array([0.0, 2.0, 0.0])
    result = G.three_link_trapezoid(np.zeros(3), target)

    assert result.status == "regular"
    e1, e2 = constructions.three_link_revised(*target)
    np.testing.assert_allclose(e1, result.selected[0], atol=1e-12)
    np.testing.assert_allclose(e2, result.selected[1], atol=1e-12)


def test_kinematic_tangent_boundaries_are_classified():
    two = G.two_link(np.zeros(2), np.array([2.0, 0.0]))
    three = G.three_link_trapezoid(np.zeros(2), np.array([3.0, 0.0]))

    assert two.status == "tangent"
    assert three.status == "tangent"
    assert two.residual < 1e-12
    assert three.residual < 1e-12


def test_revised_cga_kinematics_reject_target_at_shoulder():
    with np.testing.assert_raises(constructions.Degenerate):
        constructions.two_link_revised(0.0, 0.0, 0.0)

    with np.testing.assert_raises(constructions.Degenerate):
        constructions.three_link_revised(0.0, 0.0, 0.0)


def test_reference_kinematics_reject_nonpositive_link_lengths():
    two = G.two_link(np.zeros(2), np.array([1.0, 0.0]), l1=0.0, l2=1.0)
    three = G.three_link_trapezoid(np.zeros(2), np.array([1.0, 0.0]), l=0.0)

    assert two.status == "degenerate"
    assert three.status == "degenerate"

