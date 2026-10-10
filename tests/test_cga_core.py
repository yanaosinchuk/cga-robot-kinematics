import numpy as np

import cga


def test_basis_signature_matches_g41_metric():
    assert (cga.e1 * cga.e1).scalar() == 1.0
    assert (cga.e2 * cga.e2).scalar() == 1.0
    assert (cga.e3 * cga.e3).scalar() == 1.0
    assert (cga.ep * cga.ep).scalar() == 1.0
    assert (cga.em * cga.em).scalar() == -1.0


def test_null_basis_relations():
    assert abs((cga.einf * cga.einf).scalar()) < 1e-14
    assert abs((cga.e0 * cga.e0).scalar()) < 1e-14
    assert abs((cga.e0 | cga.einf).scalar() + 1.0) < 1e-14


def test_outer_product_is_antisymmetric_for_vectors():
    lhs = cga.e1 ^ cga.e2
    rhs = cga.e2 ^ cga.e1
    np.testing.assert_allclose(lhs.c, -rhs.c, atol=0.0, rtol=0.0)


def test_basis_vector_inverse():
    np.testing.assert_allclose((cga.e1 * cga.e1.inverse()).c, cga.MV.blade(0).c)
    np.testing.assert_allclose((cga.em * cga.em.inverse()).c, cga.MV.blade(0).c)


def test_dual_twice_returns_negative_multivector_in_g41():
    A = 2.0 * cga.e1 + 3.0 * (cga.e2 ^ cga.e3)
    np.testing.assert_allclose(A.dual().dual().c, -A.c, atol=1e-14, rtol=0.0)


def test_conformal_inner_product_encodes_squared_distance():
    x = np.array([1.2, -0.5, 2.0])
    y = np.array([-0.7, 1.5, 0.25])
    X = cga.VecN3(*x)
    Y = cga.VecN3(*y)

    cga_distance_squared = -2.0 * (X | Y).scalar()
    euclidean_distance_squared = float(np.sum((x - y) ** 2))

    assert abs(cga_distance_squared - euclidean_distance_squared) < 1e-12


def test_ipns_sphere_incidence():
    center = cga.VecN3(1.0, -2.0, 0.5)
    radius = 1.7
    sphere = cga.SphereN3_ipns(center, radius)

    point_on_sphere = cga.VecN3(1.0 + radius, -2.0, 0.5)
    point_inside = cga.VecN3(1.0, -2.0, 0.5)

    assert abs((point_on_sphere | sphere).scalar()) < 1e-12
    assert (point_inside | sphere).scalar() > 0.0


def test_normalise_point_is_scale_invariant():
    X = cga.VecN3(-1.0, 2.0, 3.0)
    scaled = 7.25 * X

    np.testing.assert_allclose(
        cga.normalise_point(scaled).c,
        cga.normalise_point(X).c,
        atol=1e-12,
        rtol=0.0,
    )


def test_pointpair_extraction_recovers_two_sphere_intersections():
    origin = cga.VecN3(0.0, 0.0, 0.0)
    target = cga.VecN3(1.0, 0.0, 0.0)

    sphere_a = cga.SphereN3(origin, 1.0)
    sphere_b = cga.SphereN3(target, 1.0)
    helper_plane = origin ^ cga.VecN3(0.0, 0.0, 1.0) ^ target ^ cga.einf

    pair = cga.meet(cga.meet(sphere_a, sphere_b), helper_plane)
    status, points = cga.pointpair_points(pair)

    assert status == "real"
    assert len(points) == 2

    for point in points:
        assert abs(np.linalg.norm(point) - 1.0) < 1e-12
        assert abs(np.linalg.norm(point - np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-12
        assert abs(point[0] - 0.5) < 1e-12
        assert abs(point[1]) < 1e-12

def test_multivector_rejects_wrong_coefficient_shape():
    with np.testing.assert_raises(ValueError):
        cga.MV(np.zeros(31))


def test_null_multivector_is_not_invertible():
    with np.testing.assert_raises(ZeroDivisionError):
        cga.MV().inverse()


def test_normalise_point_rejects_zero_weight():
    plane_like = cga.e1 + cga.e2
    with np.testing.assert_raises(ZeroDivisionError):
        cga.normalise_point(plane_like)


def test_zero_pointpair_is_classified_as_degenerate():
    zero_pair = cga.MV()
    assert cga.classify_pointpair(zero_pair) == "degenerate"
    status, points = cga.pointpair_points(zero_pair)
    assert status == "degenerate"
    assert points == []


def test_tangent_pointpair_extracts_one_unique_point():
    origin = cga.VecN3(0.0, 0.0, 0.0)
    target = cga.VecN3(2.0, 0.0, 0.0)
    helper_plane = origin ^ cga.VecN3(0.0, 0.0, 1.0) ^ target ^ cga.einf
    pair = cga.meet(
        cga.meet(cga.SphereN3(origin, 1.0), cga.SphereN3(target, 1.0)),
        helper_plane,
    )

    status, points = cga.pointpair_points(pair)

    assert status == "tangent"
    assert len(points) == 1
    np.testing.assert_allclose(points[0], [1.0, 0.0, 0.0], atol=1e-12)

def test_geometric_product_is_associative_for_random_multivectors():
    rng = np.random.default_rng(42)
    for _ in range(20):
        A = cga.MV(rng.normal(size=cga.DIM))
        B = cga.MV(rng.normal(size=cga.DIM))
        C = cga.MV(rng.normal(size=cga.DIM))
        lhs = (A * B) * C
        rhs = A * (B * C)
        np.testing.assert_allclose(lhs.c, rhs.c, atol=2e-13, rtol=2e-13)

def test_conformal_point_conversion_is_invariant_to_small_nonzero_scale():
    point = cga.VecN3(1.25, -0.75, 2.0)
    scaled = 1e-15 * point

    np.testing.assert_allclose(cga.euclid(scaled), [1.25, -0.75, 2.0], atol=1e-12)
    np.testing.assert_allclose(
        cga.normalise_point(scaled).c,
        cga.normalise_point(point).c,
        atol=1e-12,
        rtol=1e-12,
    )


def test_inverse_is_scale_invariant_for_non_null_blade():
    blade = 1e-12 * (cga.e1 ^ cga.e2)
    identity = blade * blade.inverse()
    np.testing.assert_allclose(identity.c, cga.MV.blade(0).c, atol=1e-12)

def test_pointpair_extraction_is_set_invariant_under_global_sign():
    x = np.array([-0.4, 0.8, 1.2])
    y = np.array([1.1, -0.2, 0.3])
    pair = cga.VecN3(*x) ^ cga.VecN3(*y)

    _, _, p1, p2 = cga.pointpair_extract_homework(pair)
    _, _, q1, q2 = cga.pointpair_extract_homework(-pair)

    extracted = sorted((tuple(np.round(cga.euclid(p1), 12)),
                        tuple(np.round(cga.euclid(p2), 12))))
    flipped = sorted((tuple(np.round(cga.euclid(q1), 12)),
                      tuple(np.round(cga.euclid(q2), 12))))
    expected = sorted((tuple(np.round(x, 12)), tuple(np.round(y, 12))))

    assert extracted == expected
    assert flipped == expected
    np.testing.assert_allclose(cga.euclid(p1), cga.euclid(q2), atol=1e-12)
    np.testing.assert_allclose(cga.euclid(p2), cga.euclid(q1), atol=1e-12)

