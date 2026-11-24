
import numpy as np
import pytest

from trustworthy_ai.models.splines import BSplineBasis


def test_basis_shape():

    X = np.linspace(0, 10, 100)

    spline = BSplineBasis(n_splines=10, degree=3)

    B = spline.fit_transform(X)

    assert B.shape == (100, 10)


def test_partition_of_unity():

    X = np.linspace(0, 10, 100)

    spline = BSplineBasis()

    B = spline.fit_transform(X)

    assert np.allclose(B.sum(axis=1), 1.0)


def test_non_negative_basis():

    X = np.linspace(0, 10, 100)

    spline = BSplineBasis()

    B = spline.fit_transform(X)

    assert np.all(B >= 0)


def test_transform_before_fit():

    spline = BSplineBasis()

    with pytest.raises(RuntimeError):
        spline.transform([1, 2, 3])


def test_invalid_spline_count():

    with pytest.raises(ValueError):
        BSplineBasis(n_splines=3, degree=3)


def test_constant_input():

    spline = BSplineBasis()

    with pytest.raises(ValueError):
        spline.fit([5, 5, 5])


def test_outside_training_range():

    spline = BSplineBasis()

    spline.fit([0, 2, 4, 6, 8, 10])

    with pytest.raises(ValueError):
        spline.transform([12])
