
import numpy as np
import pytest

from trustworthy_ai.models.slim import SLIM


@pytest.fixture
def sample_data():

    X = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 1],
    ])

    y = np.array([
        1, 1, -1, -1,
        1, -1, 1, -1
    ])

    return X, y


def test_model_fits(sample_data):

    X, y = sample_data

    model = SLIM(
        C0=0.01,
        coefficient_bound=5
    )

    model.fit(X, y)

    assert model.is_fitted_


def test_coefficients_are_integers(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    assert np.all(
        model.coef_ == np.round(model.coef_)
    )

    assert isinstance(model.intercept_, int)


def test_coefficient_bounds(sample_data):

    X, y = sample_data

    model = SLIM(
        coefficient_bound=4
    )

    model.fit(X, y)

    assert np.all(
        np.abs(model.coef_) <= 4
    )

    assert abs(model.intercept_) <= 4


def test_prediction_shape(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape


def test_prediction_classes(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    predictions = model.predict(X)

    assert set(np.unique(predictions)).issubset(
        {-1, 1}
    )


def test_sparsity_count(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    assert model.n_features_selected_ == np.count_nonzero(
        model.coef_
    )


def test_scorecard(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    scorecard = model.get_scorecard()

    assert "intercept" in scorecard
    assert "features" in scorecard


def test_explanation(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    explanation = model.explain(
        X,
        sample_index=0
    )

    assert "score" in explanation
    assert "prediction" in explanation
    assert "contributions" in explanation


def test_invalid_X():

    X = np.array([1, 2, 3])
    y = np.array([0, 1, 0])

    model = SLIM()

    with pytest.raises(ValueError):

        model.fit(X, y)


def test_invalid_labels():

    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    y = np.array([0, 1, 2])

    model = SLIM()

    with pytest.raises(ValueError):

        model.fit(X, y)


def test_prediction_before_fit():

    model = SLIM()

    X = np.array([
        [1, 0],
        [0, 1],
    ])

    with pytest.raises(RuntimeError):

        model.predict(X)


def test_feature_names():

    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])

    y = np.array([1, -1, 1, -1])

    model = SLIM(
        feature_names=["Age", "Risk"]
    )

    model.fit(X, y)

    assert model.feature_names_ == [
        "Age",
        "Risk"
    ]


def test_no_intercept():

    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])

    y = np.array([1, -1, 1, -1])

    model = SLIM(
        fit_intercept=False
    )

    model.fit(X, y)

    assert model.intercept_ == 0


def test_predict_proba_not_supported(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    with pytest.raises(NotImplementedError):

        model.predict_proba(X)

def test_coprime_coefficients():

    model = SLIM()

    coefficients = np.array([4, 6, -2])

    normalized = model._normalize_coprime(
        coefficients
    )

    from math import gcd
    from functools import reduce

    divisor = reduce(
        gcd,
        np.abs(normalized).tolist()
    )

    assert divisor == 1


def test_coprime_zero_coefficients():

    model = SLIM()

    with pytest.raises(ValueError):

        model._normalize_coprime(
            np.array([0, 0, 0])
        )


def test_negative_class_zero_score():

    X = np.array([
        [0],
        [1],
        [2],
        [3],
    ])

    y = np.array([
        -1,
        -1,
        1,
        1,
    ])

    model = SLIM(
        C0=0.001,
        coefficient_bound=5,
        fit_intercept=True,
    )

    model.fit(X, y)

    assert model.predict(
        np.array([[0]])
    )[0] == -1


def test_training_error_consistency():

    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    y = np.array([
        1, -1, 1, -1, 1, -1
    ])

    model = SLIM(
        C0=0.01,
        coefficient_bound=5
    )

    model.fit(X, y)

    actual_errors = (
        model.predict(X) != y
    ).astype(int)

    assert np.array_equal(
        actual_errors,
        model.misclassified_
    )


def test_model_is_coprime_after_fit(sample_data):

    X, y = sample_data

    model = SLIM()

    model.fit(X, y)

    assert model.is_coprime_
    assert model._coefficient_gcd() == 1
