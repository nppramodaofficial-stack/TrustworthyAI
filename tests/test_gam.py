
import numpy as np
import pytest

from trustworthy_ai.models.gam import GAM


@pytest.fixture
def sample_data():

    np.random.seed(42)

    X = np.random.uniform(
        0, 10, size=(150, 2)
    )

    y = (
        2 * np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + 3
        + np.random.normal(0, 0.2, 150)
    )

    return X, y


@pytest.fixture
def fitted_model(sample_data):

    X, y = sample_data

    model = GAM(
        n_splines=10,
        degree=3,
        lam=1.0,
        feature_names=["Age", "Income"]
    )

    model.fit(X, y)

    return model


# ---------------------------------------------
# 1. MODEL INITIALIZATION
# ---------------------------------------------

def test_model_initialization():

    model = GAM(
        n_splines=10,
        degree=3,
        lam=1.0
    )

    assert model.n_splines == 10
    assert model.degree == 3
    assert model.lam == 1.0
    assert model.is_fitted_ is False


def test_invalid_spline_count():

    with pytest.raises(ValueError):

        GAM(
            n_splines=3,
            degree=3
        )


def test_negative_regularization():

    with pytest.raises(ValueError):

        GAM(lam=-1)


# ---------------------------------------------
# 2. MODEL FITTING
# ---------------------------------------------

def test_model_fitting(fitted_model):

    assert fitted_model.is_fitted_ is True

    assert fitted_model.coefficients_.shape == (
        2, 10
    )

    assert np.isfinite(
        fitted_model.intercept_
    )


def test_training_predictions(sample_data, fitted_model):

    X, y = sample_data

    predictions = fitted_model.predict(X)

    assert predictions.shape == y.shape

    assert np.all(np.isfinite(predictions))


def test_training_metrics(fitted_model):

    assert fitted_model.training_mse_ >= 0

    assert fitted_model.training_rmse_ >= 0

    assert np.isfinite(
        fitted_model.training_r2_
    )


# ---------------------------------------------
# 3. PREDICTION
# ---------------------------------------------

def test_prediction_shape(fitted_model):

    X_test = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0]
    ])

    predictions = fitted_model.predict(X_test)

    assert predictions.shape == (3,)


def test_prediction_before_fit():

    model = GAM()

    with pytest.raises(RuntimeError):

        model.predict([[1, 2]])


def test_wrong_number_of_features(fitted_model):

    with pytest.raises(ValueError):

        fitted_model.predict(
            [[1, 2, 3]]
        )


# ---------------------------------------------
# 4. ADDITIVE PROPERTY
# ---------------------------------------------

def test_additive_prediction(fitted_model):

    X = np.array([
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    predictions = fitted_model.predict(X)

    for i in range(len(X)):

        explanation = fitted_model.explain(X[i])

        assert np.isclose(
            predictions[i],
            explanation["prediction"]
        )


def test_feature_contributions(fitted_model):

    x = np.array([4.0, 5.0])

    explanation = fitted_model.explain(x)

    contributions = explanation[
        "feature_contributions"
    ]

    expected = (
        explanation["intercept"]
        + sum(
            item["contribution"]
            for item in contributions.values()
        )
    )

    assert np.isclose(
        expected,
        explanation["prediction"]
    )


# ---------------------------------------------
# 5. EXPLANATIONS
# ---------------------------------------------

def test_explanation_structure(fitted_model):

    explanation = fitted_model.explain(
        [3.0, 4.0]
    )

    assert "intercept" in explanation

    assert "feature_contributions" in explanation

    assert "prediction" in explanation

    assert "Age" in explanation["feature_contributions"]

    assert "Income" in explanation["feature_contributions"]


def test_explanation_before_fit():

    model = GAM()

    with pytest.raises(RuntimeError):

        model.explain([1, 2])


def test_explanation_wrong_input(fitted_model):

    with pytest.raises(ValueError):

        fitted_model.explain([1, 2, 3])


# ---------------------------------------------
# 6. INPUT VALIDATION
# ---------------------------------------------

def test_empty_input():

    model = GAM()

    with pytest.raises(ValueError):

        model.fit(
            np.empty((0, 2)),
            np.array([])
        )


def test_nan_input():

    model = GAM()

    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, 5.0]
    ])

    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):

        model.fit(X, y)


def test_mismatched_target_length():

    model = GAM()

    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    y = np.array([1, 2])

    with pytest.raises(ValueError):

        model.fit(X, y)


def test_constant_feature():

    model = GAM()

    X = np.array([
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4]
    ])

    y = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):

        model.fit(X, y)


# ---------------------------------------------
# 7. FEATURE NAMES
# ---------------------------------------------

def test_invalid_feature_names(sample_data):

    X, y = sample_data

    model = GAM(
        feature_names=["Age"]
    )

    with pytest.raises(ValueError):

        model.fit(X, y)


# ---------------------------------------------
# 8. BOUNDARY BEHAVIOUR
# ---------------------------------------------

def test_prediction_outside_training_range(fitted_model):

    boundary = np.array([
        [10.0, 10.0]
    ])

    outside = np.array([
        [20.0, 20.0]
    ])

    boundary_prediction = fitted_model.predict(
        boundary
    )

    outside_prediction = fitted_model.predict(
        outside
    )

    assert np.allclose(
        boundary_prediction,
        outside_prediction
    )


# ---------------------------------------------
# 9. MODEL SUMMARY
# ---------------------------------------------

def test_model_summary(fitted_model):

    assert fitted_model.training_mse_ >= 0

    assert fitted_model.n_features_in_ == 2
