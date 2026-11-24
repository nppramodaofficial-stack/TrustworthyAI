import numpy as np
import pytest

from trustworthy_ai.models.falling_rule_lists import (
    FallingRuleList,
)


def make_data():
    rng = np.random.default_rng(9)
    X = rng.integers(0, 2, size=(120, 4))

    y = (
        (X[:, 0] == 1)
        | ((X[:, 1] == 1) & (X[:, 2] == 1))
    ).astype(int)

    return X, y


def test_fit_predict_and_probability_shapes():
    X, y = make_data()

    model = FallingRuleList(
        n_steps=100,
        min_support=4,
        random_state=2,
    ).fit(X, y)

    assert model.predict(X).shape == (len(y),)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(y), 2)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_rejects_non_binary_features():
    with pytest.raises(ValueError, match="binary"):
        FallingRuleList().fit(
            [[0, 0.5], [1, 0]],
            [0, 1],
        )


def test_rejects_non_binary_targets():
    with pytest.raises(ValueError, match="binary"):
        FallingRuleList().fit(
            [[0], [1]],
            [0, 2],
        )


def test_prediction_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        FallingRuleList().predict([[0, 1]])


def test_explanations_match_predictions():
    X, y = make_data()

    model = FallingRuleList(
        n_steps=100,
        min_support=4,
        random_state=2,
    ).fit(X, y)

    explanations = model.explain(X[:8])

    assert len(explanations) == 8
    assert [
        row["prediction"] for row in explanations
    ] == model.predict(X[:8]).tolist()


def test_probabilities_are_finite_and_bounded():
    X, y = make_data()

    model = FallingRuleList(
        n_steps=100,
        min_support=4,
        random_state=2,
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()