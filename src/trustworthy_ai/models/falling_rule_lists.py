"""Falling Rule List (FRL): Research-oriented implementation.

This implementation uses:
1. Binary conjunction rule generation
2. First-match rule assignment
3. Weighted isotonic regression (PAVA)
4. Smoothed probability estimation
5. Bayesian-inspired scoring
6. Simulated annealing
7. Prediction and rule-based explanations

Research note:
This is a heuristic Bayesian-inspired reimplementation,
not an exact reproduction of Wang & Rudin (AISTATS 2015).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import lgamma, log
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConjunctionRule:
    """A conjunction of binary features."""

    indices: tuple[int, ...]

    def matches(self, X: np.ndarray) -> np.ndarray:
        if not self.indices:
            return np.ones(X.shape[0], dtype=bool)

        return np.all(X[:, self.indices] == 1, axis=1)

    def describe(self, feature_names: Sequence[str]) -> str:
        return " AND ".join(
            feature_names[i] for i in self.indices
        )


class FallingRuleList:
    """Falling Rule List binary classifier."""

    def __init__(
            self,
            n_steps: int = 5000,
            max_rule_length: int = 2,
            min_support: int = 5,
            max_candidates: int = 100,
            max_rule_evaluations: int = 100_000,
            max_list_length: int = 12,
            length_prior_mean: float = 8.0,
            temperature: float = 1.0,
            random_state: int | None = 0,
            threshold: float = 0.5,
            feature_names: Sequence[str] | None = None,
    ) -> None:

        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")

        if max_rule_length < 1:
            raise ValueError("max_rule_length must be at least 1.")

        if min_support < 1:
            raise ValueError("min_support must be at least 1.")

        if max_candidates < 1 or max_rule_evaluations < 1:
            raise ValueError("Candidate limits must be positive.")

        if max_list_length < 1:
            raise ValueError("max_list_length must be at least 1.")

        if length_prior_mean <= 0 or temperature <= 0:
            raise ValueError(
                "length_prior_mean and temperature must be positive."
            )

        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1.")

        self.n_steps = int(n_steps)
        self.max_rule_length = int(max_rule_length)
        self.min_support = int(min_support)
        self.max_candidates = int(max_candidates)
        self.max_rule_evaluations = int(max_rule_evaluations)
        self.max_list_length = int(max_list_length)

        self.length_prior_mean = float(length_prior_mean)
        self.temperature = float(temperature)
        self.random_state = random_state
        self.threshold = float(threshold)
        self.feature_names = feature_names

    # --------------------------------------------------
    # 1. INPUT VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_X_y(X, y=None):

        X = np.asarray(X)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X must be a non-empty 2D array."
            )

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")

        if not np.isfinite(X).all() or not np.isin(X, [0, 1]).all():
            raise ValueError(
                "X must contain only finite binary values 0 and 1."
            )

        X = X.astype(np.int8, copy=False)

        if y is None:
            return X

        y = np.asarray(y)

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                "y must be 1D and have the same number of rows as X."
            )

        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must contain numeric values.")

        if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(
                "y must contain only finite binary labels 0 and 1."
            )

        return X, y.astype(np.int8, copy=False)

    # --------------------------------------------------
    # 2. CANDIDATE RULE GENERATION
    # --------------------------------------------------

    def _mine_rules(self, X):

        n_samples, n_features = X.shape

        found = []
        evaluations = 0

        for length in range(
                1, min(self.max_rule_length, n_features) + 1
        ):

            for indices in combinations(range(n_features), length):

                evaluations += 1

                if evaluations > self.max_rule_evaluations:
                    raise RuntimeError(
                        "Rule mining exceeded max_rule_evaluations. "
                        "Reduce max_rule_length or increase the limit."
                    )

                support = int(
                    np.all(X[:, indices] == 1, axis=1).sum()
                )

                if self.min_support <= support < n_samples:
                    found.append((indices, support))

        found.sort(
            key=lambda item: (
                -item[1],
                len(item[0]),
                item[0]
            )
        )

        return [
            ConjunctionRule(indices)
            for indices, _ in found[:self.max_candidates]
        ]

    # --------------------------------------------------
    # 3. FIRST-MATCH RULE ASSIGNMENT
    # --------------------------------------------------

    @staticmethod
    def _assign(X, rules):

        default_index = len(rules)

        assigned = np.full(
            X.shape[0],
            default_index,
            dtype=int
        )

        for rule_index, rule in enumerate(rules):
            matches = (
                    rule.matches(X)
                    & (assigned == default_index)
            )

            assigned[matches] = rule_index

        return assigned

    # --------------------------------------------------
    # 4. WEIGHTED ISOTONIC REGRESSION (PAVA)
    # --------------------------------------------------

    @staticmethod
    def _pava_decreasing(values, weights):

        blocks = []

        for value, weight in zip(values, weights):

            blocks.append([
                float(value * weight),
                float(weight)
            ])

            while len(blocks) >= 2:

                left_mean = (
                        blocks[-2][0] / blocks[-2][1]
                )

                right_mean = (
                        blocks[-1][0] / blocks[-1][1]
                )

                # Enforce non-increasing probabilities.
                if left_mean >= right_mean:
                    break

                right = blocks.pop()
                left = blocks.pop()

                blocks.append([
                    left[0] + right[0],
                    left[1] + right[1]
                ])

        fitted = []

        for weighted_sum, total_weight in blocks:
            fitted.extend(
                [weighted_sum / total_weight]
                * int(total_weight)
            )

        return np.asarray(fitted, dtype=float)

    # --------------------------------------------------
    # 5. PROBABILITY ESTIMATION
    # --------------------------------------------------

    def _estimate_probabilities(
            self, y, assignments, n_groups
    ):

        successes = np.bincount(
            assignments,
            weights=y,
            minlength=n_groups
        )

        counts = np.bincount(
            assignments,
            minlength=n_groups
        ).astype(float)

        # Jeffreys-style Beta smoothing.
        raw_rates = (
                            successes + 0.5
                    ) / (
                            counts + 1.0
                    )

        weights = counts + 1.0

        probabilities = self._pava_decreasing(
            raw_rates,
            weights
        )

        return probabilities

    # --------------------------------------------------
    # 6. LOG-LIKELIHOOD
    # --------------------------------------------------

    @staticmethod
    def _bernoulli_log_likelihood(y, probabilities):

        p = np.clip(
            probabilities,
            1e-12,
            1.0 - 1e-12
        )

        return float(
            np.sum(
                y * np.log(p)
                + (1 - y) * np.log1p(-p)
            )
        )

    # --------------------------------------------------
    # 7. BAYESIAN-INSPIRED SCORING
    # --------------------------------------------------

    def _score(self, X, y, rules):

        assignments = self._assign(X, rules)

        probabilities = self._estimate_probabilities(
            y,
            assignments,
            len(rules) + 1
        )

        sample_probabilities = probabilities[assignments]

        log_likelihood = self._bernoulli_log_likelihood(
            y,
            sample_probabilities
        )

        # Poisson-inspired prior on list length.
        length = len(rules)

        log_length_prior = (
                length * log(self.length_prior_mean)
                - self.length_prior_mean
                - lgamma(length + 1)
        )

        # Uniform-without-replacement rule selection penalty.
        candidate_count = max(
            len(self.candidate_rules_),
            1
        )

        log_rule_prior = 0.0

        for position in range(length):
            remaining = max(
                candidate_count - position,
                1
            )

            log_rule_prior -= log(remaining)

        score = (
                log_likelihood
                + log_length_prior
                + log_rule_prior
        )

        return score, probabilities

    # --------------------------------------------------
    # 8. MODEL FITTING
    # --------------------------------------------------

    def fit(self, X, y, feature_names=None):

        X, y = self._validate_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        names = (
            feature_names
            if feature_names is not None
            else self.feature_names
        )

        if names is None:
            names = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        if len(names) != self.n_features_in_:
            raise ValueError(
                "feature_names must have one name per input feature."
            )

        self.feature_names_in_ = [
            str(name) for name in names
        ]

        self.candidate_rules_ = self._mine_rules(X)

        rng = np.random.default_rng(self.random_state)

        # Handle absence of candidate rules.
        if not self.candidate_rules_:
            self.rules_ = []

            self.probabilities_ = self._estimate_probabilities(
                y,
                np.zeros(len(y), dtype=int),
                1
            )

            self.search_score_ = self._bernoulli_log_likelihood(
                y,
                np.full(len(y), self.probabilities_[0])
            )

            self.n_iter_ = 0
            self.classes_ = np.array([0, 1])

            return self

        initial_length = min(
            3,
            self.max_list_length,
            len(self.candidate_rules_)
        )

        current = list(
            self.candidate_rules_[:initial_length]
        )

        current_score, current_probabilities = self._score(
            X, y, current
        )

        best = list(current)
        best_score = current_score
        best_probabilities = current_probabilities.copy()

        # --------------------------------------------------
        # SIMULATED ANNEALING
        # --------------------------------------------------

        for step in range(self.n_steps):

            proposal = list(current)

            move = int(rng.integers(0, 3))

            # SWAP MOVE
            if move == 0 and len(proposal) >= 2:

                i, j = rng.choice(
                    len(proposal),
                    size=2,
                    replace=False
                )

                proposal[int(i)], proposal[int(j)] = (
                    proposal[int(j)],
                    proposal[int(i)]
                )

            # ADD MOVE
            elif move == 1 and len(proposal) < min(
                    self.max_list_length,
                    len(self.candidate_rules_)
            ):

                available = [
                    r for r in self.candidate_rules_
                    if r not in proposal
                ]

                if available:
                    position = int(
                        rng.integers(0, len(proposal) + 1)
                    )

                    chosen = available[
                        int(rng.integers(0, len(available)))
                    ]

                    proposal.insert(position, chosen)

            # REMOVE MOVE
            elif move == 2 and proposal:

                proposal.pop(
                    int(rng.integers(0, len(proposal)))
                )

            if len({
                rule.indices for rule in proposal
            }) != len(proposal):
                continue

            proposed_score, proposed_probabilities = self._score(
                X, y, proposal
            )

            cooling = max(
                0.05,
                1.0 - step / self.n_steps
            )

            current_temperature = (
                    self.temperature * cooling
            )

            delta = proposed_score - current_score

            accept = (
                    delta >= 0
                    or rng.random() < np.exp(
                np.clip(
                    delta / current_temperature,
                    -50.0,
                    0.0
                )
            )
            )

            if accept:
                current = proposal
                current_score = proposed_score
                current_probabilities = proposed_probabilities

            if current_score > best_score:
                best = list(current)
                best_score = current_score

                best_probabilities = (
                    current_probabilities.copy()
                )

        self.rules_ = best

        self.probabilities_ = best_probabilities

        self.search_score_ = float(best_score)

        self.n_iter_ = self.n_steps

        self.classes_ = np.array([0, 1])

        self.n_candidates_ = len(self.candidate_rules_)

        return self

    # --------------------------------------------------
    # 9. INPUT CHECKING
    # --------------------------------------------------

    def _check_X(self, X):

        if not hasattr(self, "rules_"):
            raise RuntimeError(
                "Call fit() before prediction."
            )

        X = self._validate_X_y(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; "
                f"expected {self.n_features_in_}."
            )

        return X

    # --------------------------------------------------
    # 10. PREDICTION PROBABILITIES
    # --------------------------------------------------

    def predict_proba(self, X):

        X = self._check_X(X)

        assignments = self._assign(
            X,
            self.rules_
        )

        positive = self.probabilities_[assignments]

        return np.column_stack((
            1.0 - positive,
            positive
        ))

    # --------------------------------------------------
    # 11. CLASS PREDICTION
    # --------------------------------------------------

    def predict(self, X):

        return (
                self.predict_proba(X)[:, 1]
                >= self.threshold
        ).astype(int)

    # --------------------------------------------------
    # 12. EXPLANATIONS
    # --------------------------------------------------

    def explain(self, X):

        X = self._check_X(X)

        assignments = self._assign(
            X,
            self.rules_
        )

        explanations = []

        for sample_index, group_index in enumerate(assignments):

            if group_index < len(self.rules_):

                rule = self.rules_[group_index]

                description = rule.describe(
                    self.feature_names_in_
                )

                label = (
                    f"Rule {group_index + 1}: "
                    f"{description}"
                )

            else:

                label = "Default (no rule matched)"

            probability = float(
                self.probabilities_[group_index]
            )

            explanations.append({

                "sample_index": sample_index,

                "rule_position": int(group_index),

                "rule": label,

                "probability": probability,

                "prediction": int(
                    probability >= self.threshold
                )

            })

        return explanations

    # --------------------------------------------------
    # 13. DISPLAY RULE LIST
    # --------------------------------------------------

    def get_rules(self):

        if not hasattr(self, "rules_"):
            raise RuntimeError(
                "Call fit() before get_rules()."
            )

        output = []

        for index, rule in enumerate(self.rules_):
            output.append({

                "position": index + 1,

                "conditions": rule.describe(
                    self.feature_names_in_
                ),

                "probability": float(
                    self.probabilities_[index]
                )

            })

        output.append({

            "position": "default",

            "conditions": "otherwise",

            "probability": float(
                self.probabilities_[-1]
            )

        })

        return output

    # --------------------------------------------------
    # 14. MODEL ACCURACY
    # --------------------------------------------------

    def score(self, X, y):

        X, y = self._validate_X_y(X, y)

        predictions = self.predict(X)

        return float(
            np.mean(predictions == y)
        )

    # --------------------------------------------------
    # 15. MODEL REPRESENTATION
    # --------------------------------------------------

    def __repr__(self):

        return (
            "FallingRuleList("
            f"n_steps={self.n_steps}, "
            f"max_rule_length={self.max_rule_length}, "
            f"min_support={self.min_support}, "
            f"random_state={self.random_state})"
        )
