
"""
SLIM: Supersparse Linear Integer Models
========================================

A practical Mixed Integer Programming implementation of
the core SLIM scoring-system objective.

Reference
---------
Ustun, B. and Rudin, C. (2016).
Supersparse Linear Integer Models for Optimized Medical
Scoring Systems. Machine Learning, 102, 349-391.

Optimization objective
----------------------

    Minimize:

        (1/N) * sum(z_i) + C0 * sum(u_j)

    where:

        z_i = binary misclassification indicator
        u_j = binary feature-selection indicator

    Subject to integer coefficient bounds and
    classification constraints.

Important implementation notes
------------------------------

1. Coefficients are integer-valued.
2. The intercept is included in the scoring vector.
3. Feature sparsity excludes the intercept.
4. Coprimality is enforced by post-solution normalization,
   not directly inside the MIP.
5. The positive-class constraint uses epsilon. For binary
   integer-valued inputs, epsilon=1e-6 implements the
   intended zero-boundary decision rule. For continuous
   inputs, it introduces a small positive-margin condition.
6. This is a practical SLIM-style MIP implementation,
   not a claim of complete reproduction of every detail
   of the original paper.

Dependencies
------------
numpy
scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Optional, Sequence

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, csr_matrix


# ======================================================
# 1. FEATURE CONTRIBUTION
# ======================================================

@dataclass(frozen=True)
class FeatureContribution:
    """Contribution of one feature to a prediction."""

    feature: str
    value: float
    coefficient: int
    contribution: float


# ======================================================
# 2. SLIM CLASS
# ======================================================

class SLIM:
    """
    Supersparse Linear Integer Model.

    Parameters
    ----------
    C0 : float, default=0.01
        Penalty for each selected feature.

    coefficient_bound : int, default=10
        Maximum absolute value of each coefficient
        and the intercept.

    fit_intercept : bool, default=True
        Whether to learn an integer intercept.

    penalize_intercept : bool, default=False
        Whether to include the intercept in the
        sparsity penalty.

    time_limit : float, default=120
        Maximum optimization time in seconds.

    mip_rel_gap : float, default=0.0
        Relative MIP optimality gap.

    epsilon : float, default=1e-6
        Positive-class decision margin used in
        the classification constraints.

    feature_names : sequence of str, optional
        Names of the input features.

    Attributes
    ----------
    coef_ : ndarray
        Learned integer feature coefficients.

    intercept_ : int
        Learned integer intercept.

    selected_features_ : ndarray
        Indices of selected features.

    n_features_selected_ : int
        Number of non-zero feature coefficients.

    training_accuracy_ : float
        Training accuracy.

    optimization_result_ : OptimizeResult
        Result returned by scipy.optimize.milp.

    is_optimal_ : bool
        Whether the solver proved optimality.

    is_fitted_ : bool
        Whether the model has been fitted.
    """

    # ==================================================
    # 3. INITIALIZATION
    # ==================================================

    def __init__(
        self,
        C0: float = 0.01,
        coefficient_bound: int = 10,
        fit_intercept: bool = True,
        penalize_intercept: bool = False,
        time_limit: float = 120,
        mip_rel_gap: float = 0.0,
        epsilon: float = 1e-6,
        feature_names: Optional[Sequence[str]] = None,
    ):

        if not np.isfinite(C0) or C0 < 0:
            raise ValueError(
                "C0 must be a finite non-negative number."
            )

        if (
            isinstance(coefficient_bound, bool)
            or not isinstance(
                coefficient_bound,
                (int, np.integer)
            )
            or coefficient_bound < 1
        ):
            raise ValueError(
                "coefficient_bound must be a positive integer."
            )

        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError(
                "time_limit must be positive and finite."
            )

        if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
            raise ValueError(
                "mip_rel_gap must be non-negative and finite."
            )

        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(
                "epsilon must be positive and finite."
            )

        self.C0 = float(C0)

        self.coefficient_bound = int(
            coefficient_bound
        )

        self.fit_intercept = bool(
            fit_intercept
        )

        self.penalize_intercept = bool(
            penalize_intercept
        )

        self.time_limit = float(
            time_limit
        )

        self.mip_rel_gap = float(
            mip_rel_gap
        )

        self.epsilon = float(epsilon)

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else None
        )

        self.is_fitted_ = False

    # ==================================================
    # 4. INPUT VALIDATION
    # ==================================================

    @staticmethod
    def _validate_X(X):

        X = np.asarray(
            X,
            dtype=float
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X cannot be empty."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    @staticmethod
    def _validate_y(y, n_samples):

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(
                "y must be one-dimensional."
            )

        if len(y) != n_samples:
            raise ValueError(
                "X and y must contain the same "
                "number of observations."
            )

        if not np.all(np.isin(y, [0, 1, -1])):
            raise ValueError(
                "Labels must contain only 0, 1, or -1."
            )

        unique = set(
            np.unique(y).tolist()
        )

        if unique not in ({0, 1}, {-1, 1}):
            raise ValueError(
                "Both classes must be present. "
                "Use either labels {0, 1} or {-1, +1}."
            )

        if unique == {0, 1}:

            y = np.where(
                y == 1,
                1,
                -1
            )

        else:

            y = y.astype(int)

        return y

    # ==================================================
    # 5. FEATURE NAMES
    # ==================================================

    def _prepare_feature_names(self, n_features):

        if self.feature_names is None:

            return [
                f"feature_{i}"
                for i in range(n_features)
            ]

        if len(self.feature_names) != n_features:

            raise ValueError(
                "Number of feature names must match "
                "the number of columns in X."
            )

        return list(self.feature_names)

    # ==================================================
    # 6. COPRIME COEFFICIENT NORMALIZATION
    # ==================================================

    @staticmethod
    def _normalize_coprime(coefficients):
        """
        Divide the complete integer scoring vector
        by its greatest common divisor.

        The intercept must be included in the vector.

        Dividing all coefficients by the same positive
        integer preserves the sign of every score.
        """

        coefficients = np.asarray(
            coefficients,
            dtype=int
        ).copy()

        nonzero = np.abs(
            coefficients[
                coefficients != 0
            ]
        )

        if len(nonzero) == 0:
            raise ValueError(
                "The all-zero scoring vector cannot "
                "satisfy the coprimality condition."
            )

        common_divisor = reduce(
            gcd,
            nonzero.tolist()
        )

        if common_divisor > 1:

            coefficients //= common_divisor

        return coefficients

    # ==================================================
    # 7. BUILD OBJECTIVE FUNCTION
    # ==================================================

    def _build_objective(
        self,
        n_samples,
        n_features,
        has_intercept,
    ):

        # Coefficient vector:
        #
        # [intercept, beta_1, beta_2, ..., beta_p]
        #
        # if intercept is enabled.
        #
        # Decision variables:
        #
        # beta = integer coefficients
        # u = binary feature-selection variables
        # z = binary misclassification variables

        n_beta = (
            n_features + int(has_intercept)
        )

        n_variables = (
            n_beta
            + n_features
            + n_samples
        )

        objective = np.zeros(
            n_variables,
            dtype=float
        )

        beta_start = 0

        u_start = n_beta

        z_start = (
            n_beta + n_features
        )

        # Feature sparsity penalty.
        objective[
            u_start:u_start + n_features
        ] = self.C0

        # Optional intercept penalty.
        if (
            has_intercept
            and self.penalize_intercept
        ):

            objective[0] = self.C0

        # Classification error penalty.
        objective[
            z_start:z_start + n_samples
        ] = 1.0 / n_samples

        return (
            objective,
            n_variables,
            n_beta,
            beta_start,
            u_start,
            z_start,
        )

    # ==================================================
    # 8. BUILD MIP CONSTRAINTS
    # ==================================================

    def _build_constraints(
        self,
        X,
        y,
        has_intercept,
        n_variables,
        n_beta,
        beta_start,
        u_start,
        z_start,
    ):

        n_samples, n_features = X.shape

        L = self.coefficient_bound

        # Add intercept as the FIRST column.
        if has_intercept:

            X_design = np.column_stack(
                [
                    np.ones(n_samples),
                    X
                ]
            )

        else:

            X_design = X.copy()

        # Number of constraints:
        #
        # n_samples classification constraints
        # 2 * n_features coefficient-selection constraints

        n_constraints = (
            n_samples + 2 * n_features
        )

        A = lil_matrix(
            (n_constraints, n_variables),
            dtype=float
        )

        lower = np.full(
            n_constraints,
            -np.inf
        )

        upper = np.zeros(
            n_constraints
        )

        # Maximum possible absolute score for each sample.
        #
        # Every coefficient is bounded by L.

        maximum_score = (
            L * np.sum(
                np.abs(X_design),
                axis=1
            )
        )

        big_M = (
            maximum_score + self.epsilon
        )

        row = 0

        # ----------------------------------------------
        # A. CLASSIFICATION CONSTRAINTS
        # ----------------------------------------------

        for i in range(n_samples):

            if y[i] == 1:

                # Positive class:
                #
                # score >= epsilon - M*z
                #
                # Rearranged:
                #
                # -score - M*z <= -epsilon

                for j in range(n_beta):

                    A[
                        row,
                        beta_start + j
                    ] = -X_design[i, j]

                A[
                    row,
                    z_start + i
                ] = -big_M[i]

                upper[row] = -self.epsilon

            else:

                # Negative class:
                #
                # score <= M*z
                #
                # Rearranged:
                #
                # score - M*z <= 0

                for j in range(n_beta):

                    A[
                        row,
                        beta_start + j
                    ] = X_design[i, j]

                A[
                    row,
                    z_start + i
                ] = -big_M[i]

                upper[row] = 0

            row += 1

        # ----------------------------------------------
        # B. FEATURE SELECTION CONSTRAINTS
        # ----------------------------------------------

        # -L*u_j <= beta_j <= L*u_j
        #
        # u_j = 0 forces beta_j = 0.
        #
        # u_j = 1 allows beta_j to be non-zero.

        intercept_offset = int(has_intercept)

        for j in range(n_features):

            coefficient_index = (
                beta_start
                + intercept_offset
                + j
            )

            # beta_j - L*u_j <= 0

            A[
                row,
                coefficient_index
            ] = 1

            A[
                row,
                u_start + j
            ] = -L

            upper[row] = 0

            row += 1

            # -beta_j - L*u_j <= 0

            A[
                row,
                coefficient_index
            ] = -1

            A[
                row,
                u_start + j
            ] = -L

            upper[row] = 0

            row += 1

        return LinearConstraint(
            csr_matrix(A),
            lower,
            upper
        )

    # ==================================================
    # 9. BUILD VARIABLE BOUNDS
    # ==================================================

    def _build_bounds(
        self,
        n_samples,
        n_features,
        n_beta,
        n_variables,
    ):

        L = self.coefficient_bound

        lower = np.zeros(
            n_variables
        )

        upper = np.zeros(
            n_variables
        )

        # Integer coefficient bounds.
        lower[:n_beta] = -L
        upper[:n_beta] = L

        # Binary feature-selection variables.
        u_start = n_beta

        lower[
            u_start:u_start + n_features
        ] = 0

        upper[
            u_start:u_start + n_features
        ] = 1

        # Binary error variables.
        z_start = (
            n_beta + n_features
        )

        lower[
            z_start:z_start + n_samples
        ] = 0

        upper[
            z_start:z_start + n_samples
        ] = 1

        return Bounds(
            lower,
            upper
        )

    # ==================================================
    # 10. FIT MODEL
    # ==================================================

    def fit(self, X, y):

        # Validate inputs.
        X = self._validate_X(X)

        y = self._validate_y(
            y,
            X.shape[0]
        )

        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        self.feature_names_ = (
            self._prepare_feature_names(
                n_features
            )
        )

        has_intercept = self.fit_intercept

        # Build objective.
        (
            objective,
            n_variables,
            n_beta,
            beta_start,
            u_start,
            z_start,
        ) = self._build_objective(
            n_samples,
            n_features,
            has_intercept
        )

        # Build constraints.
        constraints = self._build_constraints(
            X,
            y,
            has_intercept,
            n_variables,
            n_beta,
            beta_start,
            u_start,
            z_start
        )

        # Build bounds.
        bounds = self._build_bounds(
            n_samples,
            n_features,
            n_beta,
            n_variables
        )

        # All variables are integer:
        #
        # beta = integer coefficients
        # u = binary feature-selection variables
        # z = binary error variables

        integrality = np.ones(
            n_variables,
            dtype=int
        )

        # Solve MIP.
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": self.time_limit,
                "mip_rel_gap": self.mip_rel_gap,
                "disp": False,
            }
        )

        self.optimization_result_ = result

        if result.x is None:

            raise RuntimeError(
                "SLIM optimization did not produce "
                "a feasible solution. "
                f"Solver message: {result.message}"
            )

        solution = np.rint(
            result.x
        ).astype(int)

        # ----------------------------------------------
        # Extract coefficients.
        # ----------------------------------------------

        beta = solution[
            beta_start:beta_start + n_beta
        ]

        if has_intercept:

            self.intercept_ = int(
                beta[0]
            )

            self.coef_ = beta[1:].copy()

        else:

            self.intercept_ = 0

            self.coef_ = beta.copy()

        # ----------------------------------------------
        # Normalize complete coefficient vector.
        # ----------------------------------------------

        full_coefficients = np.concatenate(
            [
                [self.intercept_],
                self.coef_
            ]
        )

        try:

            full_coefficients = (
                self._normalize_coprime(
                    full_coefficients
                )
            )

        except ValueError as exc:

            raise RuntimeError(
                "The solver returned an all-zero "
                "scoring vector. Try adjusting C0 "
                "or the coefficient bounds."
            ) from exc

        self.intercept_ = int(
            full_coefficients[0]
        )

        self.coef_ = (
            full_coefficients[1:].copy()
        )

        # ----------------------------------------------
        # Extract solver variables.
        # ----------------------------------------------

        self.solver_feature_selection_ = (
            solution[
                u_start:u_start + n_features
            ].astype(int)
        )

        self.solver_misclassified_ = (
            solution[
                z_start:z_start + n_samples
            ].astype(int)
        )

        # ----------------------------------------------
        # Calculate actual predictions and errors.
        # ----------------------------------------------

        self.selected_features_ = (
            np.flatnonzero(
                self.coef_ != 0
            )
        )

        self.n_features_selected_ = int(
            np.count_nonzero(
                self.coef_
            )
        )

        self.is_fitted_ = True

        predictions = self.predict(X)

        self.misclassified_ = (
            predictions != y
        ).astype(int)

        self.training_accuracy_ = float(
            np.mean(
                predictions == y
            )
        )

        self.training_error_ = float(
            1.0 - self.training_accuracy_
        )

        # ----------------------------------------------
        # Objective evaluation.
        # ----------------------------------------------

        self.objective_value_ = float(
            self._calculate_objective(
                X,
                y
            )
        )

        self.solver_objective_ = (
            float(result.fun)
            if result.fun is not None
            else None
        )

        self.is_optimal_ = (
            result.status == 0
        )

        self.solver_status_ = int(
            result.status
        )

        self.solver_message_ = str(
            result.message
        )

        self.mip_gap_ = getattr(
            result,
            "mip_gap",
            None
        )

        self.n_iterations_ = getattr(
            result,
            "nit",
            None
        )

        self.n_nodes_ = getattr(
            result,
            "mip_node_count",
            None
        )

        self.is_coprime_ = (
            self._coefficient_gcd() == 1
        )

        return self

    # ==================================================
    # 11. COEFFICIENT GCD
    # ==================================================

    def _coefficient_gcd(self):

        self._check_fitted()

        coefficients = np.concatenate(
            [
                [self.intercept_],
                self.coef_
            ]
        )

        nonzero = np.abs(
            coefficients[
                coefficients != 0
            ]
        )

        if len(nonzero) == 0:
            return 0

        return reduce(
            gcd,
            nonzero.tolist()
        )

    # ==================================================
    # 12. DECISION FUNCTION
    # ==================================================

    def decision_function(self, X):

        self._check_fitted()

        X = self._validate_prediction_X(X)

        return (
            X @ self.coef_
            + self.intercept_
        )

    # ==================================================
    # 13. PREDICTION
    # ==================================================

    def predict(self, X):

        scores = self.decision_function(X)

        # Zero belongs to the negative class.
        return np.where(
            scores > 0,
            1,
            -1
        )

    # ==================================================
    # 14. PROBABILITY PREDICTION
    # ==================================================

    def predict_proba(self, X):

        """
        SLIM is an integer scoring classifier.

        Its decision scores are not probabilities.

        A separate calibration procedure is required
        to obtain probability estimates.
        """

        raise NotImplementedError(
            "SLIM does not directly produce probabilities. "
            "Use decision_function() or apply a separate "
            "probability calibration method."
        )

    # ==================================================
    # 15. FEATURE EXPLANATION
    # ==================================================

    def explain(
        self,
        X,
        sample_index=0,
    ):

        self._check_fitted()

        X = self._validate_prediction_X(X)

        if (
            sample_index < 0
            or sample_index >= len(X)
        ):

            raise IndexError(
                "sample_index is out of range."
            )

        sample = X[sample_index]

        explanations = []

        for j in range(
            self.n_features_in_
        ):

            coefficient = int(
                self.coef_[j]
            )

            if coefficient != 0:

                contribution = (
                    sample[j] * coefficient
                )

                explanations.append(
                    FeatureContribution(
                        feature=self.feature_names_[j],
                        value=float(sample[j]),
                        coefficient=coefficient,
                        contribution=float(contribution)
                    )
                )

        score = float(
            sample @ self.coef_
            + self.intercept_
        )

        prediction = int(
            1 if score > 0 else -1
        )

        return {
            "contributions": explanations,
            "intercept": self.intercept_,
            "score": score,
            "prediction": prediction,
        }

    # ==================================================
    # 16. GET RULES
    # ==================================================

    def get_rules(self):

        self._check_fitted()

        rules = []

        for j, coefficient in enumerate(
            self.coef_
        ):

            if coefficient != 0:

                rules.append({
                    "feature": self.feature_names_[j],
                    "coefficient": int(coefficient),
                })

        return rules

    # ==================================================
    # 17. GET SCORECARD
    # ==================================================

    def get_scorecard(self):

        self._check_fitted()

        return {
            "intercept": self.intercept_,
            "features": self.get_rules(),
            "decision_rule": (
                "Predict +1 if score > 0, otherwise -1"
            ),
        }

    # ==================================================
    # 18. OBJECTIVE EVALUATION
    # ==================================================

    def _calculate_objective(self, X, y):

        scores = (
            X @ self.coef_
            + self.intercept_
        )

        predictions = np.where(
            scores > 0,
            1,
            -1
        )

        error_rate = np.mean(
            predictions != y
        )

        sparsity_penalty = (
            self.C0
            * np.count_nonzero(
                self.coef_
            )
        )

        if self.penalize_intercept:

            sparsity_penalty += (
                self.C0
                * int(self.intercept_ != 0)
            )

        return float(
            error_rate + sparsity_penalty
        )

    # ==================================================
    # 19. PREDICTION INPUT VALIDATION
    # ==================================================

    def _validate_prediction_X(self, X):

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:

            raise ValueError(
                "Prediction data has a different "
                "number of features than training data."
            )

        return X

    # ==================================================
    # 20. CHECK FITTED STATUS
    # ==================================================

    def _check_fitted(self):

        if not self.is_fitted_:

            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() before prediction."
            )

    # ==================================================
    # 21. MODEL SUMMARY
    # ==================================================

    def summary(self):

        self._check_fitted()

        print("\nSLIM MODEL SUMMARY")
        print("=" * 50)

        print(
            f"Training accuracy : "
            f"{self.training_accuracy_:.4f}"
        )

        print(
            f"Training error    : "
            f"{self.training_error_:.4f}"
        )

        print(
            f"Selected features : "
            f"{self.n_features_selected_}"
        )

        print(
            f"Intercept         : "
            f"{self.intercept_}"
        )

        print(
            f"Objective value   : "
            f"{self.objective_value_:.4f}"
        )

        print(
            f"Solver objective  : "
            f"{self.solver_objective_}"
        )

        print(
            f"Solver optimal    : "
            f"{self.is_optimal_}"
        )

        print(
            f"Solver status     : "
            f"{self.solver_status_}"
        )

        print(
            f"Coprime vector    : "
            f"{self.is_coprime_}"
        )

        print(
            f"MIP gap           : "
            f"{self.mip_gap_}"
        )

        print("\nSCORING SYSTEM")
        print("-" * 50)

        print(
            f"Intercept: {self.intercept_:+d}"
        )

        for rule in self.get_rules():

            print(
                f"{rule['feature']}: "
                f"{rule['coefficient']:+d}"
            )

        print("-" * 50)

        print(
            "Prediction = +1 if score > 0, "
            "otherwise -1"
        )

    # ==================================================
    # 22. STRING REPRESENTATION
    # ==================================================

    def __repr__(self):

        if not self.is_fitted_:

            return "SLIM(not fitted)"

        return (
            f"SLIM("
            f"features={self.n_features_selected_}, "
            f"intercept={self.intercept_}, "
            f"accuracy={self.training_accuracy_:.3f}, "
            f"optimal={self.is_optimal_}"
            f")"
        )
