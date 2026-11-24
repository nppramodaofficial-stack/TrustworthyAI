
import numpy as np


class BSplineBasis:

    def __init__(self, n_splines=10, degree=3):

        if degree < 0:
            raise ValueError(
                "degree must be non-negative."
            )

        if n_splines <= degree:
            raise ValueError(
                "n_splines must be greater than degree."
            )

        self.n_splines = n_splines
        self.degree = degree

        self.knots_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = np.asarray(X, dtype=float).reshape(-1)

        if X.size == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        self.min_ = np.min(X)
        self.max_ = np.max(X)

        if self.min_ == self.max_:
            raise ValueError(
                "X must contain at least two distinct values."
            )

        n_internal = (
            self.n_splines - self.degree - 1
        )

        internal_knots = np.linspace(
            self.min_,
            self.max_,
            n_internal + 2
        )[1:-1]

        left_knots = np.repeat(
            self.min_,
            self.degree + 1
        )

        right_knots = np.repeat(
            self.max_,
            self.degree + 1
        )

        self.knots_ = np.concatenate([
            left_knots,
            internal_knots,
            right_knots
        ])

        self.is_fitted_ = True

        return self

    def _basis_recursive(self, x, i, degree):

        knots = self.knots_

        if degree == 0:

            if knots[i] <= x < knots[i + 1]:
                return 1.0

            return 0.0

        denominator_1 = (
            knots[i + degree] - knots[i]
        )

        term_1 = 0.0

        if denominator_1 != 0:

            term_1 = (
                (x - knots[i]) / denominator_1
            ) * self._basis_recursive(
                x, i, degree - 1
            )

        denominator_2 = (
            knots[i + degree + 1]
            - knots[i + 1]
        )

        term_2 = 0.0

        if denominator_2 != 0:

            term_2 = (
                (knots[i + degree + 1] - x)
                / denominator_2
            ) * self._basis_recursive(
                x, i + 1, degree - 1
            )

        return term_1 + term_2

    def transform(self, X):

        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before transform()."
            )

        X = np.asarray(X, dtype=float).reshape(-1)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        if np.any(X < self.min_) or np.any(X > self.max_):
            raise ValueError(
                "X contains values outside the training range."
            )

        basis_matrix = np.zeros(
            (len(X), self.n_splines)
        )

        for row, x in enumerate(X):

            for i in range(self.n_splines):

                if (
                    x == self.max_
                    and i == self.n_splines - 1
                ):

                    basis_matrix[row, i] = 1.0

                else:

                    basis_matrix[row, i] = (
                        self._basis_recursive(
                            x,
                            i,
                            self.degree
                        )
                    )

        return basis_matrix

    def fit_transform(self, X):

        return self.fit(X).transform(X)
