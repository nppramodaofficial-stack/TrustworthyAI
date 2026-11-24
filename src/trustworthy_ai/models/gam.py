
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import block_diag

from .splines import BSplineBasis


class GAM:
    """
    Generalized Additive Model for Gaussian regression.

    Model:
        y = intercept + f1(X1) + ... + fp(Xp) + error

    Each feature function is represented using B-spline basis
    functions and estimated using penalized least squares.

    Objective:
        ||y - X_design @ beta||^2
        + lam * beta.T @ P @ beta

    Parameters
    ----------
    n_splines : int, default=10
        Number of B-spline basis functions per feature.

    degree : int, default=3
        Degree of the B-spline basis functions.

    lam : float, default=1.0
        Smoothness regularization strength.

    feature_names : list, optional
        Names of the input features.
    """

    def __init__(
        self,
        n_splines=10,
        degree=3,
        lam=1.0,
        feature_names=None
    ):

        if n_splines <= degree:
            raise ValueError(
                "n_splines must be greater than degree."
            )

        if degree < 0:
            raise ValueError(
                "degree must be non-negative."
            )

        if lam < 0:
            raise ValueError(
                "lam must be non-negative."
            )

        self.n_splines = n_splines
        self.degree = degree
        self.lam = lam
        self.feature_names = feature_names

        self.splines_ = []
        self.coefficients_ = None
        self.intercept_ = None
        self.is_fitted_ = False

        self.n_features_in_ = None
        self.feature_names_in_ = None

    # --------------------------------------------------
    # 1. INPUT VALIDATION
    # --------------------------------------------------

    def _validate_X(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X cannot be empty."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X must contain only finite values."
            )

        return X

    def _validate_y(self, y, n_samples):

        y = np.asarray(y, dtype=float).reshape(-1)

        if len(y) != n_samples:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if not np.all(np.isfinite(y)):
            raise ValueError(
                "y must contain only finite values."
            )

        return y

    # --------------------------------------------------
    # 2. FIT SPLINE BASIS FUNCTIONS
    # --------------------------------------------------

    def _fit_splines(self, X):

        self.splines_ = []

        for j in range(X.shape[1]):

            spline = BSplineBasis(
                n_splines=self.n_splines,
                degree=self.degree
            )

            spline.fit(X[:, j])

            self.splines_.append(spline)

    # --------------------------------------------------
    # 3. CONSTRUCT ADDITIVE DESIGN MATRIX
    # --------------------------------------------------

    def _build_design_matrix(self, X):

        basis_matrices = []

        for j, spline in enumerate(self.splines_):

            # Clip values outside the training range.
            # This gives constant boundary effects
            # instead of extrapolating beyond the data.

            X_j = np.clip(
                X[:, j],
                spline.min_,
                spline.max_
            )

            basis = spline.transform(X_j)

            basis_matrices.append(basis)

        # Combine all feature basis matrices horizontally.

        B = np.hstack(basis_matrices)

        # Add intercept column.

        intercept_column = np.ones(
            (X.shape[0], 1)
        )

        design_matrix = np.hstack([
            intercept_column,
            B
        ])

        return design_matrix

    # --------------------------------------------------
    # 4. CONSTRUCT SMOOTHNESS PENALTY MATRIX
    # --------------------------------------------------

    def _build_penalty_matrix(self):

        penalty_matrices = []

        for _ in range(self.n_features_in_):

            K = self.n_splines

            # Second-order difference matrix.

            D = np.diff(
                np.eye(K),
                n=2,
                axis=0
            )

            # P = D.T @ D

            P = D.T @ D

            penalty_matrices.append(P)

        # Block diagonal penalty matrix.

        feature_penalty = block_diag(
            *penalty_matrices
        )

        # Intercept must not be penalized.

        penalty_matrix = np.zeros(
            (
                1 + self.n_features_in_ * self.n_splines,
                1 + self.n_features_in_ * self.n_splines
            )
        )

        penalty_matrix[1:, 1:] = feature_penalty

        return penalty_matrix

    # --------------------------------------------------
    # 5. FIT GAM MODEL
    # --------------------------------------------------

    def fit(self, X, y):

        X = self._validate_X(X)

        y = self._validate_y(
            y,
            X.shape[0]
        )

        self.n_features_in_ = X.shape[1]

        if self.feature_names is not None:

            if len(self.feature_names) != self.n_features_in_:
                raise ValueError(
                    "Number of feature names must match X."
                )

            self.feature_names_in_ = list(
                self.feature_names
            )

        else:

            self.feature_names_in_ = [
                f"Feature_{j + 1}"
                for j in range(self.n_features_in_)
            ]

        # Fit individual spline basis generators.

        self._fit_splines(X)

        # Construct additive design matrix.

        design_matrix = self._build_design_matrix(X)

        # Construct smoothness penalty.

        penalty_matrix = self._build_penalty_matrix()

        # Penalized least squares:
        #
        # beta = (B.T @ B + lam * P)^(-1) @ B.T @ y

        A = (
            design_matrix.T @ design_matrix
            + self.lam * penalty_matrix
        )

        b = design_matrix.T @ y

        # Solve the linear system.
        # lstsq also handles rank-deficient systems.

        try:

            beta = np.linalg.solve(A, b)

        except np.linalg.LinAlgError:

            beta = np.linalg.lstsq(
                A,
                b,
                rcond=None
            )[0]

        # Store intercept and feature coefficients.

        self.intercept_ = beta[0]

        self.coefficients_ = beta[1:]

        self.coefficients_ = self.coefficients_.reshape(
            self.n_features_in_,
            self.n_splines
        )

        self.is_fitted_ = True

        # Training predictions and diagnostics.

        self.training_predictions_ = (
            self.predict(X)
        )

        residuals = (
            y - self.training_predictions_
        )

        self.training_mse_ = np.mean(
            residuals ** 2
        )

        self.training_rmse_ = np.sqrt(
            self.training_mse_
        )

        self.training_r2_ = (
            1 - np.sum(residuals ** 2)
            / np.sum((y - np.mean(y)) ** 2)
            if np.sum((y - np.mean(y)) ** 2) > 0
            else 0.0
        )

        return self

    # --------------------------------------------------
    # 6. INDIVIDUAL FEATURE EFFECTS
    # --------------------------------------------------

    def _feature_effect(self, X, feature_index):

        spline = self.splines_[feature_index]

        X_j = np.clip(
            X[:, feature_index],
            spline.min_,
            spline.max_
        )

        basis = spline.transform(X_j)

        effect = (
            basis @ self.coefficients_[feature_index]
        )

        return effect

    # --------------------------------------------------
    # 7. PREDICTION
    # --------------------------------------------------

    def predict(self, X):

        if not self.is_fitted_:
            raise RuntimeError(
                "Fit the model before prediction."
            )

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of input features does not match training data."
            )

        predictions = np.full(
            X.shape[0],
            self.intercept_
        )

        # Add the contribution of every feature.

        for j in range(self.n_features_in_):

            predictions += self._feature_effect(
                X,
                j
            )

        return predictions

    # --------------------------------------------------
    # 8. EXPLAIN INDIVIDUAL PREDICTION
    # --------------------------------------------------

    def explain(self, x):

        if not self.is_fitted_:
            raise RuntimeError(
                "Fit the model before explanation."
            )

        x = np.asarray(x, dtype=float).reshape(1, -1)

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input feature count does not match training data."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Input must contain finite values."
            )

        explanation = {
            "intercept": float(self.intercept_),
            "feature_contributions": {},
            "prediction": None
        }

        total = self.intercept_

        for j in range(self.n_features_in_):

            effect = self._feature_effect(
                x,
                j
            )[0]

            name = self.feature_names_in_[j]

            explanation["feature_contributions"][name] = {
                "input_value": float(x[0, j]),
                "contribution": float(effect)
            }

            total += effect

        explanation["prediction"] = float(total)

        return explanation

    # --------------------------------------------------
    # 9. PLOT FEATURE EFFECTS
    # --------------------------------------------------

    def plot_effects(self, figsize=(12, 8)):

        if not self.is_fitted_:
            raise RuntimeError(
                "Fit the model before plotting."
            )

        n_features = self.n_features_in_

        fig, axes = plt.subplots(
            n_features,
            1,
            figsize=figsize,
            squeeze=False
        )

        for j in range(n_features):

            spline = self.splines_[j]

            grid = np.linspace(
                spline.min_,
                spline.max_,
                200
            )

            basis = spline.transform(grid)

            effect = (
                basis @ self.coefficients_[j]
            )

            ax = axes[j, 0]

            ax.plot(
                grid,
                effect,
                linewidth=2
            )

            ax.set_title(
                f"Effect of {self.feature_names_in_[j]}"
            )

            ax.set_xlabel(
                self.feature_names_in_[j]
            )

            ax.set_ylabel(
                "Feature contribution"
            )

            ax.grid(alpha=0.3)

        plt.tight_layout()

        plt.show()

    # --------------------------------------------------
    # 10. MODEL SUMMARY
    # --------------------------------------------------

    def summary(self):

        if not self.is_fitted_:
            raise RuntimeError(
                "Fit the model before requesting summary."
            )

        print("\nGENERALIZED ADDITIVE MODEL")
        print("-" * 45)

        print(
            f"Number of features : {self.n_features_in_}"
        )

        print(
            f"Number of splines  : {self.n_splines}"
        )

        print(
            f"Spline degree      : {self.degree}"
        )

        print(
            f"Regularization     : {self.lam}"
        )

        print(
            f"Intercept          : {self.intercept_:.4f}"
        )

        print(
            f"Training MSE       : {self.training_mse_:.4f}"
        )

        print(
            f"Training RMSE      : {self.training_rmse_:.4f}"
        )

        print(
            f"Training R2        : {self.training_r2_:.4f}"
        )

        print("-" * 45)

        print("\nFEATURES:")

        for name in self.feature_names_in_:

            print(f"  - {name}")

    # --------------------------------------------------
    # 11. STRING REPRESENTATION
    # --------------------------------------------------

    def __repr__(self):

        return (
            f"GAM(n_splines={self.n_splines}, "
            f"degree={self.degree}, "
            f"lam={self.lam})"
        )
