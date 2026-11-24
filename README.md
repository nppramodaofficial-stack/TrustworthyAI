# TrustworthyAI

### An Open-Source Python Library for Interpretable and Trustworthy Machine Learning

**TrustworthyAI** is a Python library dedicated to the implementation, exploration, and development of interpretable machine learning algorithms. The project aims to bridge the gap between theoretical foundations of interpretable machine learning and practical algorithmic implementations.

Rather than treating explainability as an afterthought, TrustworthyAI focuses on models whose internal structures, decision mechanisms, and feature contributions can be understood and examined.

The library is being developed with an emphasis on mathematical transparency, modular implementation, reproducibility, and extensibility.

---

## 1. Project Overview

Machine learning models are increasingly being applied in domains where understanding how predictions are generated is just as important as predictive performance.

Traditional machine learning approaches, particularly complex ensemble methods and deep neural networks, can make it difficult to understand the reasoning behind individual predictions.

TrustworthyAI explores an alternative approach by implementing interpretable models that incorporate transparency directly into their architecture.

The project focuses on:

- Implementing interpretable machine learning algorithms from their mathematical foundations.
- Understanding the optimization procedures and assumptions underlying each algorithm.
- Developing modular and reusable implementations.
- Providing feature-level and prediction-level explanations.
- Validating implementations through unit testing.
- Building a foundation for future research in trustworthy and explainable artificial intelligence.

The long-term objective is to develop a comprehensive Python library covering interpretable models, explanation methods, feature selection, uncertainty quantification, and other components of trustworthy machine learning.

## 2. Current Implementations

The library currently includes three principal model implementations.

### 2.1 Falling Rule Lists (FRL)

**Category:** Rule-based interpretable machine learning

Falling Rule Lists are interpretable predictive models that represent decisions through an ordered sequence of logical rules.

The model assigns observations to the first rule they satisfy and associates each rule with a corresponding prediction probability.

The falling structure imposes a monotonically decreasing ordering on the probabilities associated with successive rules.

**Mathematical concepts:**
- Ordered logical rules.
- First-match rule assignment.
- Conditional probability estimation.
- Monotonicity constraints.
- Weighted Pool Adjacent Violators Algorithm (PAVA).
- Simulated annealing for rule-search optimization.

**Implemented functionality:**
- Conjunction-based rule generation.
- Rule ordering and assignment.
- Monotonic probability estimation.
- Prediction and probability estimation.
- Rule extraction and individual explanations.

**Implementation note:** The current implementation uses a Bayesian-inspired heuristic approach. It is not an exact reproduction of the complete Bayesian Falling Rule Lists algorithm proposed in the original research.

### 2.2 Supersparse Linear Integer Models (SLIM)

**Category:** Sparse linear scoring systems

SLIM constructs interpretable scoring systems using integer-valued coefficients. These models are particularly useful when predictions need to be expressed through simple, human-readable scoring rules.

The core optimization objective is:

\[
\min_{\lambda}
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}[\hat y_i\neq y_i]
+
C_0\|\lambda\|_0
\]

Subject to integer coefficient constraints and a coprimality requirement.

The objective balances predictive errors against model sparsity.

**Mathematical concepts:**
- Integer-valued optimization.
- L0 sparsity regularization.
- Mixed-integer linear programming.
- Coefficient constraints.
- Coprime coefficient vectors.
- Sparse scoring systems.

**Implemented functionality:**
- Integer coefficient estimation.
- Mixed-integer optimization using SciPy.
- Feature selection through binary optimization variables.
- Optional intercept.
- Coefficient-bound constraints.
- Coprime coefficient normalization.
- Prediction and decision scores.
- Individual feature contribution explanations.
- Human-readable scorecard generation.

**Implementation note:** The current implementation is a practical SLIM-style mixed-integer optimization approach. Coprimality is applied through post-solution normalization rather than encoded directly into the optimization problem. It should not be interpreted as a complete reproduction of every optimization detail in the original paper.

### 2.3 Generalized Additive Models (GAM)

**Category:** Interpretable nonlinear machine learning

Generalized Additive Models extend linear models by allowing individual features to contribute through learned functions rather than restricting their effects to linear relationships.

The model is represented as:

\[
g(E[Y\mid X]) =
\beta_0+\sum_{j=1}^{p}f_j(X_j)
\]

Each feature contributes independently to the final prediction through its corresponding learned function.

The current implementation focuses on Gaussian regression using cubic B-spline basis functions and penalized least squares.

**Mathematical concepts:**
- Additive model architecture.
- B-spline basis functions.
- Cox–de Boor recursion.
- Cubic spline representations.
- Second-order difference penalties.
- Penalized least squares.
- Smoothness regularization.

**Implemented functionality:**
- B-spline basis generation.
- Additive design matrix construction.
- Feature-wise spline estimation.
- Smoothness regularization.
- Regression prediction.
- Individual feature contribution explanations.
- Feature-effect visualization.
- Training performance diagnostics.

**Implementation note:** The current GAM implementation is a Gaussian regression model. It uses a second-order coefficient-difference penalty and is still undergoing mathematical refinement, including feature-effect centering and numerical stability validation.

---

## 3. Mathematical Foundations

A central principle of TrustworthyAI is that algorithmic implementations should be supported by an understanding of their underlying mathematics.

The project therefore emphasizes the following concepts:

| Area | Mathematical concepts |
|---|---|
| Rule-based models | Logical conjunctions, conditional probabilities, monotonicity |
| Sparse optimization | L0 regularization, integer programming, mixed-integer optimization |
| Additive models | Function approximation, spline basis functions, penalized least squares |
| Model interpretation | Feature contributions, decision scores, rule-based explanations |
| Optimization | Constraint formulation, regularization, numerical optimization |

Each model is developed with attention to its objective function, assumptions, constraints, optimization strategy, and interpretability properties.

## 4. Technology Stack

The project is developed using Python and its scientific computing ecosystem.

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | Numerical computing and matrix operations |
| SciPy | Optimization, linear algebra, and spline-related computations |
| Matplotlib | Visualization of learned feature effects |
| Pytest | Unit testing |
| Git | Version control |
| GitHub | Source code management and collaboration |
| PyCharm | Development environment |

## 5. Project Structure

The project follows a modular `src`-layout Python package structure.

```text
TrustworthyAI/
│
├── src/
│   └── trustworthy_ai/
│       │
│       ├── __init__.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── falling_rule_lists.py
│           ├── slim.py
│           ├── gam.py
│           └── splines.py
│
├── tests/
│   ├── test_falling_rule_lists.py
│   ├── test_slim.py
│   ├── test_gam.py
│   └── test_splines.py
│
├── examples/
│   ├── example_slim.py
│   └── example_gam.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

### Architectural Design

The library follows a modular design:

- **models/** contains individual interpretable machine learning algorithms.
- **splines.py** provides reusable B-spline basis functionality.
- **tests/** contains unit tests for model behaviour and mathematical properties.
- **examples/** provides practical demonstrations of the implementations.
- **pyproject.toml** manages package metadata and dependencies.

This structure allows new algorithms and supporting components to be incorporated without disrupting existing implementations.

## 6. Installation

### Clone the repository

```bash
git clone https://github.com/nppramodaofficial-stack/TrustworthyAI.git
```

```bash
cd TrustworthyAI
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Install the package

```bash
pip install -e .
```

The project uses the dependencies specified in `pyproject.toml`.

## 7. Usage

The following examples demonstrate the intended model interfaces.

### 7.1 SLIM

```python
import numpy as np

from trustworthy_ai.models.slim import SLIM

X = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1]
])

y = np.array([1, -1, 1, -1, 1, -1])

model = SLIM(
    C0=0.01,
    coefficient_bound=10,
    feature_names=["Age_Risk", "Health_Risk"]
)

model.fit(X, y)

predictions = model.predict(X)

print(model.summary())
print(model.get_scorecard())
```

### 7.2 GAM

```python
import numpy as np

from trustworthy_ai.models.gam import GAM

X = np.random.uniform(0, 10, (100, 2))

y = (
    2 * np.sin(X[:, 0])
    + 0.5 * X[:, 1] ** 2
    + 3
)

model = GAM(
    n_splines=10,
    degree=3,
    lam=1.0,
    feature_names=["Feature_1", "Feature_2"]
)

model.fit(X, y)

predictions = model.predict(X)

print(model.summary())

explanation = model.explain(X[0])

print(explanation)

model.plot_effects()
```

### 7.3 Falling Rule Lists

The Falling Rule Lists implementation provides a rule-based interface for fitting, predicting, and examining learned rules.

Refer to the model implementation and its example for the current constructor parameters and supported methods.

---

## 8. Testing and Validation

Testing is an important part of the development process.

Each implementation includes unit tests designed to validate:

- Model initialization.
- Input validation.
- Model fitting.
- Prediction behaviour.
- Output dimensions.
- Explanation consistency.
- Mathematical properties and constraints.
- Handling of invalid inputs.
- Model-specific edge cases.

The test suite is executed using Pytest.

Run the complete test suite:

```bash
pytest tests/ -v
```

### Current Test Results

The current development version has successfully passed all 53 collected unit tests.

| Component | Tests passed |
|---|---:|
| Falling Rule Lists | 6 |
| SLIM | 19 |
| GAM | 21 |
| B-spline basis | 7 |
| **Total** | **53** |

These results demonstrate successful execution of the implemented test cases. They do not independently establish equivalence with the original research algorithms or guarantee performance across all datasets.

---

## 9. Development Philosophy

TrustworthyAI is being developed around five principles.

**1. Mathematical transparency**

Implementations should be supported by an understanding of their mathematical formulation, assumptions, and optimization procedures.

**2. Interpretability by design**

Preference is given to models whose internal structures and prediction mechanisms can be examined directly.

**3. Modular architecture**

Each algorithm should be independently implementable, testable, and reusable.

**4. Reproducibility**

Code, tests, examples, and documentation should make the development process understandable and reproducible.

**5. Research-oriented development**

The library is intended to serve as a foundation for experimentation, algorithmic exploration, and future research in trustworthy machine learning.

---

## 10. Roadmap

The project is under active development. The following algorithms and components are planned for future implementation.

### Interpretable Models

- [x] Falling Rule Lists (FRL)
- [x] Supersparse Linear Integer Models (SLIM)
- [x] Generalized Additive Models (GAM)
- [ ] Generalized Additive Models with Pairwise Interactions (GA2M)
- [ ] Explainable Boosting Machines (EBM)
- [ ] Varying Coefficient Models (VCM)
- [ ] Self-Explaining Neural Networks (SENN)
- [ ] Contextual Explanation Networks (CEN)

### Future Research and Development Areas

- Feature selection and statistical interpretability.
- Explainable AI methods.
- Counterfactual explanations.
- Uncertainty quantification.
- Model robustness and reliability.
- Explanation stability.
- Interpretability under distribution shifts.
- Evaluation of explanation quality.

These are planned directions rather than currently implemented capabilities.

---

## 11. Research References

The following foundational research works inform the current implementations.

**Falling Rule Lists**

Wang, T., & Rudin, C. (2015). *Falling Rule Lists*. Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS).

https://proceedings.mlr.press/v38/wang15a.html

**SLIM**

Ustun, B., & Rudin, C. (2016). *Supersparse Linear Integer Models for Optimized Medical Scoring Systems*. Machine Learning, 102, 349–391.

https://arxiv.org/abs/1502.04269

**Generalized Additive Models**

Hastie, T. J., & Tibshirani, R. J. (1990). *Generalized Additive Models*. Chapman and Hall/CRC.

The spline-based implementation also draws on the mathematical foundations of B-spline basis functions and penalized regression.

---

## 12. Project Status

TrustworthyAI is an ongoing open-source development project.

The current focus is on strengthening existing implementations, improving numerical and mathematical validation, expanding documentation, and progressively incorporating additional interpretable machine learning algorithms.

The project is intended as both a practical software library and a long-term learning and research initiative in interpretable and trustworthy artificial intelligence.

---

## 13. License

A license has not yet been selected for this project.

Until a license is added to the repository, reuse and redistribution should not be assumed to be permitted.

---

**TrustworthyAI — Building machine learning models that are not only predictive, but understandable.**
