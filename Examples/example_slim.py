
import numpy as np

from trustworthy_ai.models import SLIM


# --------------------------------
# 1. CREATE DATASET
# --------------------------------

X = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
])

y = np.array([
    1, 1, -1, -1, 1,
    -1, 1, -1, 1, -1
])

feature_names = [
    "Age_Risk",
    "Family_History",
    "Lifestyle_Risk"
]


# --------------------------------
# 2. INITIALIZE MODEL
# --------------------------------

model = SLIM(
    C0=0.01,
    coefficient_bound=5,
    fit_intercept=True,
    time_limit=120,
    feature_names=feature_names
)


# --------------------------------
# 3. TRAIN
# --------------------------------

model.fit(X, y)


# --------------------------------
# 4. MODEL SUMMARY
# --------------------------------

model.summary()


# --------------------------------
# 5. PREDICTIONS
# --------------------------------

predictions = model.predict(X)

print("\nPredictions:")
print(predictions)


# --------------------------------
# 6. INTEGER SCORES
# --------------------------------

scores = model.decision_function(X)

print("\nDecision Scores:")
print(scores)


# --------------------------------
# 7. EXPLANATION
# --------------------------------

explanation = model.explain(
    X,
    sample_index=0
)

print("\nExplanation:")
print(explanation)


# --------------------------------
# 8. SCORECARD
# --------------------------------

print("\nScorecard:")
print(model.get_scorecard())


# --------------------------------
# 9. ACCURACY
# --------------------------------

accuracy = np.mean(predictions == y)

print("\nAccuracy:", accuracy)
