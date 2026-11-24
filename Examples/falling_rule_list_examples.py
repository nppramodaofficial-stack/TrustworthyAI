
import numpy as np

from trustworthy_ai.models.falling_rule_lists import FallingRuleList


# Two numeric features and a binary target.
X = np.array([
    [20, 0],
    [25, 0],
    [30, 1],
    [35, 1],
    [40, 0],
    [45, 1],
    [50, 1],
    [55, 1],
    [60, 0],
    [65, 1],
    [70, 1],
    [75, 1],
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

model = FallingRuleList()
model.fit(X, y, max_rules=3, min_samples=2)

print("Learned rules:")
for rule in model.rules_:
    print(rule)

print("\nDefault probability:", model.default_probability_)
print("\nPredictions:", model.predict(X))
print("\nProbabilities:\n", model.predict_proba(X))
print("\nExplanations:")
for explanation in model.explain(X):
    print(explanation)
