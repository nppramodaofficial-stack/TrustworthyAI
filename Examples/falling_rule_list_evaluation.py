
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from trustworthy_ai.models.falling_rule_lists import FallingRuleList

# Reproducible synthetic dataset
rng = np.random.default_rng(42)

X = np.column_stack([
    rng.uniform(18, 75, size=200),
    rng.uniform(0, 1, size=200),
])

y = (
    (X[:, 0] > 45)
    | ((X[:, 0] > 35) & (X[:, 1] > 0.6))
).astype(int)
# Split before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = FallingRuleList()
model.fit(X_train, y_train, max_rules=5, min_samples=3)

predictions = model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nLearned rules:")
for i, rule in enumerate(model.rules_, start=1):
    print(f"Rule {i}: {rule}")

print("\nExample explanation:")
print(model.explain(X_test[:3]))

print("\nRule coverage on the test set:")

unmatched = np.ones(len(X_test), dtype=bool)

for i, rule in enumerate(model.rules_, start=1):
    matches = model._matches(X_test, rule) & unmatched
    print(
        f"Rule {i}: {matches.sum()} test samples, "
        f"probability={rule['probability']:.3f}"
    )
    unmatched[matches] = False

print(
    f"Default rule: {unmatched.sum()} test samples, "
    f"probability={model.default_probability_:.3f}"
)
print("\nTraining accuracy:", round(
    accuracy_score(y_train, model.predict(X_train)), 3
))

print("\nTraining predictions vs actual:")
for i in range(min(10, len(X_train))):
    print(
        f"Actual: {y_train[i]}, "
        f"Predicted: {model.predict(X_train[i:i+1])[0]}, "
        f"Features: {X_train[i]}"
    )
print("\nRule probability monotonicity:")

probabilities = [
    rule["probability"]
    for rule in model.rules_
]

probabilities.append(model.default_probability_)

print("Probabilities:", probabilities)
print(
    "Non-increasing:",
    all(
        probabilities[i] >= probabilities[i + 1]
        for i in range(len(probabilities) - 1)
    )
)
