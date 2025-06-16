# -*- coding: utf-8 -*- noqa
"""
Created on Fri Jun  6 23:37:37 2025

@author: JoelT
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# STEP 1: Synthetic Dataset (replace this with your own)
# -----------------------------
np.random.seed(42)

# Let's assume 5 features, some float, some int, different ranges
X = np.column_stack([
    np.random.randint(0, 10, 1000),          # f1: int 0–10
    np.random.uniform(0.0, 1.0, 1000),       # f2: float 0–1
    np.random.randint(100, 200, 1000),       # f3: int 100–200
    np.random.uniform(-5.0, 5.0, 1000),      # f4: float -5 to 5
    np.random.randint(0, 3, 1000) * 10       # f5: categorical-like 0, 10, 20
])

# Simulated target with 3 classes ranked as: 0 < 1 < 2
y = np.random.choice([0, 1, 2, 4, 5], size=1000)

# -----------------------------
# STEP 2: Train Decision Tree Classifier
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# -----------------------------
# STEP 3: Generate Counterfactuals (simple method)
# -----------------------------


def generate_simple_counterfactual(x, model, class_order):
    """Try modifying one feature at a time to reach a better class"""
    current_class = model.predict([x])[0]
    current_rank = class_order[current_class]
    best_cf = None
    best_cf_class = current_class

    for i in range(len(x)):
        for delta in [-1, 1]:  # Try increasing and decreasing
            x_mod = x.copy()
            x_mod[i] += delta * (0.05 * abs(x[i]) + 0.1)  # scale adjustment
            x_mod[i] = np.clip(x_mod[i], np.min(X[:, i]), np.max(X[:, i]))
            pred_class = model.predict([x_mod])[0]
            if class_order[pred_class] > current_rank:
                best_cf = x_mod
                best_cf_class = pred_class
                return best_cf, best_cf_class  # Return first improvement found
    return None, current_class


# Map classes to ranks (e.g., 0 < 1 < 2)
class_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Generate counterfactuals for training data
X_with_cf = []
deltas = []
cf_classes = []
valid_origins = []

for xi, yi in zip(X_train, y_train):
    cf, new_class = generate_simple_counterfactual(xi, clf, class_order)
    if cf is not None:
        delta = cf - xi
        deltas.append(delta)
        X_with_cf.append(xi)
        cf_classes.append(new_class)
        valid_origins.append(yi)

deltas = np.array(deltas)
X_with_cf = np.array(X_with_cf)
cf_classes = np.array(cf_classes)

# -----------------------------
# STEP 4: Standardize deltas and Cluster Them
# -----------------------------
scaler = StandardScaler()
delta_scaled = scaler.fit_transform(deltas)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(delta_scaled)

# -----------------------------
# STEP 5: Train Meta-Classifier (from original x → change group)
# -----------------------------
group_classifier = RandomForestClassifier()
group_classifier.fit(X_with_cf, cluster_labels)

# -----------------------------
# STEP 6: Inference Function
# -----------------------------


def explain_change(x_new, model, group_clf, kmeans, scaler, class_order):
    pred_class = model.predict([x_new])[0]
    group = group_clf.predict([x_new])[0]
    delta = scaler.inverse_transform(kmeans.cluster_centers_)[group]
    new_x = x_new + delta
    new_class = model.predict([new_x])[0]

    return {
        "original_class": pred_class,
        "suggested_change": delta,
        "improved_class": new_class,
        "feature_change": dict(enumerate(delta))
    }


# -----------------------------
# Example Usage
# -----------------------------
x_sample = X_test[0]
result = explain_change(x_sample, clf, group_classifier,
                        kmeans, scaler, class_order)

print("Original class:", result["original_class"])
print("Improved class:", result["improved_class"])
print("Feature changes needed:")
for i, change in result["feature_change"].items():
    print(f"  Feature {i}: change by {change:.3f}")
