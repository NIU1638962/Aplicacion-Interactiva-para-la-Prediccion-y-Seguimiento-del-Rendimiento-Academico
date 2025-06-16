# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun  7 02:58:43 2025

@author: JoelT
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Create Synthetic Dataset (replace with real data)
# -----------------------------
np.random.seed(42)

X = np.column_stack([
    np.random.randint(0, 10, 500),
    np.random.uniform(0.0, 1.0, 500),
    np.random.randint(100, 200, 500),
    np.random.uniform(-5.0, 5.0, 500),
    np.random.randint(0, 3, 500) * 10
])
y = np.random.choice([0, 1, 2], size=500)

# -----------------------------
# STEP 2: Train Decision Tree
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# -----------------------------
# STEP 3: Generate Simple Counterfactuals
# -----------------------------


def generate_simple_counterfactual(x, model, class_order):
    current_class = model.predict([x])[0]
    current_rank = class_order[current_class]

    for i in range(len(x)):
        for delta in [-1, 1]:
            x_mod = x.copy()
            x_mod[i] += delta * (0.05 * abs(x[i]) + 0.1)
            x_mod[i] = np.clip(x_mod[i], np.min(X[:, i]), np.max(X[:, i]))
            pred_class = model.predict([x_mod])[0]
            if class_order[pred_class] > current_rank:
                return x_mod, pred_class
    return None, current_class


def find_elbow(inertias):
    # Points: (k, inertia)
    n_points = len(inertias)
    all_coords = np.array(list(zip(range(1, n_points + 1), inertias)))
    first_point = all_coords[0]
    last_point = all_coords[-1]

    # Line vector
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for point in all_coords:
        vec_from_first = point - first_point
        proj_len = np.dot(vec_from_first, line_vec_norm)
        proj_point = first_point + proj_len * line_vec_norm
        distance = np.linalg.norm(point - proj_point)
        distances.append(distance)

    return np.argmax(distances) + 1  # k starts from 1


class_order = {0: 0, 1: 1, 2: 2}

X_with_cf, deltas, cf_classes, valid_origins = [], [], [], []
for xi, yi in zip(X_train, y_train):
    cf, new_class = generate_simple_counterfactual(xi, clf, class_order)
    if cf is not None:
        delta = cf - xi
        deltas.append(delta)
        X_with_cf.append(xi)
        cf_classes.append(new_class)
        valid_origins.append(yi)

X_with_cf = np.array(X_with_cf)
deltas = np.array(deltas)

# -----------------------------
# STEP 4: Elbow Method for Clustering
# -----------------------------
scaler = StandardScaler()
delta_scaled = scaler.fit_transform(deltas)

inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(delta_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow (optional)
plt.figure()
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Choose optimal k using "elbow" heuristic
# Here we pick k with largest drop in inertia (approximate)
elbow_k = find_elbow(inertias)
print(f"Elbow detected at k = {elbow_k}")

# Final KMeans with chosen k
kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(delta_scaled)

# -----------------------------
# STEP 5: Train Meta-Classifier
# -----------------------------
group_classifier = RandomForestClassifier(n_estimators=100)
group_classifier.fit(X_with_cf, cluster_labels)

# -----------------------------
# STEP 6: Explanation & Evaluation
# -----------------------------


def explain_change(x_new, model, group_clf, kmeans, scaler, class_order):
    pred_class = model.predict([x_new])[0]
    group = group_clf.predict([x_new])[0]
    delta = scaler.inverse_transform(kmeans.cluster_centers_)[group]
    new_x = x_new + delta
    new_class = model.predict([new_x])[0]

    return {
        "original_class": pred_class,
        "improved_class": new_class,
        "feature_change": delta
    }


# Evaluation
improved, same, worsened = 0, 0, 0
rank_changes = []

for x_sample in X_test:
    result = explain_change(
        x_sample, clf, group_classifier, kmeans, scaler, class_order)
    original_rank = class_order[result["original_class"]]
    new_rank = class_order[result["improved_class"]]
    rank_diff = new_rank - original_rank
    rank_changes.append(rank_diff)

    if rank_diff > 0:
        improved += 1
    elif rank_diff == 0:
        same += 1
    else:
        worsened += 1

# Results
print("\n--- Evaluation Results ---")
print(f"Improved: {improved} ({improved / len(X_test) * 100:.1f}%)")
print(f"Stayed the same: {same} ({same / len(X_test) * 100:.1f}%)")
print(f"Worsened: {worsened} ({worsened / len(X_test) * 100:.1f}%)")
print(f"Average change in class rank: {np.mean(rank_changes):.2f}")
