# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun  7 03:14:44 2025

@author: JoelT
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from datasets import get_datasets


def find_elbow(inertias):
    print("[INFO] Calculating elbow point from inertias...")
    n_points = len(inertias)
    all_coords = np.array(list(zip(range(1, n_points + 1), inertias)))
    first_point = all_coords[0]
    last_point = all_coords[-1]

    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = []
    for point in all_coords:
        vec_from_first = point - first_point
        proj_len = np.dot(vec_from_first, line_vec_norm)
        proj_point = first_point + proj_len * line_vec_norm
        distance = np.linalg.norm(point - proj_point)
        distances.append(distance)

    elbow_k = np.argmax(distances) + 1
    print(f"[INFO] Elbow detected at k = {elbow_k}")
    return elbow_k


def generate_simple_counterfactual(x, model, class_order, X_data):
    current_class = model.predict([x])[0]
    current_rank = class_order[current_class]

    step_multipliers = [
        0.05,
        0.10,
        0.20,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
    ]  # try multiple step sizes

    for i in range(len(x)):
        for delta in [-1, 1]:
            for step_mul in step_multipliers:
                x_mod = x.copy()
                x_mod[i] += delta * (step_mul * abs(x[i]) + 0.1)
                x_mod[i] = np.clip(x_mod[i], np.min(
                    X_data[:, i]), np.max(X_data[:, i]))
                pred_class = model.predict([x_mod])[0]
                if class_order[pred_class] > current_rank:
                    return x_mod, pred_class
    return None, current_class


def counterfactual_clustering_evaluation(X, y, class_order, test_size=0.2, max_depth=5):
    print("[STEP] Splitting dataset and training decision tree...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    clf = RandomForestClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    print("[STEP] Generating counterfactual examples...")
    X_with_cf, deltas, cf_classes, valid_origins = [], [], [], []
    total = len(X_train)

    for idx, (xi, yi) in enumerate(zip(X_train, y_train), start=1):
        print(f"[{idx}/{total}] Processing ({(idx / total) * 100:.2f}%)", end="\r")
        cf, new_class = generate_simple_counterfactual(xi, clf, class_order, X)
        if cf is not None:
            delta = cf - xi
            deltas.append(delta)
            X_with_cf.append(xi)
            cf_classes.append(new_class)
            valid_origins.append(yi)

    if len(deltas) == 0:
        raise ValueError("No counterfactuals found for the training data.")

    X_with_cf = np.array(X_with_cf)
    deltas = np.array(deltas)

    print(f"[INFO] {len(deltas)} counterfactuals generated.")
    print("[STEP] Scaling deltas and running KMeans clustering with Elbow method...")

    scaler = StandardScaler()
    delta_scaled = scaler.fit_transform(deltas)

    inertias = []
    consecutive_threshold = 3
    tolerance_pct = 0.01  # 1% inertia change allowed (adjust as needed)
    consecutive_count = 0
    max_k = 100  # Max clusters to try to avoid infinite loops

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(delta_scaled)
        inertias.append(kmeans.inertia_)

        if k > 1:
            # Calculate relative change in inertia compared to previous k
            rel_change = abs(inertias[-2] - inertias[-1]) / inertias[-2]
            print(
                f"[INFO] k={k}, inertia={inertias[-1]:.4f}"
                + ", relative change={rel_change:.4f}",
                end="\r",
            )

            if rel_change <= tolerance_pct:
                consecutive_count += 1
                if consecutive_count >= consecutive_threshold:
                    print(
                        f"[INFO] Stopping at k={k} after"
                        + " {consecutive_threshold} consecutive low changes."
                    )
                    break
            else:
                consecutive_count = 0

        else:
            print(f"[INFO] k={k}, inertia={inertias[-1]:.4f}", end="\r")

        if k + 1 == max_k:
            print(f"[INFO] Reached max_k, stopping.")

    print("[INFO] Plotting inertia values for Elbow detection...")
    plt.figure()
    plt.plot(range(1, len(inertias) + 1), inertias, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    elbow_k = find_elbow(inertias)

    print(f"[STEP] Training final KMeans with k = {elbow_k}...")
    kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(delta_scaled)

    print("[STEP] Training group classifier on original features...")
    group_classifier = RandomForestClassifier(n_estimators=100)
    group_classifier.fit(X_with_cf, cluster_labels)

    def explain_change(x_new):
        pred_class = clf.predict([x_new])[0]
        group = group_classifier.predict([x_new])[0]
        delta = scaler.inverse_transform(kmeans.cluster_centers_)[group]
        new_x = x_new + delta
        new_class = clf.predict([new_x])[0]

        original_rank = class_order[pred_class]
        new_rank = class_order[new_class]

        # If new class is worse, do nothing (return original)
        if new_rank <= original_rank:
            return {
                "original_class": pred_class,
                "improved_class": pred_class,
                "feature_change": np.zeros_like(x_new)  # no change applied
            }
        else:
            return {
                "original_class": pred_class,
                "improved_class": new_class,
                "feature_change": delta
            }

    print("[STEP] Evaluating counterfactual suggestions on test set...")
    improved, same, worsened = 0, 0, 0
    rank_changes = []

    total = len(X_test)

    for idx, x_sample in enumerate(X_test, start=1):
        print(f"[{idx}/{total}] Processing ({(idx / total) * 100:.2f}%)", end="\r")
        result = explain_change(x_sample)
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

    print("--- Evaluation Results ---                                    ")
    print(f"Improved: {improved} ({improved / len(X_test) * 100:.1f}%)")
    print(f"Stayed the same: {same} ({same / len(X_test) * 100:.1f}%)")
    print(f"Worsened: {worsened} ({worsened / len(X_test) * 100:.1f}%)")
    print(f"Average change in class rank: {np.mean(rank_changes):.2f}")

    return clf, group_classifier, kmeans, scaler, explain_change


dataset = get_datasets('full')
X = dataset.inputs.numpy()
y = dataset.targets.numpy()
class_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # Or your real class priority
counterfactual_clustering_evaluation(X, y, class_order, max_depth=None)
