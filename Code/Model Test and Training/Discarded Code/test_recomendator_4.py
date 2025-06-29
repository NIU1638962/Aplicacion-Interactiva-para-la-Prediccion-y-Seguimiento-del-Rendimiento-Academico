# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun 14 14:12:50 2025

@author: JoelT
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datasets import get_datasets  # your custom dataset loader

# === Insert the new optimization function here ===


def optimize_modifiable_parameters(x, model, class_order, modifiable_params, param_ranges, step_sizes=[0.05, 0.1, 0.2]):
    x = np.array(x)
    original_class = model.predict([x])[0]
    original_rank = class_order[original_class]
    best_x = x.copy()
    best_rank = original_rank
    best_delta = np.zeros_like(x)

    for i in modifiable_params:
        if x[i] == -1:
            continue  # Skip parameters that are not available

        min_val, max_val = param_ranges[i]
        if min_val == -1 or max_val == -1:
            continue  # Skip invalid range

        for step in step_sizes:
            for direction in [-1, 1]:
                x_modified = x.copy()
                change = direction * step * (abs(x[i]) + 0.1)
                x_modified[i] = np.clip(x[i] + change, min_val, max_val)

                new_class = model.predict([x_modified])[0]
                new_rank = class_order[new_class]

                if new_rank > best_rank:
                    best_rank = new_rank
                    best_x = x_modified
                    best_delta = x_modified - x

    return {
        "original_class": original_class,
        "improved_class": model.predict([best_x])[0],
        "modified_x": best_x,
        "feature_changes": best_delta
    }


# === Load and prepare dataset ===
dataset = get_datasets('full')
X = dataset.inputs.numpy()
y = dataset.targets.numpy()

# === Define class order (example) ===
class_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # lower = worse, higher = better

# === Split into train/test for model training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# === Train the classifier ===
model = RandomForestClassifier()
model.fit(X_train, y_train)

# === Define which parameters are modifiable ===
# Example: avoid index 1 and 3 (e.g., categorical)
modifiable_params = [i for i in range(X.shape[1]) if i not in [1, 3]]

# === Define range for each parameter ===
# You can also compute from X_train, but here’s a static example:
param_ranges = []
for i in range(X.shape[1]):
    values = X[:, i]
    valid = values[values != -1]
    if len(valid) == 0:
        param_ranges.append((-1, -1))
    else:
        param_ranges.append((np.min(valid), np.max(valid)))

# === Run optimization on test set ===
improved = 0
same = 0
worsened = 0
rank_changes = []

print("[STEP] Evaluating individual counterfactual modifications...")
total = len(X_test)

for idx, x in enumerate(X_test, start=1):
    print(f"[{idx}/{total}] Processing ({(idx / total) * 100:.2f}%)", end="\r")
    result = optimize_modifiable_parameters(
        x,
        model=model,
        class_order=class_order,
        modifiable_params=modifiable_params,
        param_ranges=param_ranges
    )
    original_rank = class_order[result["original_class"]]
    new_rank = class_order[result["improved_class"]]
    diff = new_rank - original_rank
    rank_changes.append(diff)

    if diff > 0:
        improved += 1
    elif diff == 0:
        same += 1
    else:
        worsened += 1

# === Print summary ===
print("\n--- Evaluation Results ---")
print(f"Improved: {improved} ({improved / total * 100:.1f}%)")
print(f"Stayed the same: {same} ({same / total * 100:.1f}%)")
print(f"Worsened: {worsened} ({worsened / total * 100:.1f}%)")
print(f"Average change in class rank: {np.mean(rank_changes):.2f}")
