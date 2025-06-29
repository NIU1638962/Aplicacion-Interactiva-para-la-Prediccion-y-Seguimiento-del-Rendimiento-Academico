# -*- coding: utf-8 -*- noqa
"""
Created on Sun Jun 15 15:36:43 2025

@author: JoelT
"""
# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun 14 14:12:50 2025
@author: JoelT

✔️ Now prints min, max, and 'not known' value for each parameter
"""




import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datasets import get_datasets  # your custom dataset loader
def optimize_modifiable_parameters(x, model, class_order, modifiable_params, param_ranges, unknown_values, step_sizes=[0.05, 0.1, 0.2]):
    x = np.array(x)
    original_class = model.predict([x])[0]
    original_rank = class_order[original_class]
    best_x = x.copy()
    best_rank = original_rank
    best_delta = np.zeros_like(x)

    for i in modifiable_params:
        if x[i] == unknown_values[i]:
            continue

        min_val, max_val = param_ranges[i]
        if min_val == -1 or max_val == -1:
            continue

        for step in step_sizes:
            for direction in (-1, 1):
                x_mod = x.copy()
                change = direction * step * (abs(x[i]) + 0.1)
                x_mod[i] = np.clip(x[i] + change, min_val, max_val)

                new_class = model.predict([x_mod])[0]
                new_rank = class_order[new_class]

                if new_rank > best_rank:
                    best_rank = new_rank
                    best_x = x_mod
                    best_delta = x_mod - x

    return {
        "original_class": original_class,
        "improved_class": model.predict([best_x])[0],
        "modified_x": best_x,
        "feature_changes": best_delta
    }


if __name__ == "__main__":
    # — Load mapping JSON —
    with open("..\\Data\\general_mapping.json", "r", encoding="utf-8") as f:
        general_mapping = json.load(f)

    # — Load dataset —
    dataset = get_datasets('full')
    X = dataset.inputs.numpy()
    y = dataset.targets.numpy()
    feature_names = dataset.feature_names

    # — Prepare parameter ranges and unknown constants —
    param_ranges = []
    unknown_values = []

    print("\n--- Parameter Range Summary ---")
    for i, fname in enumerate(feature_names):
        mapping = general_mapping.get(fname, None)
        if mapping is None:
            param_ranges.append((-1, -1))
            unknown_values.append(-1)
            print(f"{i}: {fname} — min: -1, max: -1, not known: -1 (MISSING in JSON)")
            continue

        unknown = mapping.get("not known", -1)
        unknown_values.append(unknown)

        if "grade" in mapping:
            rmin, rmax = mapping["grade"]
        elif "age" in mapping:
            rmin, rmax = mapping["age"]
        elif "number" in mapping:
            rmin, rmax = mapping["number"]
        elif "target" in mapping:
            rmin, rmax = mapping["target"]
        else:
            values = [v for k, v in mapping.items() if k != "not known"]
            rmin, rmax = (min(values), max(values)) if values else (-1, -1)

        param_ranges.append((rmin, rmax))
        print(f"{i}: {fname} — min: {rmin}, max: {rmax}, not known: {unknown}")

    # — Define modifiable parameters —
    modifiable_params = [i for i in range(X.shape[1]) if i not in [1, 3]]

    # — Train/test split —
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # — Train the classifier —
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # — Define class ranking —
    class_order = {cls: cls for cls in np.unique(y)}

    # — Evaluate optimization —
    improved = same = worsened = 0
    rank_changes = []
    total = len(X_test)

    print("\n[STEP] Evaluating individual counterfactual modifications...")
    for idx, x in enumerate(X_test, start=1):
        print(f"[{idx}/{total}] Processing ({idx/total*100:.2f}%)", end="\r")
        result = optimize_modifiable_parameters(
            x, model, class_order,
            modifiable_params, param_ranges, unknown_values
        )
        orig = class_order[result["original_class"]]
        new = class_order[result["improved_class"]]
        diff = new - orig
        rank_changes.append(diff)

        if diff > 0:
            improved += 1
        elif diff == 0:
            same += 1
        else:
            worsened += 1

    # — Print summary —
    print("\n--- Evaluation Results ---")
    print(f"Improved: {improved} ({improved/total*100:.1f}%)")
    print(f"Stayed the same: {same} ({same/total*100:.1f}%)")
    print(f"Worsened: {worsened} ({worsened/total*100:.1f}%)")
    print(f"Average change in class rank: {np.mean(rank_changes):.2f}")
