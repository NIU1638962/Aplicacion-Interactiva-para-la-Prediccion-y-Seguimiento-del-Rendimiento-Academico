# -*- coding: utf-8 -*- noqa
"""
Created on Sun Jun 15 16:02:29 2025

@author: JoelT
"""
import environment
import utils

from analize_mappings import analyze_mapping
from datasets import get_datasets, train_test_split
import models_sklearn


def optimize_modifiable_parameters(
    x,
    model,
    class_order,
    modifiable_features_info,
):
    best_possible_class = max(class_order.items(), key=lambda x: x[1])[0]
    x = environment.numpy.array(x)
    original_class = environment.torch.tensor(
        environment.numpy.clip(
            environment.numpy.round(model.predict([x])),
            0,
            environment.NUMBER_GRADES - 1,
        ).astype('int64'),
    ).item()
    original_rank = class_order[original_class]
    best_x = environment.deepcopy(x)
    best_class = original_class
    best_rank = original_rank
    best_changes = {
        feature_name: ('No change' if best_x[feature_info['index']] == x[feature_info['index']] else f'Change to {best_x[feature_info["index"]]}') for feature_name, feature_info in modifiable_features_info.items()
    }

    not_known_counter = 0

    if original_class != best_possible_class:
        for feature_name, feature_info in modifiable_features_info.items():
            if x[feature_info['index']] == feature_info['not_known']:
                not_known_counter += 1
                continue

            for value in feature_info['possible_values']:
                x_mod = environment.deepcopy(x)
                x_mod[feature_info['index']] = value

                new_class = environment.torch.tensor(
                    environment.numpy.clip(
                        environment.numpy.round(model.predict([x_mod])),
                        0,
                        environment.NUMBER_GRADES - 1,
                    ).astype('int64'),
                ).item()

                new_rank = class_order[new_class]

                if new_rank > best_rank:
                    best_class = new_class
                    best_rank = new_rank
                    best_x = environment.deepcopy(x_mod)
                    best_changes = {
                        feature_name: ('No change' if best_x[feature_info['index']] == x[feature_info['index']] else f'Change to {best_x[feature_info["index"]]}') for feature_name, feature_info in modifiable_features_info.items()
                    }

    return {
        "original_class": original_class,
        "improved_class": best_class,
        "modified_x": best_x,
        "feature_changes": best_changes,
        "rank": best_rank,
        "no_change_possible": not_known_counter == len(modifiable_features_info),
        "best_class_already": original_class == best_possible_class,
    }


def prepare_features(
    raw_mapping,
    float_range_sample,
    availbale_feature_names,
    modifiable_feature_names,

):
    analyzed_mapping = analyze_mapping(raw_mapping, float_range_sample)

    features_info = {}

    print("\n--- Parameter Range Summary ---")
    for index, feature_name in enumerate(availbale_feature_names):
        info = analyzed_mapping.get(feature_name, None)
        if info is None:
            print(f"{feature_name}: MISSING from mapping.")
            continue

        features_info[feature_name] = environment.deepcopy(info)
        features_info[feature_name]['index'] = index

        print(
            f"{feature_name} — min: {info['min']}, max: {info['max']}"
            + f", not known: {info['not_known']}"
        )

    modifiable_features_info = {
        modifiable_feature_name: environment.deepcopy(features_info[modifiable_feature_name]) for modifiable_feature_name in modifiable_feature_names if features_info.get(modifiable_feature_name, False)
    }

    return modifiable_features_info


def test_recomendator(
    raw_mapping,
    float_range_sample,
    dataset,
    modifiable_feature_names,
    model,
    class_order,
):
    modifiable_features_info = prepare_features(
        raw_mapping,
        float_range_sample,
        dataset.feature_names,
        modifiable_feature_names,
    )

    improved = same = worsened = no_change_possible = best_class_already = 0
    rank_changes = []
    results = []
    total = len(dataset)

    print("\n[STEP] Evaluating individual counterfactual modifications...")
    for idx, x in enumerate(dataset.inputs, start=1):
        print(f"[{idx}/{total}] Processing ({idx/total*100:.2f}%)", end="\r")
        result = optimize_modifiable_parameters(
            x,
            model,
            class_order,
            modifiable_features_info,
        )

        results.append(result)

        if result['best_class_already']:
            best_class_already += 1
            continue

        if result['no_change_possible']:
            no_change_possible += 1
            continue

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

    # — Summary —
    print("\n--- Evaluation Results ---")
    print(
        f"Best class already: {best_class_already}"
        + f" ({best_class_already/total*100:.1f}%)"
    )
    print(
        f"No change possible: {no_change_possible}"
        + f" ({no_change_possible/total*100:.1f}%)"
    )
    print(f"Improved: {improved} ({improved/total*100:.1f}%)")
    print(f"Stayed the same: {same} ({same/total*100:.1f}%)")
    print(f"Worsened: {worsened} ({worsened/total*100:.1f}%)")
    print(
        "Average change in class rank with possible change:"
        + f" {environment.numpy.mean(rank_changes):.2f}"
    )

    return results


if __name__ == "__main__":
    environment.NUMBER_GRADES = 5

    raw_mapping = utils.load_json(
        environment.os.path.join(
            environment.DATA_PATH,
            'general_mapping.json',
        ),
    )

    # — Load dataset —
    dataset = get_datasets('full')

    # — Train model —
    model = models_sklearn.models['random_forest_classifier']()
    model.fit(dataset.inputs, dataset.targets)

    # — Create class order map —
    class_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    float_range_sample = [0, 0.25, 0.5, 0.75, 1]

    modifiable_feature_names = [
        'absences',
        'additional_work',
        'alcohol_weekend',
        'alcohol_workday',
        'attendance_classes',
        'attendance_seminars',
        # 'attendance_time',
        'extra_curricular_activities',
        'free_time',
        'go_out',
        'listening_classes',
        'personal_classes',
        'reading_frequency_non_scientific',
        'reading_frequency_scientific',
        'taking_notes_classes',
        'weekly_study_hours',
    ]

    results = test_recomendator(
        raw_mapping,
        float_range_sample,
        dataset,
        modifiable_feature_names,
        model,
        class_order,
    )
