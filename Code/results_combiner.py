# -*- coding: utf-8 -*- noqa
"""
Created on Sun Jun 15 23:05:28 2025

@author: JoelT
"""
import environment
import utils

# Load the CSV files
means = utils.load_csv(
    environment.os.path.join(
        environment.RESULTS_PATH,
        'experiment_2_metrics_mean.csv',
    ),
    index_column=0,
)
stds = utils.load_csv(
    environment.os.path.join(
        environment.RESULTS_PATH,
        'experiment_2_metrics_std.csv',
    ),
    index_column=0,
)

# Load the hash-to-name mapping
lookup = utils.load_csv(
    environment.os.path.join(
        environment.RESULTS_PATH,
        'experiment_1_look_up_names.csv',
    ),
)
hash_to_name = dict(zip(lookup['hash_name'], lookup['name']))

# Rename the index (model hashes) to their readable names
means.index = means.index.map(hash_to_name)
stds.index = stds.index.map(hash_to_name)

# Combine into "mean ± std" format
combined = means.copy()
for col in means.columns:
    if col in ['accuracy']:
        combined[col] = (
            (means[col].round(3) * 100).astype(str) + "% ± "
            + (stds[col].round(3) * 100).astype(str) + "%"
        )
    else:
        combined[col] = (
            means[col].round(3).astype(str) + " ± "
            + stds[col].round(3).astype(str)
        )

# Save the result to a new CSV
utils.save_csv(
    combined,
    environment.os.path.join(
        environment.RESULTS_PATH,
        'experiment_2_combined_metrics.csv',
    ),
    index=True,
)

# Optional: display as a markdown-style table in console
# print(combined.to_markdown())
