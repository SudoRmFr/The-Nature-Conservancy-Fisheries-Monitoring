import csv
import os
from datetime import datetime

import numpy as np

os.chdir("???")

# === Set use what prediction files to vote ===

csv_files = [
    # 1.9 is an example of its submit score (used to calculate the weight),
    # this value should be in 1.65163 ~ 2.07944154168. You can modify the range limit.
    ['path to prediction csv file 1', 1.9],
    ['path to prediction csv file 2', 1.88],
]

# Sum of all csv
names_to_prob = {}
for csv_file, score in csv_files:
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            name = row[0]
            probs = np.array(list(map(float, row[1:])))
            # === weight === Modify the way to calculate weights
            exp_max = 2.07944154168
            exp_min = 1.65163
            assert exp_min <= score <= exp_max
            weight = (exp_max - score) / (exp_max - exp_min)
            probs *= 1.0000001 + weight * 2
            """
            Training images
            -----------
            ALB    1719 (0.455)
            SHARK   176 (0.047)
            LAG      67 (0.018)
            YFT     734 (0.194)
            NoF     465 (0.123)
            OTHER   299 (0.079)
            DOL     117 (0.031)
            BET     200 (0.053)
            """
            if name in names_to_prob:
                names_to_prob[name] += probs
            else:
                names_to_prob[name] = probs

# Normalization
for name in names_to_prob:
    sum_p = np.sum(names_to_prob[name])
    names_to_prob[name] /= sum_p

# Manually give probs
for name in names_to_prob:
    probs = names_to_prob[name]
    # prob_and_idx = sorted([[prob, i] for i, prob in enumerate(probs)], key=lambda x: x[0] if x[1] != 0 else -1, reverse=True)
    prob_and_idx = sorted([[prob, i] for i, prob in enumerate(probs)], key=lambda x: x[0], reverse=True)
    new_probs = [0., 0., 0., 0., 0., 0., 0., 0.]

    # Use top-1-probability-rule
    # new_probs[prob_and_idx[0][1]] = 0.25
    # new_probs[prob_and_idx[1][1]] = 0.25
    # new_probs[prob_and_idx[2][1]] = 0.25
    # new_probs[prob_and_idx[3][1]] = 0.05
    # new_probs[prob_and_idx[4][1]] = 0.05
    # new_probs[prob_and_idx[5][1]] = 0.05
    # new_probs[prob_and_idx[6][1]] = 0.05
    # new_probs[prob_and_idx[7][1]] = 0.05

    # Modified version
    # new_probs[prob_and_idx[0][1]] = 0.30
    # new_probs[prob_and_idx[1][1]] = 0.25
    # new_probs[prob_and_idx[2][1]] = 0.20
    # new_probs[prob_and_idx[3][1]] = 0.08
    # new_probs[prob_and_idx[4][1]] = 0.08
    # new_probs[prob_and_idx[5][1]] = 0.05
    # new_probs[prob_and_idx[6][1]] = 0.02
    # new_probs[prob_and_idx[7][1]] = 0.02

    # Modified version
    # new_probs[prob_and_idx[0][1]] = 0.35
    # new_probs[prob_and_idx[1][1]] = 0.30
    # new_probs[prob_and_idx[2][1]] = 0.10
    # new_probs[prob_and_idx[3][1]] = 0.08
    # new_probs[prob_and_idx[4][1]] = 0.08
    # new_probs[prob_and_idx[5][1]] = 0.05
    # new_probs[prob_and_idx[6][1]] = 0.02
    # new_probs[prob_and_idx[7][1]] = 0.02

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.0225
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.0225
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.0225
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.0225
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.0225
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.0225
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.0225
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.0225

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.075
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.075
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.075
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.075
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.025
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.025
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.025
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.025

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.10
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.10
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.075
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.075
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.05
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.05
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.025
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.025

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.1000
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.0875
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.0750
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.0625
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.0500
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.0375
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.0250
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.0125

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.1000 * 2
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.0875 * 2
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.0750 * 2
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.0625 * 2
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.0500 * 2
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.0375 * 2
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.0250 * 2
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.0125 * 2

    # Other version
    # new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.1000 * 1.25
    # new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.0875 * 1.25
    # new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.0750 * 1.25
    # new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.0625 * 1.25
    # new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.0500 * 1.25
    # new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.0375 * 1.25
    # new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.0250 * 1.25
    # new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.0125 * 1.25

    # Other version
    new_probs[prob_and_idx[0][1]] = probs[prob_and_idx[0][1]] + 0.1000 * 1.28
    new_probs[prob_and_idx[1][1]] = probs[prob_and_idx[1][1]] + 0.0875 * 1.27
    new_probs[prob_and_idx[2][1]] = probs[prob_and_idx[2][1]] + 0.0750 * 1.26
    new_probs[prob_and_idx[3][1]] = probs[prob_and_idx[3][1]] + 0.0625 * 1.25
    new_probs[prob_and_idx[4][1]] = probs[prob_and_idx[4][1]] + 0.0500 * 1.24
    new_probs[prob_and_idx[5][1]] = probs[prob_and_idx[5][1]] + 0.0375 * 1.23
    new_probs[prob_and_idx[6][1]] = probs[prob_and_idx[6][1]] + 0.0250 * 1.22
    new_probs[prob_and_idx[7][1]] = probs[prob_and_idx[7][1]] + 0.0125 * 1.21

    # Other version
    # new_probs[prob_and_idx[0][1]] = 0.32
    # new_probs[prob_and_idx[1][1]] = 0.27
    # new_probs[prob_and_idx[2][1]] = 0.21
    # new_probs[prob_and_idx[3][1]] = 0.10
    # new_probs[prob_and_idx[4][1]] = 0.04
    # new_probs[prob_and_idx[5][1]] = 0.03
    # new_probs[prob_and_idx[6][1]] = 0.02
    # new_probs[prob_and_idx[7][1]] = 0.01

    # Use top-1-probability-rule
    # new_probs[prob_and_idx[0][1]] = 0.507
    # new_probs[prob_and_idx[1][1]] = 0.07042857142
    # new_probs[prob_and_idx[2][1]] = 0.07042857142
    # new_probs[prob_and_idx[3][1]] = 0.07042857142
    # new_probs[prob_and_idx[4][1]] = 0.07042857142
    # new_probs[prob_and_idx[5][1]] = 0.07042857142
    # new_probs[prob_and_idx[6][1]] = 0.07042857142
    # new_probs[prob_and_idx[7][1]] = 0.07042857142

    new_probs = np.array(new_probs)

    new_probs /= sum(new_probs)

    # === Comment the following line to not manually modified probabilities ===
    names_to_prob[name] = np.array(new_probs)

# Output
os.makedirs('./pred', exist_ok=True)
output_path = os.path.join('./pred', "agg_{}.csv".format(datetime.now().strftime("UTC+8_%Y_%m-%d_%H:%M")))
with open(output_path, "w+") as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    for name in names_to_prob:
        writer.writerow([name] + names_to_prob[name].tolist())
print("Aggregate the following files done:")
for csv_file in csv_files:
    print(csv_file)
print(f"Saved at {output_path}")
