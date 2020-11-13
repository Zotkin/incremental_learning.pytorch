import os

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    root = "/home/leet/projects/checkpoints/podnet/"

    for run in range(1,2):
        for num_examples_per_stage in (1,):

            baseline_path = os.path.join(root, f"seeded/baseline/accuracy/{run}/{num_examples_per_stage}/")
            continual_path = os.path.join(root, f"seeded/continual/accuracy/{run}/{num_examples_per_stage}/")
            for task in range(51):
                accuracy = np.load(os.path.join(baseline_path, f"accuracy_task_{task}.npy"), allow_pickle=True)
                print(accuracy)