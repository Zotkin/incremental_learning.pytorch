import os
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams["figure.figsize"] = (20,20)

def plot_jointly(metrics: Dict[str,Dict[str, List[float]]]):
    plt.figure()
    current_plot_number = 1
    num_points = len(metrics[1]['continual']['accuracy'])
    for metric in metrics[1]['baseline'].keys():
        plt.subplot(410 + current_plot_number)
        plt.gca().set_title(metric)
        for model in metrics[1].keys():
            color = "red" if model == "continual" else "blue"
            for run in range(3):
                if model == "baseline" and metric == "accuracy":
                    plt.plot(metrics[run][model][metric][:num_points], label=model, color=color)
                else:
                    plt.plot(metrics[run][model][metric], label=model, color=color)
        current_plot_number += 1
    plt.legend()
    plt.show()


def get_sv_values(linear_layer: torch.Tensor) -> Tuple[float, float]:
    u, s, v = torch.svd(torch.matmul(linear_layer, linear_layer.T))

    sv_entropy = float(-torch.sum(F.softmax(torch.sqrt(s), dim=0)*F.log_softmax(torch.sqrt(s), dim=0)).numpy())
    sv_ratio = float(s[0] / (s[-1] + 0.0001))
    return sv_entropy, sv_ratio

def get_norm(linear_layer: torch.Tensor) -> float:
    norm = float(torch.mean(torch.norm(linear_layer, dim=1)).numpy())
    return norm

def get_sorted_checkpoint_paths(path: str) -> List[str]:
    return list(map(lambda x: os.path.join(path, x),sorted(os.listdir(path))))


def main(root_path: str, examples_per_stage: int):
    runs = []
    for run in range(1, 4):
        continual_model_folder = os.path.join(root_path, f"continual/{run}/{examples_per_stage}/")
        baseline_model_folder = os.path.join(root_path, f"baseline/{run}/{examples_per_stage}/")


        metrics = {
            "baseline": {
                "sv_ratio":[],
                "sv_entropy":[],
                "norm":[]
            },
            "continual": {
                "sv_ratio":[],
                "sv_entropy":[],
                "norm":[]
            }
        }

        for name, folder_path in [("baseline", baseline_model_folder), ("continual", continual_model_folder)]:
            checkpoint_paths = get_sorted_checkpoint_paths(folder_path)


            for path in tqdm(checkpoint_paths):
                if "accuracy" in path:
                    with open(path, "r") as f:
                        accuracy = f.read()
                        metrics[name]['accuracy'] = list(map(float, accuracy.split(" ")[:-1]))
                else:
                    state_dict = torch.load(path, map_location=torch.device("cpu"))

                    try:
                        linear_matrix = state_dict['fc.weight']
                    except:
                        linear_matrix = torch.cat([state_dict['fc.fc1.weight'], state_dict['fc.fc2.weight']])
                    sv_entropy, sv_ratio = get_sv_values(linear_matrix)
                    sv_norm = get_norm(linear_matrix)
                    metrics[name]["sv_entropy"].append(sv_entropy)
                    metrics[name]['sv_ratio'].append(sv_ratio)
                    metrics[name]['norm'].append(sv_norm)
        runs.append(metrics)
    plot_jointly(runs)




if __name__ == "__main__":
    root_path = "/home/leet/projects/checkpoints/ucir/seeded"
    examples_per_stage = 1
    main(root_path, examples_per_stage)