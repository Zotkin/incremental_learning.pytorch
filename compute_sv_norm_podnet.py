import os
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_jointly(metrics: Dict[str,Dict[str, List[float]]]):
    plt.figure()

    current_plot_number = 1
    for metric in metrics['baseline'].keys():
        plt.subplot(410 + current_plot_number)
        plt.gca().set_title(metric)
        for model in metrics.keys():
            if metric == "accuracy": # accuracy is computer per stage
                total_num_epochs = len(metrics[model]["sv_ratio"])
                total_num_stages = len(metrics[model]["accuracy"])
                num_epochs_per_stage = total_num_epochs/total_num_stages
                epochs = [num_epochs_per_stage*i for i in range(total_num_stages)]

                plt.plot(epochs, metrics[model][metric], label = model)
            else:
                plt.plot(metrics[model][metric], label=model)
        current_plot_number += 1
    plt.legend()
    plt.show()


def get_sv_values(linear_layer: torch.Tensor) -> Tuple[float, float]:
    u, s, v = torch.svd(torch.matmul(linear_layer, linear_layer.T))

    sv_entropy = float(-torch.sum(F.softmax(torch.sqrt(s), dim=0)*F.log_softmax(torch.sqrt(s), dim=0)).numpy())
    sv_ratio = float(s[0] / (s[-1] + 0.00001))
    return sv_entropy, sv_ratio

def get_norm(linear_layer: torch.Tensor) -> float:
    norm = float(torch.mean(torch.norm(linear_layer, dim=1)).numpy())
    return norm

def get_sorted_checkpoint_paths(path: str) -> List[str]:
    return list(map(lambda x: os.path.join(path, x),sorted(os.listdir(path))))

def get_sorted_accuracies(root_path: str, name: str, examples_per_stage: int):

    num_stages = int(1 + 50/examples_per_stage)
    accuracies = []
    for i in range(num_stages):
        path = os.path.join(root_path,f"seeded/{name}/accuracy/1/{examples_per_stage}/accuracy_task_{i}.npy")
        accuracy_object = np.load(path, allow_pickle=True)
        accuracies.append(accuracy_object.item()['total'])
    return accuracies





def main(root_path: str, examples_per_stage: int):

    continual_model_folder = os.path.join(root_path, f"seeded/continual/1/{examples_per_stage}/")
    baseline_model_folder = os.path.join(root_path, f"seeded/baseline/1/{examples_per_stage}/")

    NUM_ENCODER_TENSORS = 187

    metrics = {
        "baseline": {
            "sv_ratio":[],
            "sv_entropy":[],
            "norm":[],
            "accuracy": []
        },
        "continual": {
            "sv_ratio":[],
            "sv_entropy":[],
            "norm":[],
            "accuracy":[]
        }
    }

    for name, folder_path in [("baseline", baseline_model_folder), ("continual", continual_model_folder)]:
        checkpoint_paths = get_sorted_checkpoint_paths(folder_path)

        metrics[name]['accuracy'] = get_sorted_accuracies(root_path, name, examples_per_stage)

        for path in tqdm(checkpoint_paths):

            state_dict = torch.load(path, map_location=torch.device("cpu"))
            num_linear_tensors = len(state_dict) - NUM_ENCODER_TENSORS
            linear_tensors = []
            for i in range(num_linear_tensors):
                linear_tensors.append(state_dict[f"classifier._weights.{i}"])
            linear_matrix = torch.cat(linear_tensors)
            sv_entropy, sv_ratio = get_sv_values(linear_matrix)
            sv_norm = get_norm(linear_matrix)
            metrics[name]["sv_entropy"].append(sv_entropy)
            metrics[name]['sv_ratio'].append(sv_ratio)
            metrics[name]['norm'].append(sv_norm)

    plot_jointly(metrics)




if __name__ == "__main__":
    root_path = "/home/leet/projects/checkpoints/podnet"
    examples_per_stage = 1
    main(root_path, examples_per_stage)