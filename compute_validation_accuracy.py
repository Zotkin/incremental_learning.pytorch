import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import yaml
import torch
import numpy as np

from inclearn.lib.data.incdataset import IncrementalDataset
from inclearn.lib import metrics
from inclearn.models import PODNet


def read_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d


def get_number_of_stages_and_epochs(path: str) -> List[Tuple[int, int]]:
    epochs = defaultdict(int)
    filenames = os.listdir(path)
    for fname in filenames:
        if "accuracy" in fname:
            continue
        stage = int(fname.split("_")[2])
        epoch = int(fname.split("_")[-1].split(".")[0])
        if epochs[stage] < epoch:
            epochs[stage] = epoch

    return [(stage, epochs[stage]) for stage in sorted(epochs.keys())]


def get_dataset(class_order: List[int], examples_per_stage: int) -> IncrementalDataset: # todo annotate types
    dataset = IncrementalDataset(
        dataset_name="cifar100",
        random_order=class_order,
        increment=examples_per_stage,
        initial_increment=50,
        data_path="/home/leet/projects/incremental_learning.pytorch/data",
        dataset_transforms={'color_jitter': True}
    )
    return dataset

def get_model(config: Dict[str, Any]) -> PODNet:
    config['device'] = [0]
    config['batch_size'] = 128
    config["memory_size"] = 2000
    config["fixed_memory"] = True
    model = PODNet(config)
    return model


if __name__ == "__main__":
    for case in ['continual', 'baseline']:
        for seed in range(1,4):
            class_order_config_path = f"/home/leet/projects/incremental_learning.pytorch/options/data/cifar100_permutation_{seed}.yaml"
            class_order = read_yaml_config(class_order_config_path)['order'][0]
            for num_classes_per_stage in (1, 5, 10):

                model_config_path = "/home/leet/projects/incremental_learning.pytorch/options/podnet/podnet_cnn_cifar100.yaml"
                model_config = read_yaml_config(model_config_path)
                dataset = get_dataset(class_order, num_classes_per_stage)
                model = get_model(config=model_config)
                model.inc_dataset = dataset

                metrics_logger = metrics.MetricLogger(dataset.n_tasks, dataset.n_classes, dataset.increments)

                experiment_folder = f"/home/leet/projects/checkpoints/podnet/seeded/{case}/{seed}/{num_classes_per_stage}/"
                experiment_accuracy = []
                memory, memory_val = None, None
                for (stage, num_epochs) in get_number_of_stages_and_epochs(experiment_folder):

                    task_info, train_loader, val_loader, test_loader = dataset.new_task(memory, memory_val)
                    model.set_task_info(task_info)

                    model.eval()
                    model.before_task(train_loader, val_loader if val_loader else test_loader)


                    state_dict_path = os.path.join(experiment_folder, f"podnet_task_{stage}_epoch_{num_epochs}.pth")
                    state_dict = torch.load(state_dict_path)
                    model._network.load_state_dict(state_dict, strict=True)


                    model.eval()
                    model.after_task_intensive(dataset)
                    model._after_task(dataset)
                    y_hat, y = model.eval_task(test_loader)

                    metrics_logger.log_task(y_hat, y, task_size=task_info['increment'])

                    print(metrics_logger.last_results['incremental_accuracy'])

                    #y_hat = np.argmin(y_hat, axis=1)
                    #accuracy = ((y_hat == y).sum()/len(y))
                    #print(accuracy)
                    #experiment_accuracy.append(accuracy)
                    memory = model.get_memory()
                    memory_val = model.get_val_memory()

                experiment_accuracy_array = np.array(experiment_accuracy)
                experiment_accuracy_path = os.path.join(experiment_folder, "accuracy.npy")
                np.save(experiment_accuracy_path, experiment_accuracy_array)
