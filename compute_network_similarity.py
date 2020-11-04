import os
from typing import List, Dict, Any
import torch
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from cka import linear_CKA
from inclearn.lib.network.basenet import BasicNet


def get_last_epoch(path, stage):
    checkpoints = os.listdir(path)
    max_epoch = -1
    for chk in checkpoints:
        s = int(chk.split("_")[2])
        if s == stage:
            epoch = int(chk.split("_")[-1].split(".")[0])
            max_epoch = max(max_epoch, epoch)
    return max_epoch

def read_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

def get_cifar_100(path: str) -> List[np.ndarray]:
    X_train = np.load(os.path.join(path, "x_train.npy"))
    X_test = np.load(os.path.join(path, "x_test.npy"))
    y_train = np.load(os.path.join(path, "y_train.npy"))
    y_test = np.load(os.path.join(path, "y_test.npy"))
    return [X_train, y_train, X_test, y_test]


class ActivationObject:
    def __init__(self):
        self.activation = OrderedDict()

    def __len__(self):
        return len(self.activation.keys())

    def get_activation_hook(self, name):
        def hook(model, input, output):
            output = output.detach().numpy()
            pooled_activation = np.mean(output, axis=(1,2))
            self.activation[name] = pooled_activation

        return hook



def register_activation_hooks(encoder: torch.nn.Module, activation_object: ActivationObject) -> None:
    for i, (name, module) in enumerate(encoder.named_modules()):
        if i > 1 and isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(activation_object.get_activation_hook(name))


def get_per_layer_similarity(model_1: torch.nn.Module, model_2: torch.nn.Module, X: torch.FloatTensor):
    activation_object_model_1 = ActivationObject()
    activation_object_model_2 = ActivationObject()

    register_activation_hooks(model_1.convnet, activation_object_model_1)
    register_activation_hooks(model_2.convnet, activation_object_model_2)

    with torch.no_grad():
        print("Recording from hooks model 1")
        model_1.convnet(X)
        print("Recording from hooks model 2")
        model_2.convnet(X)

    similarity = []

    print("Computing the CCA")
    for (name_1, activation_1), (name_2, activation_2) in tqdm(
            zip(activation_object_model_1.activation.items(), activation_object_model_2.activation.items()),
            total=len(activation_object_model_1)):
        assert name_1 == name_2

        cka = linear_CKA(activation_1, activation_2)
        similarity.append(cka)

    return similarity


if __name__ == "__main__":
    NUM_STAGES = 51
    CIFAR_10_ROOT = "./data/"
    CONFIG_PATH = "./options/podnet/podnet_cnn_cifar100.yaml"
    EXAMPLES_PER_CLASS = 1
    global_config = read_yaml_config(CONFIG_PATH)
    net_config = {
        "convnet_type": global_config['convnet'],
        "classifier_kwargs": global_config.get("classifier_config", {}),
        "postprocessor_kwargs": global_config.get("postprocessor_config", {}),
        "device": torch.device("cpu"),
        "return_features": True,
        "extract_no_act": True,
        "classifier_no_act": global_config.get("classifier_no_act", True),
        "attention_hook": True,
        "gradcam_hook": global_config.get("gradcam_distil", {})
    }

    X_train, y_train, X_test, y_test = get_cifar_100(CIFAR_10_ROOT)
    print(X_train.shape)
    random_index = np.random.randint(low=0, high=len(X_train), size=(1000,))
    X = torch.Tensor(np.moveaxis(X_train[random_index], 3, 1))


    similarity_ext_data = []
    similarity_baseline = []

    for external_data in [True, False]:
        for stage in range(1,NUM_STAGES):

            last_epoch_stage_0 = get_last_epoch(f"/home/leet/projects/checkpoints/podnet/{'continual' if external_data else 'baseline'}/{EXAMPLES_PER_CLASS}", stage-1)
            last_epoch_stage_1 = get_last_epoch(f"/home/leet/projects/checkpoints/podnet/{'continual' if external_data else 'baseline'}/{EXAMPLES_PER_CLASS}", stage)

            CHECKPOITN_PATH_STAGE_0 = f"/home/leet/projects/checkpoints/podnet/{'continual' if external_data else 'baseline'}" \
                                      f"/{EXAMPLES_PER_CLASS}/podnet_task_{stage-1}_epoch_{last_epoch_stage_0}.pth"
            CHECKPOITN_PATH_STAGE_1 = f"/home/leet/projects/checkpoints/podnet/{'continual' if external_data else 'baseline'}" \
                                      f"/{EXAMPLES_PER_CLASS}/podnet_task_{stage}_epoch_{last_epoch_stage_1}.pth"


            state_dict_model_1 = torch.load(CHECKPOITN_PATH_STAGE_0, map_location=torch.device("cpu"))
            state_dict_model_2 = torch.load(CHECKPOITN_PATH_STAGE_1, map_location=torch.device("cpu"))

            model_1 = BasicNet(**net_config)
            model_1.load_state_dict(state_dict_model_1, strict = False)
            model_2 = BasicNet(**net_config)
            model_2.load_state_dict(state_dict_model_2, strict= False)
            model_1.to(torch.device('cpu'))
            model_2.to(torch.device('cpu'))
            similarity = get_per_layer_similarity(model_1, model_2, X)

            if external_data:
                similarity_ext_data.append(similarity)
            else:
                similarity_baseline.append(similarity)


#    for s in similarity_ext_data:
#        plt.scatter( np.arange(len(s)),s, color="red", label="external data")
    for s in similarity_baseline:
        plt.plot(s,  color="blue", label="baseline")
    plt.title("STD of CKA per layer")
    plt.xlabel("layer")
    plt.ylabel("CKA")
#    plt.legend()
    plt.savefig(f"podnet_cka_{EXAMPLES_PER_CLASS}_similarity_per_layer_baseline.png")
