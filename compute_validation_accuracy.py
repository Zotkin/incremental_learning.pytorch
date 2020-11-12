from typing import Any, Dict

import yaml

def load_yaml(path: str) -> Dict[str, Any]:

    with open(path, "r") as f:
        d = yaml.load(f, )


if __name__ == "__main__":

    for seed in range(1,4):

        class_order = load_yaml()
        for num_classes_per_stage in (1,5,10):
            pass
