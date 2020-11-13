import numpy as np

if __name__ == "__main__":
    path = "/home/leet/projects/checkpoints/podnet/seeded/baseline/1/1/accuracy.npy"
    print(np.load(path,  allow_pickle=True))