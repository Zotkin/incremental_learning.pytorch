#!/bin/bash
#SBATCH -n 1                  # number of tasks
#SBATCH -N 1                  # number of nodes
#SBATCH -c 4                  # number of cores per task
#SBATCH -t 7-0                # time limit ; format : "minutes:seconds" | "hours:minutes:seconds" | "days-hours"
#SBATCH -p GPU                # partition to use
#SBATCH --gres=gpu:1          # total number of gpu to allocate
#SBATCH --mem=32G             # maximum amout of RAM that can be used
#SBATCH -x calcul-gpu-lahc-3


FIRST_INCREMENT="$1"
INCREMENT="$2"
MEMORY="$3"

export OMP_NUM_THREADS=1
source /home/zoy07590/virtenv_project-lamaml/bin/activate
srun python -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_1.yaml \
        --initial-increment "$FIRST_INCREMENT" --increment "$INCREMENT" --fixed-memory \
        --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
        --data-path /home/zoy07590/incremental_learning.pytorch/data \
        --memory-size "$MEMORY"



