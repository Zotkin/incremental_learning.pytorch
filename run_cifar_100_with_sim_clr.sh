docker run --gpus device=1  --mount type=bind,source="$(pwd)"/checkpoints_sim_clr/,target=/checkpoints/  \
                     --mount type=bind,source="$(pwd)"/data/,target=/workspace/incremental_learning.pytorch/data/ \
                     --mount type=bind,source="$(pwd)"/accuracy_sim_clr/,target=/accuracy/ \
                     podnet:sim_clr \
python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_1.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
    --use-sim-clr --nt-xent-temperature 0.1 --sim-clr-alpha 0.0  \
     --data-path "$(pwd)"/data

#mv /home/leet/projects/incremental_learning.pytorch/checkpoints/* /home/leet/projects/checkpoints/podnet/with_sim_clr/continual/1/
#mv /home/leet/projects/incremental_learning.pytorch/accuracy/* /home/leet/projects/checkpoints/podnet/with_sim_clr/continual/accuracy/1/
