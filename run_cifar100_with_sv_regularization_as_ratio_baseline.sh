docker run --gpus device=1  --mount type=bind,source="$(pwd)"/checkpoints_forgetting/,target=/checkpoints/  \
                     --mount type=bind,source="$(pwd)"/data/,target=/data/ \
                     --mount type=bind,source="$(pwd)"/accuracy_forgetting/,target=/accuracy/ \
                     podnet \
python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_1.yaml \
    --initial-increment 50 --increment 1 --fixed-memory --memory-size 100  \
    --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
    --sv-regularization --sv-regularization-type ratio --sv-regularization-strength 2.0  --data-path "$(pwd)"/data

#mv /home/leet/projects/incremental_learning.pytorch/checkpoints_with_regularization_baseline/* /home/leet/projects/checkpoints/podnet/with_sv_regularization/baseline/1/
#mv /home/leet/projects/incremental_learning.pytorch/accuracy_with_regularization_baseline/* /home/leet/projects/checkpoints/podnet/with_sv_regularization/baseline/accuracy/1/