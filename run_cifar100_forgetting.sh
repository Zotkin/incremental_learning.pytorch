for ((i=1; i<3; i++)); do
  for j in {1,}; do

    docker run --gpus 0  --mount type=bind,source="$(pwd)"/checkpoints_forgetting/,target=/checkpoints/  \
                         --mount type=bind,source="$(pwd)"/data/,target=/workspace/incremental_learning.pytorch/data/ \
                         --mount type=bind,source="$(pwd)"/accuracy_forgetting/,target=/accuracy/ \
                         podnet \
    python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_$i.yaml \
        --initial-increment 50 --increment $j --fixed-memory --memory-size 100 \
        --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
        --data-path "$(pwd)"/data

    mv /home/leet/projects/incremental_learning.pytorch/checkpoints_forgetting/* /home/leet/projects/checkpoints/podnet/seeded/baseline/$i/$j
    mv /home/leet/projects/incremental_learning.pytorch/accuracy_forgetting/* /home/leet/projects/checkpoints/podnet/seeded/baseline/accuracy/$i/$j

  done
done