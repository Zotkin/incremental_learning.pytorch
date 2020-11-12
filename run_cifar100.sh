for ((i=1; i<4; i++)); do
  for j in {1,5,10}; do

    docker run --gpus 0  --mount type=bind,source="$(pwd)"/checkpoints/,target=/checkpoints/  \
                         --mount type=bind,source="$(pwd)"/data/,target=/data/ \
                         --mount type=bind,source="$(pwd)"/accuracy/,target=/accuracy/ \
                         podnet \
    python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_$i.yaml \
        --initial-increment 50 --increment $j --fixed-memory \
        --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
        --data-path "$(pwd)"/data

    mv /home/leet/projects/incremental_learning.pytorch/checkpoints/* /home/leet/projects/checkpoints/podnet/seeded/continual/$i/$j
    mv /home/leet/projects/incremental_learning.pytorch/accuracy/* /home/leet/projects/checkpoints/podnet/seeded/continual/accuracy/$i/$j
  done
done