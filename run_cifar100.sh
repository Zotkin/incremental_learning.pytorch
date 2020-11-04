docker run --gpus 0  --mount type=bind,source="$(pwd)"/checkpoints/,target=/checkpoints/ podnet \
python3 -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device 0 --label podnet_cnn_cifar100_50steps \
    --data-path "$(pwd)"/data
