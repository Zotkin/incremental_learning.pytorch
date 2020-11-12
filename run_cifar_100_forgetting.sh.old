docker run --gpus 0  --mount type=bind,source="$(pwd)"/checkpoints/,target=/checkpoints/ podnet \
python3 -minclearn --options options/podnet/ablations/perceptual_losses/podnet_cnn_podFlat_podPixels_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --fixed-memory --memory-size 100 \
    --device 0 --label podnet_cnn_cifar100_50steps \
    --data-path "$(pwd)"/data
