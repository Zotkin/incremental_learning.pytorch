
source /home/zoy07590/virtenv_project-lamaml/bin/activate
python -minclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_permutation_1.yaml \
        --initial-increment "$FIRST_INCREMENT" --increment "$INCREMENT" --fixed-memory \
        --device 0 --label podnet_cnn_cifar100_50steps --no-benchmark \
        --data-path /home/zoy07590/incremental_learning.pytorch/data \
        --memory-size "$MEMORY"
