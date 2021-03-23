for ((i=1; i<2; i++)); do
  for j in {1,}; do

    docker run --gpus 0  --mount type=bind,source="$(pwd)"/checkpoints/,target=/checkpoints/  \
                         --mount type=bind,source="$(pwd)"/data/,target=/data/ \
                         --mount type=bind,source="$(pwd)"/accuracy/,target=/accuracy/ \
                         podnet \
    python3

#    mv /home/leet/projects/incremental_learning.pytorch/checkpoints/* /home/leet/projects/checkpoints/podnet/seeded/continual/$i/$j
#    mv /home/leet/projects/incremental_learning.pytorch/accuracy/* /home/leet/projects/checkpoints/podnet/seeded/continual/accuracy/$i/$j
  done
done