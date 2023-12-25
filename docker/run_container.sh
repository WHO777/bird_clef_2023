#!/bin/bash
docker run \
-p 8869:8869 \
--gpus all \
--runtime=nvidia \
--mount type=bind,source=$KAGGLE_ROOT/BirdCLEF_2023,target=/app \
--mount type=bind,source=$DATASETS_ROOT,target=/app/datasets \
--network="host" \
--shm-size=2g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--name BirdCLEF_2023 \
-it nvcr.io/nvidia/pytorch:23.01-py3 /bin/bash
