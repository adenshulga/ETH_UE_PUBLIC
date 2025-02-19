#! /bin/bash
source container_setup/credentials

docker run \
    -d \
    --shm-size=8g \
    --memory=32g \
    --cpuset-cpus=96-107 \
    --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
    --name ${CONTAINER_NAME} \
    --rm \
    -it \
    --init \
    --gpus '"device=4"' \
    -v /home/${USER}/${SRC}:/app \
    -p ${INNER_PORT}:${CONTAINER_PORT} \
    ${DOCKER_NAME} \
    bash