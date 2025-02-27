#! /bin/bash

source container_setup/credentials

docker build -f container_setup/Dockerfile -t ${DOCKER_NAME} . \
        --build-arg DOCKER_NAME=${DOCKER_NAME} \
        --build-arg USER_ID=${DOCKER_USER_ID} \
        --build-arg GROUP_ID=${DOCKER_GROUP_ID}
