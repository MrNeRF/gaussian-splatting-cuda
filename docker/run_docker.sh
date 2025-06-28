#!/bin/bash

usage() {
    echo "Usage: $0 [-h] [-b] [-n] [-u] [-c] [-j | -f]"
    echo " -h    Show this help message"
    echo " -b    Build with cache"
    echo " -n    Build without cache"
    echo " -u    Start docker-compose and enter container"
    echo " -c    Stop and clean up"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

DOCKERFILE="docker/Dockerfile"
BUILD=false
BUILD_ARGS=""
COMPOSEUP=false

while getopts "hbnuc" opt; do
    case ${opt} in
        h ) usage; exit 0 ;;
        b ) BUILD=true ;;
        n ) BUILD=true; BUILD_ARGS="--no-cache" ;;
        u ) COMPOSEUP=true ;;
        c ) docker compose down --remove-orphans; exit 0 ;;
        * ) usage; exit 1 ;;
    esac
done

export USER_UID=$(id -u)
export USER_GID=$(id -g)
export USERNAME=$(id -un)
export USER_PASSWORD=${USERNAME}
export HOSTNAME=$(hostname)
export HOME=$HOME
export DISPLAY=$DISPLAY
export XAUTHORITY=$XAUTHORITY
export SSH_AUTH_SOCK=$SSH_AUTH_SOCK
export DOCKERFILE  # Needed by docker-compose

if [ "$BUILD" = true ]; then
    echo -e "Building docker image with \033[0;31m$DOCKERFILE\033[0m..."
    DOCKER_BUILDKIT=1 docker compose build $BUILD_ARGS
    if [ "$?" -ne 0 ]; then
        echo "Docker build failed!"
        exit 1
    fi
fi

if [ "$COMPOSEUP" = true ]; then
    echo "Starting docker container..."
    docker compose up -d
    if [ "$?" -ne 0 ]; then
        echo "Docker compose up failed!"
        exit 1
    fi
    docker compose exec gs-cuda bash
fi
