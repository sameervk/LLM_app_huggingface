#!/bin/bash

# Extract the latest commit hash
LATEST_COMMIT=$(git rev-parse --short=7 HEAD)

# Export the latest commit hash to the shell environment
export LATEST_COMMIT

# Print the latest commit hash
echo "The latest commit hash is: $LATEST_COMMIT"

source $PWD/../api/access_tokens.env

export VERSION=v1.1

# Build the Docker image and start the container
docker compose -f ../compose_api_volume.yaml up -d api

# for recreating if change in config
#docker compose -f compose_api_volume.yaml up -d api --force-recreate

# fresh build
#docker compose -f compose_api_volume.yaml up -d api --build



