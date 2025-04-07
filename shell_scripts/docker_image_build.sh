#!/bin/bash

# shell script to build an image using compose

# Extract the latest commit hash
LATEST_COMMIT=$(git rev-parse --short=7 HEAD)

# Export the latest commit hash to the shell environment
export LATEST_COMMIT

# Print the latest commit hash
echo "The latest commit hash is: $LATEST_COMMIT"

source $PWD/../api/access_tokens.env

export VERSION=v1.1

# Build the Docker image
docker compose -f compose.yaml build api

