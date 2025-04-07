#!/bin/bash

# shell script to push the image to docker

# Extract the latest commit hash
LATEST_COMMIT=$(git rev-parse --short=7 HEAD)

# Export the latest commit hash to the shell environment
export LATEST_COMMIT

# Print the latest commit hash
echo "The latest commit hash is: $LATEST_COMMIT"

# Tag the Docker image
docker tag llmapp-api:latest sameervk/llmapp-api:$LATEST_COMMIT
# docker tag image:tag repository:tag

# Push the Docker image to Docker Hub
docker push sameervk/llmapp-api:$LATEST_COMMIT