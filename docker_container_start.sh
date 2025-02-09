#!/bin/bash

# Extract the latest commit hash
LATEST_COMMIT=$(git rev-parse --short=7 HEAD)

# Export the latest commit hash to the shell environment
export LATEST_COMMIT

# Start the Docker container with the service named 'api'
docker compose run -dit --name LLM_app_huggingface-api api


