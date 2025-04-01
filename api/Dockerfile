
# Use the official Python image from the Docker Hub
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates make

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.5.25 /uv /uvx /bin/

WORKDIR /api

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . .

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# add this if running commands using uv is not preferred
#ENV PATH="/api/.venv/bin:$PATH"

RUN mkdir RnD && touch RnD/__init__.py

EXPOSE 5000

ENTRYPOINT ["/bin/bash"]

LABEL authors="sameervk"
