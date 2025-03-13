
# Use the official Python image from the Docker Hub
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates make

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.5.25 /uv /uvx /bin/

COPY . /api

WORKDIR /api

RUN uv sync --frozen --no-cache

ENV PATH="/api/.venv/bin:$PATH"

#EXPOSE 8000
RUN source .venv/bin/activate && make all

LABEL authors="sameervk"
