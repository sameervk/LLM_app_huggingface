# Use the official Python image from the Docker Hub
FROM python:3.12-slim-bookworm

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.5.25 /uv /uvx /bin/

COPY . /api

WORKDIR /api


#EXPOSE 8000


LABEL authors="sameervk"

