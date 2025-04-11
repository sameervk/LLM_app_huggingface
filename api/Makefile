
# install:
# 	uv sync --frozen --no-cache

lint:
	uvx ruff check --exclude *.ipynb

format:
	uvx ruff format --exclude *.ipynb

test:
	python -m pytest -vv test_*.py
# using * to include all files starting with name: test_

all:	format	lint
