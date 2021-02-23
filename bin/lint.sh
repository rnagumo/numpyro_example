set -xe

poetry run mypy .
poetry run flake8 .
poetry run isort .
poetry run black .
