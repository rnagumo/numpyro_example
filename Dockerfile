FROM python:3.8-buster

RUN pip install --no-cache --upgrade pip
RUN pip install --no-cache poetry

WORKDIR /app
COPY . .
RUN poetry install --no-dev --no-interaction
