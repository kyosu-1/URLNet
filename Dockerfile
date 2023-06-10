FROM python:3.9.10-slim-buster

ENV PROJECT_DIR /app
WORKDIR $PROJECT_DIR

RUN pip install --upgrade pip && \
    pip install poetry=="1.5.1"
COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt > requirements.txt && \
    pip install -r requirements.txt

COPY ./api $PROJECT_DIR/api

CMD ["uvicorn", "api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]