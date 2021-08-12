FROM python:3.9.2

ENV PYTHONUNBUFFERED 1
ENV GROUP_ID=1000 \
    USER_ID=1000

# EXPOSE 8080
ENV APP_HOME /app
WORKDIR ${APP_HOME}

COPY poetry.lock pyproject.toml ./
RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

RUN pip3 install gunicorn

COPY . ./
ENV PYTHONPATH app

CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
# CMD exec gunicorn --bind :${PORT:-5000} --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app
