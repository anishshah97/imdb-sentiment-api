FROM python:3.9.2

ENV PYTHONUNBUFFERED 1
ENV GROUP_ID=1000 \
    USER_ID=1000

# EXPOSE 8080
WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip3 install --upgrade pip && \
    pip3 install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

RUN pip3 install "uvicorn[standard]" gunicorn

#TODO: Add gunicorn as opposed to uvicorn alone
# COPY ./scripts/start.sh /start.sh
# RUN chmod +x /start.sh

# COPY ./scripts/gunicorn_conf.py /gunicorn_conf.py

# COPY ./scripts/start-reload.sh /start-reload.sh
# RUN chmod +x /start-reload.sh

COPY . ./
ENV PYTHONPATH app

#TODO: Not default to admin
RUN useradd -m admin
RUN chown -R admin:admin ./
RUN chmod 755 ./
USER admin

# Run the start script, it will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Gunicorn with Uvicorn
# CMD ["/start.sh"]
CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
# ENTRYPOINT ["python", "app/main.py"]
