FROM python:3.9-slim-buster
EXPOSE 5000
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install -r requirements.txt; \
    python -m pip install gunicorn

WORKDIR /app
COPY . /app

RUN useradd appuser && chown -R appuser /app
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "chatbot-webgui:app"]
