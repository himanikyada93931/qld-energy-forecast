FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# No database is copied in. startup.py builds one on an empty volume and
# skips the build when the volume already holds a complete database.

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["sh", "-c", "python scripts/startup.py && exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]