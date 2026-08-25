FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements.wheels.txt ./requirements.wheels.txt
RUN python -m pip install --upgrade pip setuptools wheel
# Try installing binary wheels only to avoid long builds and source compiles
RUN pip install --no-cache-dir -r requirements.wheels.txt || echo "wheel install failed, retrying with network-enabled requirements"
COPY docker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY src/ ./src/

ENV PORT=8080
EXPOSE 8080

CMD ["python", "src/app.py"]
