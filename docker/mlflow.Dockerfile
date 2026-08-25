FROM python:3.11-slim

WORKDIR /mlflow

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install mlflow==3.15.1

VOLUME ["/mlruns"]
EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlruns", "-h", "0.0.0.0", "-p", "5000"]
