FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    pillow \
    requests

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
