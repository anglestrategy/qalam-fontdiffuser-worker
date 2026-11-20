FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_SDF_BATCH_SIZE=32

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Build-time verification that CUDA typographic SDF worker loads cleanly with all deps
RUN python -c "import handler, torch, vtracer, scipy; print('Qalam CUDA Typographic SDF Worker v3 verified! torch=', torch.__version__)"

CMD ["python", "-u", "/app/handler.py"]
