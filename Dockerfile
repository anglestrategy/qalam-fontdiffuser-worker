# Qalam Foundry — RunPod Serverless Flex GPU Worker (Qwen-Image-2.1 + 4-Step Lightning LoRA)
# Runs Qwen/Qwen-Image-2.1 (compact 7B DiT + Qwen3-VL multimodal encoder, native Arabic &
# multi-reference image editing, public ungated weights) on 48GB+ / 80GB+ BF16 CUDA GPUs
# with zero local MacBook RAM usage.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Verify every Qwen-Image-2.1 + LoRA + vtracer + runpod import succeeds at build time
RUN python3 -c "\
import torch, transformers, diffusers, peft, accelerate, cv2, scipy, vtracer, runpod, jinja2; \
from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler; \
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor; \
assert tuple(int(x) for x in jinja2.__version__.split('.')[:3]) >= (3, 1, 6), jinja2.__version__; \
print('QWEN_IMAGE_21_BUILD_IMPORTS_OK', \
      'torch=', torch.__version__, \
      'transformers=', transformers.__version__, \
      'diffusers=', diffusers.__version__, \
      'peft=', peft.__version__, \
      'jinja2=', jinja2.__version__)"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
