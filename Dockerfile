FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FONTDIFFUSER_REPO=/app/fontdiffuser
ENV FONTDIFFUSER_CKPT_DIR=/runpod-volume/fontdiffuser_ckpt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Clone official FontDiffuser model architecture (MCA + DPM-Solver++) and keep inference-only exports cleanly
RUN git clone --depth 1 https://github.com/yeungchenwa/FontDiffuser.git /app/fontdiffuser \
    && printf "from .content_encoder import ContentEncoder\nfrom .style_encoder import StyleEncoder\nfrom .unet import UNet\n" > /app/fontdiffuser/src/modules/__init__.py \
    && printf "from .model import FontDiffuserModel, FontDiffuserModelDPM\nfrom .dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline\nfrom .modules import ContentEncoder, StyleEncoder, UNet\nfrom .build import build_unet, build_ddpm_scheduler, build_style_encoder, build_content_encoder\n" > /app/fontdiffuser/src/__init__.py \
    && python -c "from pathlib import Path; p = Path('/app/fontdiffuser/src/build.py'); txt = p.read_text(); p.write_text(txt.split('def build_scr')[0].replace(',\n                 SCR', '').replace(', SCR', ''))"

# Verify FontDiffuser inference imports at build time so runtime never silently falls back!
RUN python -c "import sys; sys.path.insert(0, '/app/fontdiffuser'); from src import FontDiffuserModelDPM, FontDiffuserDPMPipeline, build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler; print('FontDiffuser imports verified!')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
