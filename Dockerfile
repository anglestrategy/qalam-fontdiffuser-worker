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
    && python -c "import re; from pathlib import Path; p = Path('/app/fontdiffuser/src/build.py'); p.write_text(re.sub(r',\s*SCR\b', '', p.read_text()))"

# Verify FontDiffuser inference imports & architecture instantiation at build time!
RUN python -c "import sys; from types import SimpleNamespace; sys.path.insert(0, '/app/fontdiffuser'); from src import FontDiffuserModelDPM, FontDiffuserDPMPipeline, build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler; args = SimpleNamespace(resolution=96, unet_channels=(64,128,256,512), style_image_size=(96,96), content_image_size=(96,96), content_encoder_downsample_size=3, channel_attn=True, content_start_channel=64, style_start_channel=64, beta_scheduler='scaled_linear'); m = FontDiffuserModelDPM(unet=build_unet(args), style_encoder=build_style_encoder(args), content_encoder=build_content_encoder(args)); s = build_ddpm_scheduler(args); p = FontDiffuserDPMPipeline(model=m, ddpm_train_scheduler=s, model_type='noise', guidance_type='classifier-free', guidance_scale=7.5); print('FontDiffuser full pipeline verified!')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
