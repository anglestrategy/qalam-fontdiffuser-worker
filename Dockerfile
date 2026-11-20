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

# Clone official FontDiffuser model architecture (MCA + SCR + DPM-Solver++)
RUN git clone --depth 1 https://github.com/yeungchenwa/FontDiffuser.git /app/fontdiffuser \
    && sed -i '/SCR/d' /app/fontdiffuser/src/modules/__init__.py \
    && sed -i '/SCR/d; /build_scr/d; /ContentPerceptualLoss/d' /app/fontdiffuser/src/__init__.py \
    && sed -i '/build_scr/d; /SCR/d' /app/fontdiffuser/src/build.py

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verify FontDiffuser inference imports at build time so runtime never silently falls back!
RUN python -c "import sys; sys.path.insert(0, '/app/fontdiffuser'); from src import FontDiffuserModelDPM, FontDiffuserDPMPipeline, build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler; print('FontDiffuser imports verified!')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
