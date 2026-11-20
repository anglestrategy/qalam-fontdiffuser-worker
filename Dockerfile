# Qalam Foundry — RunPod Serverless Flex GPU Worker (GlyphAtlas 12B FLUX)
# Runs Fr0zencr4nE/GlyphAtlas (12B FLUX-based multilingual scene-text & glyph synthesis)
# on 48GB+ / 80GB+ BF16 CUDA GPUs with zero local MacBook RAM usage.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    GLYPH_OMNI_ROOT=/app/GlyphAtlas

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone official GlyphAtlas repository and bootstrap hash-pinned AMAP-ML/FluxText sources
RUN git clone --depth 1 https://github.com/Fr0zenCrane/GlyphAtlas.git /app/GlyphAtlas \
    && cd /app/GlyphAtlas \
    && python3 tools/bootstrap_upstream.py

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Verify every GlyphAtlas + FluxText + vtracer + runpod import succeeds at build time
RUN python3 -c "import sys; sys.path[:0]=['/app/GlyphAtlas/runtime','/app/GlyphAtlas/runtime/scripts','/app/GlyphAtlas/runtime/scripts/reproduce_flux_text','/app/GlyphAtlas/vendor/FluxText']; \
    import common; \
    from glyph_atlas.model import AtlasTransformer; \
    from glyph_atlas.data import make_routes; \
    from pipeline_precision import StableTrainingPipeline; \
    from src.flux.condition import Condition; \
    from src.flux.generate_fill import generate_fill; \
    import vtracer, runpod, hf_transfer; \
    print('GLYPHATLAS_BUILD_IMPORTS_OK')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
