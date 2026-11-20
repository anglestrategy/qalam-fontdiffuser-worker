"""
Qalam Foundry — RunPod Serverless Flex GPU Worker (GlyphAtlas 12B FLUX)

True 1:1 Arabic + Latin neural font synthesis using Fr0zencr4nE/GlyphAtlas
(12B FLUX-based multilingual glyph synthesis model, public ungated weights)
paired with unsloth/FLUX.1-dev non-transformer components (VAE + CLIP + T5-XXL).

Architecture:
  - Packs up to 8 target glyphs per 512x512 sheet (2 rows x 4 cols, exact 16px
    latent-patch grid with unmasked white gutters).
  - Unmasked top strip (y = 0..176) displays the user's ACTUAL reference image on
    pure #FFFFFF white so FLUX's 12B global self-attention directly transfers exact
    stroke thickness, geometric bowls, curves, terminal cuts, and square/round dots.
  - Each masked cell (112x144, inner 112x112 baseline-locked box) receives:
      1) Scene-aligned white-on-black skeleton in `condition` (anchors exact baseline
         & origin for HarfBuzz Arabic cursive shaping)
      2) 64x64 VAE-encoded glyph tile in `atlas` routed via `make_routes()` (anchors
         character identity & Arabic dot count/placement without PIL Arabic bugs).
  - Supports both `exact_clone` and `inspired` modes.
  - Vectorizes each generated cell with smooth bicubic upsampling -> adaptive ink
    extraction -> stroke-matched Arabic baseline connector bridging -> vtracer spline SVG.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import runpod
import torch
import torch.nn.functional as F
import vtracer
import yaml
from PIL import Image
from scipy import ndimage

# Ensure GlyphAtlas + vendored AMAP-ML/FluxText are on sys.path
GLYPHATLAS_ROOT = Path(os.environ.get("GLYPH_OMNI_ROOT", "/app/GlyphAtlas")).resolve()
os.environ["GLYPH_OMNI_ROOT"] = str(GLYPHATLAS_ROOT)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

for p in (
    str(GLYPHATLAS_ROOT / "runtime"),
    str(GLYPHATLAS_ROOT / "runtime" / "scripts"),
    str(GLYPHATLAS_ROOT / "runtime" / "scripts" / "reproduce_flux_text"),
    str(GLYPHATLAS_ROOT / "vendor" / "FluxText"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusers import FluxFillPipeline
from huggingface_hub import snapshot_download
from glyph_atlas.data import make_routes
from glyph_atlas.model import AtlasTransformer
from pipeline_precision import StableTrainingPipeline
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill


# ---------------------------------------------------------------------------
# Global persistent state across warm RunPod Serverless requests
# ---------------------------------------------------------------------------
_PIPE: StableTrainingPipeline | None = None
_MODEL_CONFIG: dict[str, Any] | None = None
_OFFLOAD_TEXT_ENCODERS: bool = False
_PROMPT_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_GPU_NAME: str = "unknown"
_VRAM_GB: float = 0.0

# 16px-aligned 2x4 cell grid on a 512x512 canvas:
#   Top unmasked reference strip: y = 0..176 (11 latent rows)
#   Unmasked white gutter:        y = 176..192 (1 latent row)
#   Row 0 cells (112x144 each):   y = 192..336 (9 latent rows)
#   Unmasked white gutter:        y = 336..352 (1 latent row)
#   Row 1 cells (112x144 each):   y = 352..496 (9 latent rows)
#   Bottom unmasked margin:       y = 496..512 (1 latent row)
#   4 columns (112px wide each):  x = 16..128, 144..256, 272..384, 400..512
CELL_W = 112
CELL_H = 144
INNER_BOX = 112
INNER_Y_PAD = 16  # centered vertically inside 144px cell: [y0+16 : y0+128]
COL_X = (16, 144, 272, 400)
ROW_Y = (192, 352)
CELL_BOXES: list[tuple[int, int, int, int]] = [
    (x0, y0, x0 + CELL_W, y0 + CELL_H)
    for y0 in ROW_Y
    for x0 in COL_X
]


def _resolve_model_root() -> Path:
    custom = os.environ.get("GLYPH_MODEL_CACHE", "").strip()
    if custom:
        root = Path(custom)
        root.mkdir(parents=True, exist_ok=True)
        return root
    vol = Path("/runpod-volume")
    if vol.exists() and os.access(str(vol), os.W_OK):
        root = vol / "glyphatlas_models"
        root.mkdir(parents=True, exist_ok=True)
        return root
    # NEVER use /tmp (often RAM-backed tmpfs on cloud hosts!). Use container overlay disk /app/models_cache.
    root = Path("/app/models_cache")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_model_files(model_root: Path) -> tuple[Path, Path]:
    """
    Download public ungated weights ONCE directly into HF_HOME on container disk / Network Volume
    with ZERO duplicate local_dir copying (34 GB total instead of 68 GB!):
      1) Fr0zencr4nE/GlyphAtlas transformer (~23.8 GB, 5 shards)
      2) unsloth/FLUX.1-dev non-transformer files (VAE + CLIP + T5-XXL + scheduler, ~10.1 GB)
    """
    hf_cache = model_root / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache / "hub")
    # Use steady streaming downloads (max_workers=2) to avoid RAM spikes from 16x concurrent 5GB chunks
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    t0 = time.monotonic()
    print(f"[GlyphAtlas] Ensuring Fr0zencr4nE/GlyphAtlas transformer in {hf_cache} ...", flush=True)
    ga_snap = Path(
        snapshot_download(
            repo_id="Fr0zencr4nE/GlyphAtlas",
            cache_dir=str(hf_cache / "hub"),
            allow_patterns=["transformer/*"],
            max_workers=2,
        )
    )
    ckpt_dir = ga_snap / "transformer"
    print(f"[GlyphAtlas] GlyphAtlas transformer ready at {ckpt_dir} ({time.monotonic() - t0:.1f}s)", flush=True)

    t1 = time.monotonic()
    print(f"[GlyphAtlas] Ensuring unsloth/FLUX.1-dev base components in {hf_cache} ...", flush=True)
    base_dir = Path(
        snapshot_download(
            repo_id="unsloth/FLUX.1-dev",
            cache_dir=str(hf_cache / "hub"),
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "vae/*",
                "text_encoder/*",
                "text_encoder_2/*",
                "tokenizer/*",
                "tokenizer_2/*",
            ],
            max_workers=2,
        )
    )
    print(f"[GlyphAtlas] FLUX base ready at {base_dir} ({time.monotonic() - t1:.1f}s)", flush=True)
    return ckpt_dir, base_dir


def _ensure_pipeline() -> tuple[StableTrainingPipeline, dict[str, Any]]:
    global _PIPE, _MODEL_CONFIG, _OFFLOAD_TEXT_ENCODERS, _GPU_NAME, _VRAM_GB

    if _PIPE is not None and _MODEL_CONFIG is not None:
        return _PIPE, _MODEL_CONFIG

    if not torch.cuda.is_available():
        raise RuntimeError("GlyphAtlas worker requires a CUDA GPU with BF16 support.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bfloat16 required by GlyphAtlas.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(0)
    _GPU_NAME = props.name
    _VRAM_GB = props.total_memory / (1024.0 ** 3)
    print(f"[GlyphAtlas] Initializing on {_GPU_NAME} ({_VRAM_GB:.1f} GB VRAM)...", flush=True)

    model_root = _resolve_model_root()
    ckpt_dir, base_dir = _ensure_model_files(model_root)

    # Load 12B transformer directly in BF16 and move to CUDA BEFORE loading T5-XXL
    # so host CPU RAM never peaks above ~14 GB!
    t0 = time.monotonic()
    transformer, info = AtlasTransformer.from_pretrained(
        str(ckpt_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        local_files_only=True,
    )
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"AtlasTransformer strict load mismatch: {info}")
    transformer.eval().requires_grad_(False)
    transformer.to(device=device, dtype=torch.bfloat16)
    gc.collect()
    print(f"[GlyphAtlas] Loaded 12B AtlasTransformer onto {device} in {time.monotonic() - t0:.1f}s", flush=True)

    pipe = FluxFillPipeline.from_pretrained(
        str(base_dir),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    pipe.vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    if _VRAM_GB >= 64.0:
        _OFFLOAD_TEXT_ENCODERS = False
        pipe.text_encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        pipe.text_encoder_2.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    else:
        _OFFLOAD_TEXT_ENCODERS = True
        pipe.text_encoder.to(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
        pipe.text_encoder_2.to(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)

    pipe.__class__ = StableTrainingPipeline
    pipe.set_progress_bar_config(disable=True)

    cfg_path = GLYPHATLAS_ROOT / "vendor" / "FluxText" / "train" / "config" / "word_512_size.yaml"
    model_config = yaml.safe_load(cfg_path.read_text())["model"]

    _PIPE = pipe
    _MODEL_CONFIG = model_config
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[GlyphAtlas] Pipeline ready in {time.monotonic() - t0:.1f}s "
        f"(offload_text_encoders={_OFFLOAD_TEXT_ENCODERS}, dtype={pipe.dtype}, device={pipe.device})",
        flush=True,
    )
    return _PIPE, _MODEL_CONFIG


@torch.inference_mode()
def _get_prompt_embeds(
    pipe: StableTrainingPipeline,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt once with CLIP + T5-XXL and cache GPU tensors across all sheets."""
    cached = _PROMPT_CACHE.get(prompt)
    if cached is not None:
        return cached

    device = pipe.device
    if _OFFLOAD_TEXT_ENCODERS:
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)

    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    if _OFFLOAD_TEXT_ENCODERS:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    # Keep cache small (last 8 prompts)
    if len(_PROMPT_CACHE) >= 8:
        _PROMPT_CACHE.pop(next(iter(_PROMPT_CACHE)))
    _PROMPT_CACHE[prompt] = (prompt_embeds, pooled_prompt_embeds)
    return prompt_embeds, pooled_prompt_embeds


def _snap_bf16_guidance(val: float) -> float:
    """GlyphAtlas expects guidance exact in BF16 (e.g. 1.0, 1.5, 2.0, 3.5)."""
    v = float(max(0.5, min(7.0, val)))
    return float(torch.tensor(v, dtype=torch.bfloat16).item())


# ---------------------------------------------------------------------------
# Reference image cleaning + top-strip placement
# ---------------------------------------------------------------------------
def _decode_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return im.convert("RGB")


def _prepare_reference_canvas(ref_images_b64: list[str], collage_b64: str | None = None) -> np.ndarray:
    """
    Build a 512x512 pure-white RGB uint8 canvas with the user's crisp reference
    typography centered cleanly in the unmasked top strip (y = 8..168, x = 12..500).
    """
    canvas = np.full((512, 512, 3), 255, dtype=np.uint8)

    ref_im: Image.Image | None = None
    if ref_images_b64:
        try:
            ref_im = _decode_pil(ref_images_b64[0])
        except Exception:
            ref_im = None
    if ref_im is None and collage_b64:
        try:
            ref_im = _decode_pil(collage_b64)
        except Exception:
            ref_im = None
    if ref_im is None:
        return canvas

    gray = np.asarray(ref_im.convert("L"), dtype=np.uint8)

    # Detect inverted (white-on-dark) reference and flip to dark-on-white
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.median(border)) < 120.0:
        gray = 255 - gray

    # Gently normalize near-white background to pure 255 while keeping crisp anti-aliased thin strokes
    bg_val = float(np.percentile(gray, 95))
    fg_val = float(np.percentile(gray, 2))
    span = max(25.0, bg_val - fg_val)
    norm = np.clip((gray.astype(np.float32) - fg_val) / span * 255.0, 0.0, 255.0).astype(np.uint8)
    # Push very light background dust (> 238) to pure 255
    norm[norm > 238] = 255

    # Tightly crop ink bounding box
    ink = norm < 220
    ys, xs = np.where(ink)
    if len(ys) > 0:
        pad = 4
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(norm.shape[0], int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(norm.shape[1], int(xs.max()) + 1 + pad)
        norm = norm[y0:y1, x0:x1]

    # If the reference is very tall (multi-line poster), pick the sharpest central text band
    # so the reference letters stay ~75-110px tall matching our bottom glyph cells.
    h, w = norm.shape
    if h > w * 0.65 and h > 180:
        row_ink = (norm < 210).sum(axis=1)
        band_h = max(int(w * 0.35), min(h, 160))
        # Find vertical window with maximum ink density
        cs = np.cumsum(np.r_[0, row_ink])
        scores = cs[band_h:] - cs[:-band_h]
        best_y = int(np.argmax(scores))
        norm = norm[best_y : best_y + band_h, :]
        h, w = norm.shape

    # Fit inside top strip (max_w = 488, max_h = 152) preserving exact aspect ratio
    max_w, max_h = 488, 152
    scale = min(max_w / max(1, w), max_h / max(1, h))
    # Cap upscaling so tiny images stay sharp, and ensure reasonable visual height
    new_w = max(16, min(max_w, int(round(w * scale))))
    new_h = max(16, min(max_h, int(round(h * scale))))

    pil_ref = Image.fromarray(norm, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS)
    ref_arr = np.asarray(pil_ref, dtype=np.uint8)
    ref_rgb = np.stack([ref_arr, ref_arr, ref_arr], axis=-1)

    px = (512 - new_w) // 2
    py = 12 + (156 - new_h) // 2
    canvas[py : py + new_h, px : px + new_w] = ref_rgb
    return canvas


# ---------------------------------------------------------------------------
# Build GlyphAtlas inputs (condition, hint, atlas) for up to 8 glyphs per sheet
# ---------------------------------------------------------------------------
def _make_64_atlas_tile(skel_ink_bool: np.ndarray) -> np.ndarray:
    """
    Create a 64x64 uint8 white-on-black tile matching GlyphAtlas `tile_for()` convention:
    white glyph strokes (255) centered inside 54x54 on pure black (0) background.
    """
    ys, xs = np.where(skel_ink_bool)
    if len(ys) == 0:
        return np.zeros((64, 64), dtype=np.uint8)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = (skel_ink_bool[y0:y1, x0:x1].astype(np.uint8)) * 255
    ch, cw = crop.shape

    ratio = min(54.0 / max(1, cw), 54.0 / max(1, ch))
    nw = max(2, min(56, int(round(cw * ratio))))
    nh = max(2, min(56, int(round(ch * ratio))))
    resized = np.asarray(
        Image.fromarray(crop, mode="L").resize((nw, nh), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    tile = np.zeros((64, 64), dtype=np.uint8)
    ox = (64 - nw) // 2
    oy = (64 - nh) // 2
    tile[oy : oy + nh, ox : ox + nw] = resized
    return tile


@torch.inference_mode()
def _encode_custom_atlas(
    pipe: StableTrainingPipeline,
    tiles_64: list[np.ndarray],
    cell_masks_512: list[np.ndarray],
) -> dict[str, Any]:
    """
    Encode custom 64x64 white-on-black glyph tiles through FLUX VAE and construct
    the exact `atlas` dictionary expected by `AtlasTransformer` + `make_routes()`.
    """
    device = pipe.device
    dtype = pipe.dtype
    k = len(tiles_64)
    assert 1 <= k <= 8

    arr = np.stack(tiles_64, axis=0)  # [K, 64, 64] uint8, white ink on black
    tensor = (
        torch.from_numpy(arr)
        .to(device=device, dtype=dtype)[:, None]
        .expand(-1, 3, -1, -1)
        / 127.5
        - 1.0
    )
    z = pipe.vae.encode(tensor).latent_dist.mean
    z = (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    tokens = pipe._pack_latents(z, *z.shape).detach()  # [K, 16, 64]
    features = tokens.reshape(1, k * 16, 64)

    owners = torch.arange(k, device=device, dtype=torch.long).repeat_interleave(16)
    ids: list[list[float]] = []
    geometry: list[list[float]] = []
    for owner in range(k):
        for y in range(4):
            for x in range(4):
                ids.append([1.0, float(owner * 4 + y), float(x)])
                geometry.append([(x + 0.5) / 4.0, (y + 0.5) / 4.0, 0.0, 1.0 / 20.0])

    masks_t = torch.stack(
        [torch.from_numpy(m.astype(np.float32)).to(device=device) for m in cell_masks_512],
        dim=0,
    )  # [K, 512, 512]
    region_map = F.avg_pool2d(masks_t[:, None], 16, 16).flatten(1).T > 0  # [1024, K]
    routes = make_routes(region_map, owners)

    return {
        "features": features,
        "owners": owners,
        "ids": torch.tensor(ids, device=device, dtype=torch.float32),
        "geometry": torch.tensor(geometry, device=device, dtype=dtype)[None],
        "regions": region_map,
        "routes": routes,
        "glyph_count": k,
        "unit_count": k,
    }


def _build_style_prompt(job_input: dict[str, Any]) -> str:
    mode = str(job_input.get("generation_mode") or "exact_clone").strip().lower()
    ref_tel = job_input.get("ref_telemetry") or {}
    dot_shape = str(ref_tel.get("dot_shape") or "square")
    stroke_ratio = float(ref_tel.get("ref_stroke_ratio") or 0.04)
    extra = str(job_input.get("style_prompt") or "").strip()

    if stroke_ratio < 0.032:
        weight_desc = "ultra-thin hairline monoline strokes"
    elif stroke_ratio < 0.055:
        weight_desc = "light clean monoline strokes"
    elif stroke_ratio < 0.085:
        weight_desc = "regular geometric strokes"
    else:
        weight_desc = "bold heavy display strokes"

    dot_desc = (
        "bold crisp solid square dots"
        if dot_shape == "square"
        else "crisp diamond dots"
        if dot_shape == "diamond"
        else "clean circular dots"
    )

    if mode == "exact_clone":
        base = (
            "A crisp high-contrast black-on-white typography specimen sheet. "
            "Every individual letter in the bottom grid is drawn in the EXACT SAME custom typeface design "
            f"as the reference lettering shown at the top of the page, matching its {weight_desc}, "
            f"exact wide open geometric bowls, letterform anatomy, terminal cuts, and {dot_desc}. "
            "Pure solid black vector glyph ink on a flat pure white background, clean 2D typography, no shadows, no gradients."
        )
    else:
        base = (
            "A crisp high-contrast black-on-white typography specimen sheet. "
            "Every individual letter in the bottom grid is an original custom brand display typeface "
            "harmoniously inspired by the reference lettering shown at the top of the page, "
            f"featuring refined {weight_desc}, distinctive custom terminals, and {dot_desc}. "
            "Pure solid black vector glyph ink on a flat pure white background, clean 2D typography, no shadows."
        )
    if extra and len(extra) < 220:
        base = f"{base} Style notes: {extra}."
    return base


# ---------------------------------------------------------------------------
# Post-process generated cell -> crisp mask + Arabic bridge + vtracer SVG
# ---------------------------------------------------------------------------
def _extract_cell_mask(
    cell_rgb: np.ndarray,
    skel_ink: np.ndarray,
    *,
    canvas_size: int,
    origin_x_px: float,
    baseline_y_px: float,
    advance_width: int,
    scale: float,
    is_arabic_joining: bool,
    preserve_arabic_connectors: bool,
) -> np.ndarray:
    """
    Upscale the 112x112 generated cell to canvas_size with smooth bicubic interpolation,
    threshold cleanly (preserving ultra-thin hairline strokes!), clean tiny noise specks,
    and gently ensure thin stroke-matched Arabic cursive entry/exit bridges.
    """
    gray_small = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)
    gray_big = np.asarray(
        Image.fromarray(gray_small, mode="L").resize((canvas_size, canvas_size), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )

    # Adaptive luminance threshold tuned to keep thin anti-aliased hairlines intact
    bg_med = float(np.percentile(gray_big, 92))
    ink_p2 = float(np.percentile(gray_big, 1))
    contrast = bg_med - ink_p2
    if contrast < 30.0:
        # Almost blank cell — fallback safely to skeleton
        return skel_ink.copy()

    # For thin hairlines, anti-aliased stroke centers can sit at ~130-175 gray;
    # placing threshold at 62% from dark ink toward white background captures hairlines crisply!
    thresh = int(round(min(212.0, max(128.0, ink_p2 + 0.62 * contrast))))
    mask = gray_big < thresh

    # Clear 2px outer border artifacts
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False

    # Remove tiny noise specks (< 10 px at 384px, preserving every real dot which is >= 25 px)
    labeled, num = ndimage.label(mask)
    if num > 0:
        min_area = max(8, int((canvas_size / 384.0) ** 2 * 10))
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_area
        keep[0] = False
        mask = keep[labeled]

    # Gentle anti-staircase smooth before vtracer
    if mask.any():
        blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.75)
        mask = blurred >= 0.48

    # Sanity check: if completely empty, fallback to skeleton
    if int(mask.sum()) < 24:
        return skel_ink.copy()

    # Stroke-matched Arabic cursive baseline connector lock:
    # Only if the original skeleton connects at left (x=0) or right (x=advance_width) on the baseline,
    # ensure the generated glyph ALSO reaches that exact edge at the generated glyph's OWN stroke thickness!
    if preserve_arabic_connectors and is_arabic_joining:
        dist_gen = ndimage.distance_transform_edt(mask)
        nonzero_r = dist_gen[mask]
        stroke_half = float(np.percentile(nonzero_r, 65)) if nonzero_r.size > 0 else 2.0
        half_t = max(1, min(6, int(round(stroke_half))))

        y_top = max(0, int(round(baseline_y_px - 220.0 * scale)))
        y_bot = min(canvas_size, int(round(baseline_y_px + 70.0 * scale)))
        if y_bot > y_top:
            # Left exit (x = origin_x_px)
            exit_x = max(2, min(canvas_size - 3, int(round(origin_x_px))))
            skel_exit_rows = np.where(skel_ink[y_top:y_bot, max(0, exit_x - 2) : min(canvas_size, exit_x + 6)].any(axis=1))[0]
            if skel_exit_rows.size > 0:
                # Find generated ink near the exit band (within 24px to the right)
                search_x1 = min(canvas_size, exit_x + int(round(28 * canvas_size / 384.0)))
                gen_band = mask[y_top:y_bot, exit_x:search_x1]
                if gen_band.any():
                    gy_rel, gx_rel = np.where(gen_band)
                    first_x = exit_x + int(gx_rel.min())
                    cy = y_top + int(round(float(np.median(gy_rel[gx_rel <= gx_rel.min() + 3]))))
                    if first_x > exit_x - 2:
                        mask[max(0, cy - half_t) : min(canvas_size, cy + half_t + 1), max(0, exit_x - 2) : first_x + 2] = True

            # Right entry (x = origin_x_px + advance_width * scale)
            entry_x = max(2, min(canvas_size - 3, int(round(origin_x_px + advance_width * scale))))
            skel_entry_rows = np.where(skel_ink[y_top:y_bot, max(0, entry_x - 6) : min(canvas_size, entry_x + 3)].any(axis=1))[0]
            if skel_entry_rows.size > 0:
                search_x0 = max(0, entry_x - int(round(28 * canvas_size / 384.0)))
                gen_band = mask[y_top:y_bot, search_x0 : min(canvas_size, entry_x + 2)]
                if gen_band.any():
                    gy_rel, gx_rel = np.where(gen_band)
                    last_x = search_x0 + int(gx_rel.max())
                    cy = y_top + int(round(float(np.median(gy_rel[gx_rel >= gx_rel.max() - 3]))))
                    if last_x < entry_x + 2:
                        mask[max(0, cy - half_t) : min(canvas_size, cy + half_t + 1), max(0, last_x - 1) : min(canvas_size, entry_x + 3)] = True

    return mask


def _mask_to_svg(mask: np.ndarray, speckle_threshold: int = 6) -> str:
    img_arr = np.where(mask, 0, 255).astype(np.uint8)
    im = Image.fromarray(img_arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return vtracer.convert_raw_image_to_svg(
        buf.getvalue(),
        img_format="png",
        colormode="binary",
        mode="spline",
        filter_speckle=max(2, min(32, int(speckle_threshold))),
        corner_threshold=55,
        length_threshold=3.0,
        splice_threshold=45,
        path_precision=3,
    )


def _pil_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Main RunPod Serverless Handler
# ---------------------------------------------------------------------------
def handler(job: dict[str, Any]) -> dict[str, Any]:
    t_start = time.monotonic()
    job_input = job.get("input") or {}
    action = str(job_input.get("action") or "synthesize").strip().lower()

    if action == "diag":
        import shutil
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3), 1) if torch.cuda.is_available() else 0.0
        disk_app = shutil.disk_usage("/app")
        disk_tmp = shutil.disk_usage("/tmp")
        mem_info = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
                k, v = line.split(":", 1)
                mem_info[k.strip()] = v.strip()
        except Exception:
            pass
        return {
            "status": "ok",
            "gpu": gpu,
            "vram_gb": vram,
            "app_free_gb": round(disk_app.free / (1024.0 ** 3), 1),
            "tmp_free_gb": round(disk_tmp.free / (1024.0 ** 3), 1),
            "mem_info": mem_info,
            "model_root": str(_resolve_model_root()),
        }

    try:
        pipe, model_config = _ensure_pipeline()
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Pipeline init failed: {exc}", "traceback": traceback.format_exc()}

    if action in ("ping", "warmup"):
        return {
            "status": "ok",
            "engine": "glyphatlas_12b_flux_gpu",
            "gpu": _GPU_NAME,
            "vram_gb": round(_VRAM_GB, 1),
            "offload_text_encoders": _OFFLOAD_TEXT_ENCODERS,
            "elapsed_s": round(time.monotonic() - t_start, 2),
        }

    glyphs: list[dict[str, Any]] = job_input.get("glyphs") or []
    if not glyphs:
        return {"error": "No glyphs provided in input."}

    mode = str(job_input.get("generation_mode") or "exact_clone").strip().lower()
    preserve_connectors = bool(job_input.get("preserve_arabic_connectors", True))
    speckle_threshold = int(job_input.get("vtracer_speckle_threshold", 6))
    return_debug_sheets = bool(job_input.get("return_debug_sheets", False))
    return_masks = bool(job_input.get("return_masks", False))

    # Sampling hyperparameters (official GlyphAtlas uses steps=10, guidance=1.0)
    default_steps = 12 if mode == "exact_clone" else 14
    steps = int(job_input.get("glyphatlas_steps") or default_steps)
    steps = max(6, min(28, steps))

    default_guidance = 1.0 if mode == "exact_clone" else 2.0
    guidance = _snap_bf16_guidance(float(job_input.get("glyphatlas_guidance") or default_guidance))

    seed_base = int(job_input.get("differentiation_seed") or 42)
    prompt = _build_style_prompt(job_input)

    # Encode prompt ONCE for the entire font job
    prompt_embeds, pooled_prompt_embeds = _get_prompt_embeds(pipe, prompt)

    # Prepare unmasked top reference strip canvas (512x512 uint8 RGB)
    ref_canvas_u8 = _prepare_reference_canvas(
        job_input.get("reference_images_b64") or [],
        job_input.get("collage_patch_b64"),
    )
    # Convert to torch tensor [1, 3, 512, 512] float32 in [0.0, 1.0]
    img_tensor = torch.from_numpy(ref_canvas_u8).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    glyph_svgs: dict[str, str] = {}
    glyph_masks_b64: dict[str, str] = {}
    debug_sheets_b64: list[dict[str, str]] = []

    sheets_run = 0
    for batch_idx, start in enumerate(range(0, len(glyphs), 8)):
        batch = glyphs[start : start + 8]
        k = len(batch)

        cond_gray = np.zeros((512, 512), dtype=np.float32)
        hint_gray = np.zeros((512, 512), dtype=np.float32)
        tiles_64: list[np.ndarray] = []
        cell_masks: list[np.ndarray] = []
        parsed_items: list[tuple[dict[str, Any], np.ndarray]] = []

        for idx, g in enumerate(batch):
            x0, y0, x1, y1 = CELL_BOXES[idx]
            raw_png = base64.b64decode(g["skeleton_png_b64"])
            skel_im = Image.open(io.BytesIO(raw_png)).convert("L")
            skel_arr = np.asarray(skel_im, dtype=np.uint8)
            skel_ink = skel_arr < 128

            # White ink (255) on black (0) resized into INNER_BOX x INNER_BOX (112x112)
            inv_u8 = np.where(skel_ink, 255, 0).astype(np.uint8)
            small_u8 = np.asarray(
                Image.fromarray(inv_u8, mode="L").resize((INNER_BOX, INNER_BOX), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
            cond_gray[y0 + INNER_Y_PAD : y0 + INNER_Y_PAD + INNER_BOX, x0 : x0 + INNER_BOX] = (
                small_u8.astype(np.float32) / 255.0
            )

            # Mask full 112x144 cell (exact 16px latent boundaries!)
            hint_gray[y0:y1, x0:x1] = 1.0
            cm = np.zeros((512, 512), dtype=np.float32)
            cm[y0:y1, x0:x1] = 1.0
            cell_masks.append(cm)

            tiles_64.append(_make_64_atlas_tile(skel_ink))
            parsed_items.append((g, skel_ink))

        condition_3ch = np.stack([cond_gray, cond_gray, cond_gray], axis=-1)
        hint_3ch = np.stack([hint_gray, hint_gray, hint_gray], axis=-1)

        sheet_seed = (seed_base + batch_idx * 97) % (2 ** 31 - 1)
        with torch.inference_mode():
            atlas = _encode_custom_atlas(pipe, tiles_64, cell_masks)
            cond = Condition(
                condition_type="word_fill",
                condition=[condition_3ch, hint_3ch, img_tensor],
                position_delta=[0, 0],
            )
            gen_out = generate_fill(
                pipe,
                prompt=None,
                prompt_2=None,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                conditions=[cond],
                height=512,
                width=512,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator(device=pipe.device).manual_seed(sheet_seed),
                model_config=dict(model_config, atlas=atlas),
                default_lora=True,
            )
            result_pil: Image.Image = gen_out.images[0]

        result_rgb = np.asarray(result_pil.convert("RGB"), dtype=np.uint8)
        sheets_run += 1

        if return_debug_sheets and len(debug_sheets_b64) < 3:
            cond_vis = Image.fromarray(np.uint8(np.clip(cond_gray, 0, 1) * 255), mode="L")
            ref_vis = Image.fromarray(ref_canvas_u8, mode="RGB")
            debug_sheets_b64.append(
                {
                    "sheet_index": batch_idx,
                    "glyph_names": [g["glyph_name"] for g in batch],
                    "result_png_b64": _pil_to_b64(result_pil),
                    "condition_png_b64": _pil_to_b64(cond_vis),
                    "input_canvas_png_b64": _pil_to_b64(ref_vis),
                }
            )

        for idx, (g, skel_ink) in enumerate(parsed_items):
            x0, y0, x1, y1 = CELL_BOXES[idx]
            cell_rgb = result_rgb[y0 + INNER_Y_PAD : y0 + INNER_Y_PAD + INNER_BOX, x0 : x0 + INNER_BOX, :]
            canvas_size = int(g.get("canvas_size") or skel_ink.shape[0])

            out_mask = _extract_cell_mask(
                cell_rgb,
                skel_ink,
                canvas_size=canvas_size,
                origin_x_px=float(g.get("origin_x_px", canvas_size * 0.5)),
                baseline_y_px=float(g.get("baseline_y_px", canvas_size * 0.68)),
                advance_width=int(g.get("advance_width", 500)),
                scale=float(g.get("scale", canvas_size / 1620.0)),
                is_arabic_joining=bool(g.get("is_arabic_joining", False)),
                preserve_arabic_connectors=preserve_connectors,
            )

            gname = g["glyph_name"]
            glyph_svgs[gname] = _mask_to_svg(out_mask, speckle_threshold=speckle_threshold)
            if return_masks:
                m_u8 = np.where(out_mask, 0, 255).astype(np.uint8)
                glyph_masks_b64[gname] = _pil_to_b64(Image.fromarray(m_u8, mode="L"))

        del atlas, cond, gen_out
        if batch_idx % 6 == 5:
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    resp: dict[str, Any] = {
        "engine": "glyphatlas_12b_flux_gpu",
        "generation_mode": mode,
        "gpu": _GPU_NAME,
        "vram_gb": round(_VRAM_GB, 1),
        "sheets_run": sheets_run,
        "steps": steps,
        "guidance": guidance,
        "glyphs_processed": len(glyph_svgs),
        "elapsed_s": round(time.monotonic() - t_start, 2),
        "glyph_svgs": glyph_svgs,
    }
    if return_masks:
        resp["glyph_masks_b64"] = glyph_masks_b64
    if return_debug_sheets:
        resp["debug_sheets"] = debug_sheets_b64
    return resp


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
