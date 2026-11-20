"""
Qalam Foundry — RunPod Serverless Flex GPU Worker (Qwen-Image-2.1 + 4-Step Lightning LoRA)

True 1:1 Arabic + Latin neural font synthesis using Qwen/Qwen-Image-2.1
(compact 7B DiT + Qwen3-VL multimodal encoder, native Arabic & multi-reference image editing,
public ungated weights) accelerated with lightx2v/Qwen-Image-2.1-Lightning (4-step LoRA).

Architecture:
  - Packs up to 16 glyphs per 1024x1024 specimen sheet (4x4 grid, 256x256 per glyph!)
    OR 9 glyphs per 1024x1024 sheet (3x3 grid, 341x341 per glyph!).
  - Passes TWO crisp visual references into QwenImage21Pipeline:
      Image 1: HarfBuzz-shaped Arabic contextual + Latin glyph skeletons locked on exact
               baselines inside a clean 4x4 grid on pure #FFFFFF white.
      Image 2: The user's crisp reference typography image cleanly normalized onto #FFFFFF.
    (Also supports layout_mode="in_context" for single-canvas reference-above-grid A/B tests.)
  - Supports BOTH:
      * 4-step Lightning LoRA mode (~2-3s per 16-glyph sheet on H100 80GB!)
      * Full base multi-step mode (LoRA disabled, 16-30 steps for ultra-fine detail!)
  - Supports both `exact_clone` and `inspired` generation modes.
  - Extracts crisp anti-aliased masks (preserving ultra-thin hairlines & square dots),
    applies stroke-matched Arabic cursive connector locking, and traces smooth spline SVGs
    via vtracer.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import runpod
import torch
import vtracer
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from scipy import ndimage

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

BASE_MODEL_ID = "Qwen/Qwen-Image-2.1"
LIGHTNING_LORA_ID = "lightx2v/Qwen-Image-2.1-Lightning"
LIGHTNING_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

# ---------------------------------------------------------------------------
# Global persistent state across warm RunPod Serverless requests
# ---------------------------------------------------------------------------
_PIPE: QwenImage21Pipeline | None = None
_BASE_SCHEDULER: FlowMatchEulerDiscreteScheduler | None = None
_LIGHTNING_SCHEDULER: FlowMatchEulerDiscreteScheduler | None = None
_LORA_LOADED: bool = False
_LORA_ENABLED: bool = False
_GPU_NAME: str = "unknown"
_VRAM_GB: float = 0.0
_OFFLOAD_ENABLED: bool = False


def _resolve_model_root() -> Path:
    custom = os.environ.get("QWEN_MODEL_CACHE", "").strip()
    if custom:
        root = Path(custom)
        root.mkdir(parents=True, exist_ok=True)
        return root
    vol = Path("/runpod-volume")
    if vol.exists() and os.access(str(vol), os.W_OK):
        root = vol / "qwen21_models"
        root.mkdir(parents=True, exist_ok=True)
        return root
    # NEVER use /tmp (often RAM-backed tmpfs on cloud hosts!). Use container overlay disk /app/models_cache.
    root = Path("/app/models_cache")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _snapshot_with_retry(
    repo_id: str,
    cache_dir: Path,
    allow_patterns: list[str] | None = None,
    max_attempts: int = 3,
) -> Path:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.monotonic()
            print(
                f"[Qwen2.1] Downloading/verifying {repo_id} (attempt {attempt}/{max_attempts}) into {cache_dir} ...",
                flush=True,
            )
            kwargs: dict[str, Any] = {
                "repo_id": repo_id,
                "cache_dir": str(cache_dir),
                "max_workers": 2,
            }
            if allow_patterns is not None:
                kwargs["allow_patterns"] = allow_patterns
            snap = Path(snapshot_download(**kwargs))
            print(f"[Qwen2.1] {repo_id} ready at {snap} ({time.monotonic() - t0:.1f}s)", flush=True)
            return snap
        except Exception as exc:
            last_exc = exc
            print(f"[Qwen2.1] Download attempt {attempt} failed for {repo_id}: {exc}", flush=True)
            if attempt < max_attempts:
                time.sleep(4 * attempt)
    raise RuntimeError(f"Failed to download {repo_id} after {max_attempts} attempts: {last_exc}")


def _ensure_model_files(model_root: Path) -> tuple[Path, Path]:
    """
    Download public ungated weights ONCE directly into HF_HOME on container disk / Network Volume
    with ZERO duplicate local_dir copying (~30.9 GB total!):
      1) Qwen/Qwen-Image-2.1 (~30.86 GB: Qwen3-VL + 7B DiT + VAE)
      2) lightx2v/Qwen-Image-2.1-Lightning (0.06 GB 4-step LoRA)
    """
    hf_cache = model_root / "hf_cache"
    hub_dir = hf_cache / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    base_dir = _snapshot_with_retry(
        BASE_MODEL_ID,
        hub_dir,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
            "transformer/*",
        ],
    )
    lora_dir = _snapshot_with_retry(
        LIGHTNING_LORA_ID,
        hub_dir,
        allow_patterns=[LIGHTNING_WEIGHT_NAME],
    )
    return base_dir, lora_dir


def _ensure_pipeline() -> QwenImage21Pipeline:
    global _PIPE, _BASE_SCHEDULER, _LIGHTNING_SCHEDULER
    global _LORA_LOADED, _LORA_ENABLED, _GPU_NAME, _VRAM_GB, _OFFLOAD_ENABLED

    if _PIPE is not None:
        return _PIPE

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-2.1 worker requires a CUDA GPU with BF16 support.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bfloat16 required by Qwen-Image-2.1.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(0)
    _GPU_NAME = props.name
    _VRAM_GB = props.total_memory / (1024.0 ** 3)
    print(f"[Qwen2.1] Initializing on {_GPU_NAME} ({_VRAM_GB:.1f} GB VRAM)...", flush=True)

    model_root = _resolve_model_root()
    base_dir, lora_dir = _ensure_model_files(model_root)

    t0 = time.monotonic()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Save exact base scheduler before configuring Lightning scheduler
    _BASE_SCHEDULER = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    _LIGHTNING_SCHEDULER = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        shift_terminal=None,
    )

    # Load 4-step Lightning LoRA (can be toggled on/off per request with zero reload!)
    pipe.load_lora_weights(
        str(lora_dir),
        weight_name=LIGHTNING_WEIGHT_NAME,
        adapter_name="lightning",
    )
    _LORA_LOADED = True
    _LORA_ENABLED = True
    pipe.scheduler = _LIGHTNING_SCHEDULER

    if _VRAM_GB >= 44.0:
        _OFFLOAD_ENABLED = False
        pipe.to(device)
    else:
        _OFFLOAD_ENABLED = True
        pipe.enable_model_cpu_offload()

    pipe.set_progress_bar_config(disable=True)
    _PIPE = pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Qwen2.1] Pipeline + Lightning LoRA ready in {time.monotonic() - t0:.1f}s "
        f"(offload={_OFFLOAD_ENABLED}, vram={_VRAM_GB:.1f}GB, gpu={_GPU_NAME})",
        flush=True,
    )
    return _PIPE


def _set_sampling_mode(pipe: QwenImage21Pipeline, use_lightning: bool) -> None:
    global _LORA_ENABLED
    if use_lightning:
        if _LORA_LOADED and not _LORA_ENABLED:
            pipe.enable_lora()
            _LORA_ENABLED = True
        if _LIGHTNING_SCHEDULER is not None:
            pipe.scheduler = _LIGHTNING_SCHEDULER
    else:
        if _LORA_LOADED and _LORA_ENABLED:
            pipe.disable_lora()
            _LORA_ENABLED = False
        if _BASE_SCHEDULER is not None:
            pipe.scheduler = _BASE_SCHEDULER


# ---------------------------------------------------------------------------
# Reference & Skeleton Grid Builders
# ---------------------------------------------------------------------------
def _prepare_reference_image(
    reference_images_b64: list[str],
    collage_patch_b64: str | None,
    target_max_side: int = 1024,
) -> Image.Image:
    """
    Decode the user's primary reference image, normalize the background cleanly to pure #FFFFFF
    while preserving exact ink stroke anti-aliasing and crisp contrast, trim outer whitespace,
    and pad onto a clean rectangular #FFFFFF canvas so Qwen3-VL + VAE see every single detail.
    """
    src_pil: Image.Image | None = None
    for b64 in reference_images_b64 or []:
        if not b64:
            continue
        try:
            raw = base64.b64decode(b64)
            im = Image.open(io.BytesIO(raw))
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                src_pil = bg
            else:
                src_pil = im.convert("RGB")
            break
        except Exception:
            continue

    if src_pil is None and collage_patch_b64:
        raw = base64.b64decode(collage_patch_b64)
        src_pil = Image.open(io.BytesIO(raw)).convert("RGB")

    if src_pil is None:
        return Image.new("RGB", (512, 256), (255, 255, 255))

    gray = np.asarray(src_pil.convert("L"), dtype=np.uint8)
    # Invert if light text on dark background
    if float(np.median(gray)) < 120.0:
        gray = 255 - gray

    # Normalize contrast gently: push near-white bg (> p90) to 255, dark ink (< p2) to 0
    bg_hi = float(np.percentile(gray, 92))
    ink_lo = float(np.percentile(gray, 1))
    if bg_hi - ink_lo > 25.0:
        norm = np.clip((gray.astype(np.float32) - ink_lo) / (bg_hi - ink_lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = gray

    # Tightly crop non-white ink region with comfortable breathing margin
    ys, xs = np.where(norm < 220)
    if ys.size > 0 and xs.size > 0:
        h, w = norm.shape
        pad_y = max(12, int(0.12 * (ys.max() - ys.min() + 1)))
        pad_x = max(16, int(0.05 * (xs.max() - xs.min() + 1)))
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(h, int(ys.max()) + 1 + pad_y)
        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(w, int(xs.max()) + 1 + pad_x)
        norm = norm[y0:y1, x0:x1]

    # Ensure crisp readability: upscale tiny strips to at least 768px wide (capped at target_max_side)
    nh, nw = norm.shape
    scale = 1.0
    if max(nw, nh) > target_max_side:
        scale = target_max_side / float(max(nw, nh))
    elif nw < 768:
        scale = min(3.0, 768.0 / float(max(1, nw)))

    new_w = max(64, int(round(nw * scale)))
    new_h = max(64, int(round(nh * scale)))
    # Round to multiples of 16 for clean VAE tokenization
    new_w = max(64, ((new_w + 15) // 16) * 16)
    new_h = max(64, ((new_h + 15) // 16) * 16)

    resized = Image.fromarray(norm, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGB")
    return resized


def _build_skeleton_grid(
    batch: list[dict[str, Any]],
    *,
    grid_dim: int = 4,
    canvas_px: int = 1024,
    draw_grid_lines: bool = True,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], list[tuple[dict[str, Any], np.ndarray]]]:
    """
    Build a crisp black-on-white `canvas_px x canvas_px` specimen grid containing up to
    `grid_dim * grid_dim` baseline-anchored HarfBuzz glyph skeletons.
    """
    cell_px = canvas_px // grid_dim
    canvas = Image.new("RGB", (canvas_px, canvas_px), (255, 255, 255))
    boxes: list[tuple[int, int, int, int]] = []
    parsed_items: list[tuple[dict[str, Any], np.ndarray]] = []

    for idx, g in enumerate(batch):
        row = idx // grid_dim
        col = idx % grid_dim
        x0 = col * cell_px
        y0 = row * cell_px
        x1 = x0 + cell_px
        y1 = y0 + cell_px
        boxes.append((x0, y0, x1, y1))

        raw_png = base64.b64decode(g["skeleton_png_b64"])
        skel_im = Image.open(io.BytesIO(raw_png)).convert("L")
        skel_arr = np.asarray(skel_im, dtype=np.uint8)
        skel_ink = skel_arr < 128

        # Render clean black (#000000) ink on pure white (#FFFFFF) into cell
        cell_l = Image.fromarray(np.where(skel_ink, 0, 255).astype(np.uint8), mode="L").resize(
            (cell_px, cell_px), Image.Resampling.LANCZOS
        )
        canvas.paste(cell_l.convert("RGB"), (x0, y0))
        parsed_items.append((g, skel_ink))

    if draw_grid_lines:
        draw = ImageDraw.Draw(canvas)
        line_color = (224, 224, 224)
        for i in range(1, grid_dim):
            pos = i * cell_px
            draw.line([(pos, 0), (pos, canvas_px - 1)], fill=line_color, width=1)
            draw.line([(0, pos), (canvas_px - 1, pos)], fill=line_color, width=1)

    return canvas, boxes, parsed_items


def _build_in_context_canvas(
    ref_img: Image.Image,
    batch: list[dict[str, Any]],
    *,
    grid_dim: int = 4,
    canvas_px: int = 1024,
    ref_strip_h: int = 256,
    draw_grid_lines: bool = True,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], list[tuple[dict[str, Any], np.ndarray]]]:
    """
    Optional single-image in-context layout:
      - Top strip (0 .. ref_strip_h): crisp reference typography on pure white
      - Bottom area (ref_strip_h .. canvas_px): glyph skeleton grid (rows x cols)
    """
    canvas = Image.new("RGB", (canvas_px, canvas_px), (255, 255, 255))

    # Paste reference strip centered in top band
    rw, rh = ref_img.size
    max_w = canvas_px - 48
    max_h = ref_strip_h - 32
    sc = min(max_w / float(max(1, rw)), max_h / float(max(1, rh)))
    tw = max(16, int(round(rw * sc)))
    th = max(16, int(round(rh * sc)))
    ref_fit = ref_img.resize((tw, th), Image.Resampling.LANCZOS)
    canvas.paste(ref_fit, ((canvas_px - tw) // 2, (ref_strip_h - th) // 2))

    # Fit glyph grid into remaining height
    grid_h = canvas_px - ref_strip_h
    rows = min(grid_dim, max(1, (len(batch) + grid_dim - 1) // grid_dim))
    rows = max(rows, 3 if grid_dim == 4 else grid_dim)
    cell_w = canvas_px // grid_dim
    cell_h = grid_h // rows
    inner = min(cell_w, cell_h)

    boxes: list[tuple[int, int, int, int]] = []
    parsed_items: list[tuple[dict[str, Any], np.ndarray]] = []

    for idx, g in enumerate(batch[: rows * grid_dim]):
        row = idx // grid_dim
        col = idx % grid_dim
        x0 = col * cell_w + (cell_w - inner) // 2
        y0 = ref_strip_h + row * cell_h + (cell_h - inner) // 2
        x1 = x0 + inner
        y1 = y0 + inner
        boxes.append((x0, y0, x1, y1))

        raw_png = base64.b64decode(g["skeleton_png_b64"])
        skel_im = Image.open(io.BytesIO(raw_png)).convert("L")
        skel_arr = np.asarray(skel_im, dtype=np.uint8)
        skel_ink = skel_arr < 128

        cell_l = Image.fromarray(np.where(skel_ink, 0, 255).astype(np.uint8), mode="L").resize(
            (inner, inner), Image.Resampling.LANCZOS
        )
        canvas.paste(cell_l.convert("RGB"), (x0, y0))
        parsed_items.append((g, skel_ink))

    if draw_grid_lines:
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, ref_strip_h), (canvas_px - 1, ref_strip_h)], fill=(205, 205, 205), width=2)
        for c in range(1, grid_dim):
            draw.line([(c * cell_w, ref_strip_h), (c * cell_w, canvas_px - 1)], fill=(230, 230, 230), width=1)
        for r in range(1, rows):
            y = ref_strip_h + r * cell_h
            draw.line([(0, y), (canvas_px - 1, y)], fill=(230, 230, 230), width=1)

    return canvas, boxes, parsed_items


def _build_prompt(
    job_input: dict[str, Any],
    *,
    grid_dim: int,
    layout_mode: str,
) -> str:
    custom = str(job_input.get("custom_prompt") or "").strip()
    if custom:
        return custom

    mode = str(job_input.get("generation_mode") or "exact_clone").strip().lower()
    ref_tel = job_input.get("ref_telemetry") or {}
    dot_shape = str(ref_tel.get("dot_shape") or job_input.get("dot_style_override") or "").strip().lower()
    stroke_ratio = float(ref_tel.get("ref_stroke_ratio") or 0.0)
    contrast_ratio = float(ref_tel.get("stroke_contrast_ratio") or 1.0)
    extra_style = str(job_input.get("style_prompt") or "").strip()

    # Build objective physical cues from the reference image telemetry
    cues: list[str] = []
    if 0.0 < stroke_ratio <= 0.045:
        cues.append("ultra-thin delicate uniform monoline hairline stroke weight with zero calligraphic brush taper")
    elif 0.045 < stroke_ratio <= 0.085:
        cues.append("clean light-to-regular geometric monoline stroke weight")
    elif stroke_ratio > 0.12:
        cues.append("bold heavy display stroke weight matching the reference")

    if contrast_ratio <= 1.35:
        cues.append("strictly uniform monoline stroke thickness everywhere (no thick-thin Naskh pen contrast)")
    elif contrast_ratio >= 1.8:
        cues.append("distinct calligraphic thick-thin stroke contrast matching the reference")

    if dot_shape == "square":
        cues.append("crisp solid sharp-cornered square diacritic dots matching the reference")
    elif dot_shape == "diamond":
        cues.append("sharp diamond-shaped diacritic dots matching the reference")
    elif dot_shape == "round":
        cues.append("clean circular diacritic dots matching the reference")

    cue_str = ("; ".join(cues) + ". ") if cues else ""

    if layout_mode == "in_context":
        if mode == "inspired":
            base = (
                "Redraw every single black typography glyph in the bottom grid cells as an original, "
                "commercial brand typeface inspired by the design language and aesthetic of the reference "
                "typeface shown in the top strip, with refined distinctive curves and terminals. "
                f"{cue_str}"
                "Keep the top reference strip unchanged, and keep EVERY glyph in the bottom grid in its "
                "EXACT cell position, baseline alignment, character identity, Arabic contextual joining shape, "
                "and exact dot count. Pure flat #000000 black vector ink on pure #FFFFFF white background."
            )
        else:
            base = (
                "Redraw every single black typography glyph in the bottom grid cells so that EVERY glyph "
                "is a 1:1 EXACT typeface clone of the Arabic/Latin font shown in the top reference strip — "
                "matching its exact stroke thickness, exact geometric bowls, exact curves, exact terminals, "
                "and exact dot shape. "
                f"{cue_str}"
                "Keep the top reference strip unchanged, and keep EVERY glyph in the bottom grid in its "
                "EXACT cell position, baseline alignment, character identity, Arabic contextual joining form, "
                "and exact dot count/placement. Pure flat #000000 black 2D vector ink on pure #FFFFFF white "
                "background, no shading, no shadows, no texture."
            )
    else:
        grid_desc = f"{grid_dim}x{grid_dim}"
        if mode == "inspired":
            base = (
                f"Edit the first image (a {grid_desc} grid of black typography glyphs on pure white) to design "
                "an original, cohesive custom typeface inspired by the aesthetic and geometric proportions of "
                "the reference font shown in the second image, while introducing fresh refined terminals and "
                "distinctive character. "
                f"{cue_str}"
                f"Preserve EVERY glyph's exact {grid_desc} grid cell position, baseline alignment, character "
                "identity, Arabic contextual joining form (isolated/initial/medial/final), and exact dot count "
                "from the first image. Render razor-sharp flat pure black (#000000) 2D vector-style glyphs on "
                "pure white (#FFFFFF) with no shadows, no gradients, no extra text, and no borders."
            )
        else:
            base = (
                f"Edit the first image (a {grid_desc} grid of black typography glyphs on pure white) so that "
                "EVERY single glyph in the grid is redrawn as a 1:1 EXACT clone of the typeface shown in the "
                "second reference image — transferring its exact stroke thickness, exact monoline/contrast profile, "
                "exact wide circular bowls and geometric curves, exact stem terminals, and exact dot geometry. "
                f"{cue_str}"
                f"Strictly preserve EVERY glyph's exact {grid_desc} grid cell position, baseline height, "
                "character identity, Arabic contextual joining shape (isolated/initial/medial/final), and exact "
                "diacritic dot count and placement from the first image. Do NOT add or remove dots or letters. "
                "Render razor-sharp flat pure black (#000000) 2D vector-like glyphs on pure solid white (#FFFFFF) "
                "with zero shadows, zero gradients, zero textures, and zero extra symbols."
            )

    if extra_style and len(extra_style) < 260:
        base = f"{base} Additional typography notes: {extra_style}."
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
    trim_border_px: int = 5,
) -> np.ndarray:
    """
    Extract a razor-sharp binary glyph mask from a 256x256 (or 341x341) generated cell,
    preserving ultra-thin geometric hairlines and square dots, stripping any faint grid lines,
    and locking Arabic cursive baseline connectors at the generated glyph's own stroke weight.
    """
    gray_small = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)

    # Clear faint grid divider lines along the outer cell border BEFORE resizing
    if trim_border_px > 0 and min(gray_small.shape) > trim_border_px * 3:
        gray_small = gray_small.copy()
        gray_small[:trim_border_px, :] = 255
        gray_small[-trim_border_px:, :] = 255
        gray_small[:, :trim_border_px] = 255
        gray_small[:, -trim_border_px:] = 255

    gray_big = np.asarray(
        Image.fromarray(gray_small, mode="L").resize((canvas_size, canvas_size), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )

    # Adaptive luminance threshold tuned to preserve thin anti-aliased hairlines cleanly
    bg_med = float(np.percentile(gray_big, 95))
    ink_p1 = float(np.percentile(gray_big, 0.5))
    contrast = bg_med - ink_p1
    if contrast < 35.0:
        return skel_ink.copy()

    # Keep ultra-thin hairlines crisp (threshold at 58% from ink toward white, ignoring faint #E0E0E0 grid lines)
    thresh = int(round(min(205.0, max(120.0, ink_p1 + 0.58 * contrast))))
    mask = gray_big < thresh

    # Clear 3px outer border
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False

    # Remove tiny noise specks (< 10 px at 384px, preserving every real dot)
    labeled, num = ndimage.label(mask)
    if num > 0:
        min_area = max(8, int((canvas_size / 384.0) ** 2 * 10))
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_area
        keep[0] = False
        mask = keep[labeled]

    # Gentle anti-staircase smooth before vtracer
    if mask.any():
        blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=0.6)
        mask = blurred >= 0.48

    if int(mask.sum()) < 20:
        return skel_ink.copy()

    # Stroke-matched Arabic cursive baseline connector lock
    if preserve_arabic_connectors and is_arabic_joining:
        dist_gen = ndimage.distance_transform_edt(mask)
        nonzero_r = dist_gen[mask]
        stroke_half = float(np.percentile(nonzero_r, 65)) if nonzero_r.size > 0 else 2.0
        half_t = max(1, min(6, int(round(stroke_half))))

        y_top = max(0, int(round(baseline_y_px - 220.0 * scale)))
        y_bot = min(canvas_size, int(round(baseline_y_px + 70.0 * scale)))
        if y_bot > y_top:
            exit_x = max(2, min(canvas_size - 3, int(round(origin_x_px))))
            skel_exit_rows = np.where(
                skel_ink[y_top:y_bot, max(0, exit_x - 2) : min(canvas_size, exit_x + 6)].any(axis=1)
            )[0]
            if skel_exit_rows.size > 0:
                search_x1 = min(canvas_size, exit_x + int(round(28 * canvas_size / 384.0)))
                gen_band = mask[y_top:y_bot, exit_x:search_x1]
                if gen_band.any():
                    gy_rel, gx_rel = np.where(gen_band)
                    first_x = exit_x + int(gx_rel.min())
                    cy = y_top + int(round(float(np.median(gy_rel[gx_rel <= gx_rel.min() + 3]))))
                    if first_x > exit_x - 2:
                        mask[
                            max(0, cy - half_t) : min(canvas_size, cy + half_t + 1),
                            max(0, exit_x - 2) : first_x + 2,
                        ] = True

            entry_x = max(2, min(canvas_size - 3, int(round(origin_x_px + advance_width * scale))))
            skel_entry_rows = np.where(
                skel_ink[y_top:y_bot, max(0, entry_x - 6) : min(canvas_size, entry_x + 3)].any(axis=1)
            )[0]
            if skel_entry_rows.size > 0:
                search_x0 = max(0, entry_x - int(round(28 * canvas_size / 384.0)))
                gen_band = mask[y_top:y_bot, search_x0 : min(canvas_size, entry_x + 2)]
                if gen_band.any():
                    gy_rel, gx_rel = np.where(gen_band)
                    last_x = search_x0 + int(gx_rel.max())
                    cy = y_top + int(round(float(np.median(gy_rel[gx_rel >= gx_rel.max() - 3]))))
                    if last_x < entry_x + 2:
                        mask[
                            max(0, cy - half_t) : min(canvas_size, cy + half_t + 1),
                            max(0, last_x - 1) : min(canvas_size, entry_x + 3),
                        ] = True

    return mask


def _mask_to_svg(mask: np.ndarray, speckle_threshold: int = 4) -> str:
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
            "engine": "qwen_image_21_gpu",
            "gpu": gpu,
            "vram_gb": vram,
            "app_free_gb": round(disk_app.free / (1024.0 ** 3), 1),
            "tmp_free_gb": round(disk_tmp.free / (1024.0 ** 3), 1),
            "mem_info": mem_info,
            "model_root": str(_resolve_model_root()),
        }

    try:
        pipe = _ensure_pipeline()
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Pipeline init failed: {exc}", "traceback": traceback.format_exc()}

    if action in ("ping", "warmup"):
        return {
            "status": "ok",
            "engine": "qwen_image_21_gpu",
            "gpu": _GPU_NAME,
            "vram_gb": round(_VRAM_GB, 1),
            "offload": _OFFLOAD_ENABLED,
            "lora_loaded": _LORA_LOADED,
            "elapsed_s": round(time.monotonic() - t_start, 2),
        }

    glyphs: list[dict[str, Any]] = job_input.get("glyphs") or []
    if not glyphs:
        return {"error": "No glyphs provided in input."}

    mode = str(job_input.get("generation_mode") or "exact_clone").strip().lower()
    preserve_connectors = bool(job_input.get("preserve_arabic_connectors", True))
    speckle_threshold = int(job_input.get("vtracer_speckle_threshold", 4))
    return_debug_sheets = bool(job_input.get("return_debug_sheets", False))
    return_masks = bool(job_input.get("return_masks", False))

    # Runtime-tunable Qwen-Image-2.1 synthesis controls
    layout_mode = str(job_input.get("layout_mode") or "dual_image").strip().lower()
    grid_dim = int(job_input.get("grid_dim") or 4)
    grid_dim = max(2, min(4, grid_dim))
    canvas_px = int(job_input.get("canvas_px") or 1024)
    canvas_px = 1024 if canvas_px <= 1024 else 2048
    draw_grid_lines = bool(job_input.get("draw_grid_lines", True))
    use_lightning = bool(job_input.get("use_lightning", True))
    true_cfg_scale = float(job_input.get("true_cfg_scale", 1.0))
    negative_prompt = str(job_input.get("negative_prompt") or "").strip() or None

    if use_lightning:
        steps = int(job_input.get("qwen_steps") or 4)
        steps = max(4, min(8, steps))
        sigmas = [round(1.0 - i / float(steps), 4) for i in range(steps)]
    else:
        steps = int(job_input.get("qwen_steps") or 20)
        steps = max(8, min(40, steps))
        sigmas = None

    _set_sampling_mode(pipe, use_lightning=use_lightning)

    seed_base = int(job_input.get("differentiation_seed") or 42)
    prompt = _build_prompt(job_input, grid_dim=grid_dim, layout_mode=layout_mode)

    ref_img = _prepare_reference_image(
        job_input.get("reference_images_b64") or [],
        job_input.get("collage_patch_b64"),
        target_max_side=canvas_px,
    )

    per_sheet = (grid_dim * grid_dim) if layout_mode != "in_context" else (grid_dim * (3 if grid_dim == 4 else grid_dim))
    glyph_svgs: dict[str, str] = {}
    glyph_masks_b64: dict[str, str] = {}
    debug_sheets_b64: list[dict[str, Any]] = []

    sheets_run = 0
    gen_device = "cuda" if not _OFFLOAD_ENABLED else "cpu"

    for batch_idx, start in enumerate(range(0, len(glyphs), per_sheet)):
        batch = glyphs[start : start + per_sheet]

        if layout_mode == "in_context":
            skel_grid_img, boxes, parsed_items = _build_in_context_canvas(
                ref_img,
                batch,
                grid_dim=grid_dim,
                canvas_px=canvas_px,
                draw_grid_lines=draw_grid_lines,
            )
            cond_images: Any = skel_grid_img
        else:
            skel_grid_img, boxes, parsed_items = _build_skeleton_grid(
                batch,
                grid_dim=grid_dim,
                canvas_px=canvas_px,
                draw_grid_lines=draw_grid_lines,
            )
            cond_images = [skel_grid_img, ref_img]

        sheet_seed = (seed_base + batch_idx * 97) % (2 ** 31 - 1)
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": cond_images,
            "width": canvas_px,
            "height": canvas_px,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "use_kv_cache": not use_lightning,
            "generator": torch.Generator(device=gen_device).manual_seed(sheet_seed),
        }
        if sigmas is not None:
            call_kwargs["sigmas"] = sigmas
        if true_cfg_scale > 1.0 and negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        t_sheet = time.monotonic()
        with torch.inference_mode():
            out = pipe(**call_kwargs)
            result_pil: Image.Image = out.images[0]
        sheet_elapsed = round(time.monotonic() - t_sheet, 2)
        print(
            f"[Qwen2.1] Sheet {batch_idx} ({len(parsed_items)} glyphs, lightning={use_lightning}, "
            f"steps={steps}, layout={layout_mode}) done in {sheet_elapsed}s",
            flush=True,
        )

        if result_pil.size != (canvas_px, canvas_px):
            result_pil = result_pil.resize((canvas_px, canvas_px), Image.Resampling.LANCZOS)

        result_rgb = np.asarray(result_pil.convert("RGB"), dtype=np.uint8)
        sheets_run += 1

        if return_debug_sheets and len(debug_sheets_b64) < 4:
            debug_sheets_b64.append(
                {
                    "sheet_index": batch_idx,
                    "sheet_elapsed_s": sheet_elapsed,
                    "glyph_names": [g["glyph_name"] for g, _ in parsed_items],
                    "result_png_b64": _pil_to_b64(result_pil),
                    "skeleton_grid_png_b64": _pil_to_b64(skel_grid_img),
                    "reference_png_b64": _pil_to_b64(ref_img),
                }
            )

        trim_px = 6 if draw_grid_lines else 2
        for idx, (g, skel_ink) in enumerate(parsed_items):
            x0, y0, x1, y1 = boxes[idx]
            cell_rgb = result_rgb[y0:y1, x0:x1, :]
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
                trim_border_px=trim_px,
            )

            gname = g["glyph_name"]
            glyph_svgs[gname] = _mask_to_svg(out_mask, speckle_threshold=speckle_threshold)
            if return_masks:
                m_u8 = np.where(out_mask, 0, 255).astype(np.uint8)
                glyph_masks_b64[gname] = _pil_to_b64(Image.fromarray(m_u8, mode="L"))

        del out
        if batch_idx % 4 == 3:
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    resp: dict[str, Any] = {
        "engine": "qwen_image_21_gpu",
        "generation_mode": mode,
        "layout_mode": layout_mode,
        "use_lightning": use_lightning,
        "gpu": _GPU_NAME,
        "vram_gb": round(_VRAM_GB, 1),
        "sheets_run": sheets_run,
        "steps": steps,
        "grid_dim": grid_dim,
        "canvas_px": canvas_px,
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
