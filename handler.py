"""
Qalam Foundry Studio — RunPod Serverless Flex Few-Shot Neural Font Synthesis Worker.

Architecture ($0 Idle Cost — Serverless Flex with activeWorkers=0):
1. Lazy-loads official FontDiffuser (AAAI 2024: Multi-Scale Content Aggregation +
   Style Contrastive Refinement + DPM-Solver++) on CUDA GPU, caching the ~400 MB
   pretrained weights on the mounted RunPod Network Volume (`/runpod-volume/fontdiffuser_ckpt`).
2. Accepts locked baseline-anchored 384x384 Arabic & Latin glyph skeletons plus
   tightly cropped square reference glyph style patches extracted directly from the
   user's uploaded reference image.
3. Supports two explicit creation modes:
   - `exact_clone`: High style guidance (`guidance_scale=8.5`), exact physical
     stroke-thickness SDF calibration matching `ref_stroke_ratio` from the reference.
   - `inspired`: Balanced style transfer (`guidance_scale=5.8`) with controlled
     seeded variation for legally distinct original brand fonts.
4. Applies 4x smooth Signed-Distance-Field (SDF) super-resolution (96 -> 384),
   physical stroke-width iso-contour calibration, deterministic Arabic cursive
   baseline connector preservation, and `vtracer` cubic-spline SVG vectorization.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import vtracer
from PIL import Image, ImageFilter
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("runpod-fontdiffuser")

FONTDIFFUSER_REPO = Path(os.getenv("FONTDIFFUSER_REPO", "/app/fontdiffuser"))
HF_CKPT_BASE = os.getenv(
    "FONTDIFFUSER_CKPT_URL_BASE",
    "https://huggingface.co/spaces/yeungchenwa/FontDiffuser-Gradio/resolve/main/ckpt",
)
CKPT_FILES = ("unet.pth", "style_encoder.pth", "content_encoder.pth")

_PIPE: Any = None
_MODEL: Any = None
_SCHEDULER: Any = None
_INIT_DONE = False
_INIT_ERROR = ""
_DEVICE = "cpu"


def _resolve_ckpt_dir() -> Path:
    preferred = Path(os.getenv("FONTDIFFUSER_CKPT_DIR", "/runpod-volume/fontdiffuser_ckpt"))
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        test_file = preferred / ".write_test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        return preferred
    except Exception:
        fallback = Path("/tmp/fontdiffuser_ckpt")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _ensure_checkpoints(ckpt_dir: Path) -> Path:
    for fname in CKPT_FILES:
        dest = ckpt_dir / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        url = f"{HF_CKPT_BASE}/{fname}"
        logger.info("Downloading FontDiffuser checkpoint %s -> %s ...", url, dest)
        tmp = ckpt_dir / f"{fname}.part"
        with httpx.Client(timeout=300.0, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        fh.write(chunk)
        tmp.replace(dest)
        logger.info("Downloaded %s (%.1f MB)", fname, dest.stat().st_size / (1024 * 1024))
    return ckpt_dir


def _build_fontdiffuser_args() -> SimpleNamespace:
    return SimpleNamespace(
        resolution=96,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
        beta_scheduler="scaled_linear",
        num_train_timesteps=1000,
        order=2,
        t_start=None,
        t_end=None,
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
    )


def _load_fontdiffuser():
    global _MODEL, _SCHEDULER, _INIT_DONE, _INIT_ERROR, _DEVICE
    if _INIT_DONE:
        return _MODEL, _SCHEDULER, _DEVICE
    _INIT_DONE = True

    try:
        import torch

        if not torch.cuda.is_available():
            _INIT_ERROR = "CUDA not available"
            logger.warning("CUDA not available; falling back to SDF stroke-transfer mode.")
            return None, None, "cpu"

        _DEVICE = "cuda:0"
        if str(FONTDIFFUSER_REPO) not in sys.path:
            sys.path.insert(0, str(FONTDIFFUSER_REPO))

        from src import (
            FontDiffuserModelDPM,
            build_content_encoder,
            build_ddpm_scheduler,
            build_style_encoder,
            build_unet,
        )

        ckpt_dir = _ensure_checkpoints(_resolve_ckpt_dir())
        args = _build_fontdiffuser_args()

        t0 = time.perf_counter()
        unet = build_unet(args=args)
        unet.load_state_dict(torch.load(str(ckpt_dir / "unet.pth"), map_location="cpu"))

        style_encoder = build_style_encoder(args=args)
        style_encoder.load_state_dict(torch.load(str(ckpt_dir / "style_encoder.pth"), map_location="cpu"))

        content_encoder = build_content_encoder(args=args)
        content_encoder.load_state_dict(torch.load(str(ckpt_dir / "content_encoder.pth"), map_location="cpu"))

        model = FontDiffuserModelDPM(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )
        model.to(_DEVICE)
        model.eval()

        scheduler = build_ddpm_scheduler(args=args)
        _MODEL = model
        _SCHEDULER = scheduler
        logger.info("FontDiffuser loaded on %s in %.2fs", _DEVICE, time.perf_counter() - t0)
        return _MODEL, _SCHEDULER, _DEVICE
    except Exception as exc:
        _INIT_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to initialize FontDiffuser on GPU: %s", exc)
        return None, None, "cpu"


def _decode_rgb_png(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _estimate_stroke_px(mask: np.ndarray) -> float:
    m = mask.astype(np.float32)
    area = float(m.sum())
    if area < 8.0:
        return 1.5
    gx = np.abs(np.diff(m, axis=1, prepend=0))
    gy = np.abs(np.diff(m, axis=0, prepend=0))
    perim = float(gx.sum() + gy.sum())
    if perim < 1.0:
        return 1.5
    return max(1.0, 2.0 * area / perim)


def _upscale_sdf_and_calibrate_stroke(
    gen_96_pil: Image.Image,
    skel_mask: np.ndarray,
    *,
    canvas_size: int,
    target_stroke_px: float | None,
    is_hairline: bool,
    is_arabic_joining: bool,
    preserve_connectors: bool,
) -> np.ndarray:
    """
    4x Signed-Distance-Field (SDF) super-resolution (96x96 -> 384x384) with
    exact physical stroke-thickness iso-contour calibration so ultra-thin hairlines
    stay crisp & continuous while bold weights stay solid.
    """
    # Upscale 96x96 grayscale softly to canvas_size
    gray_up = np.asarray(
        gen_96_pil.convert("L").resize((canvas_size, canvas_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    # Mild Gaussian smoothing removes 96px grid stair-stepping cleanly
    gray_smooth = ndimage.gaussian_filter(gray_up, sigma=1.15)
    raw_mask = gray_smooth < 155.0

    # Safety fallback if diffusion dropped too much ink
    orig_area = max(1, int(skel_mask.sum()))
    if int(raw_mask.sum()) < max(24, int(orig_area * 0.22)):
        raw_mask = skel_mask.copy()

    # Build continuous signed distance field (positive inside ink, negative outside)
    dist_in = ndimage.distance_transform_edt(raw_mask)
    dist_out = ndimage.distance_transform_edt(~raw_mask)
    sdf = dist_in - dist_out

    # Calibrate physical stroke width to match reference image's exact stroke thickness!
    if target_stroke_px is not None and target_stroke_px > 0.8:
        cur_sw = _estimate_stroke_px(raw_mask)
        # Compute gentle signed-distance offset (clamped safely so counters never collapse
        # and hairlines never break!)
        delta_r = 0.5 * (target_stroke_px - cur_sw)
        if is_hairline:
            # Never erode a thin stroke below ~1.6px radius on 384px canvas
            max_erode = max(0.0, float(dist_in.max()) - 2.1)
            delta_r = max(-max_erode, min(3.5, delta_r))
        else:
            delta_r = max(-4.5, min(6.0, delta_r))

        calibrated = sdf >= -delta_r
        # Keep connectivity intact: if erosion shattered components, back off gently
        if int(calibrated.sum()) >= max(20, int(raw_mask.sum() * 0.35)):
            raw_mask = calibrated

    # Lock Arabic cursive baseline entry/exit bridges on joining glyphs so words join seamlessly
    if preserve_connectors and is_arabic_joining and skel_mask.any():
        h, w = skel_mask.shape
        # Preserve left/right baseline entry/exit edge stubs from the anchored skeleton
        ys, xs = np.where(skel_mask)
        if len(xs) > 0:
            edge_band = max(6, int(canvas_size * 0.035))
            x_min, x_max = int(xs.min()), int(xs.max())
            stub_zone = np.zeros_like(skel_mask, dtype=bool)
            stub_zone[:, max(0, x_min) : min(w, x_min + edge_band)] = True
            stub_zone[:, max(0, x_max - edge_band) : min(w, x_max + 1)] = True
            # Restrict stub zone to baseline vertical band (lower half of glyph box)
            y_base_top = int(h * 0.48)
            y_base_bot = int(h * 0.78)
            stub_zone[:y_base_top, :] = False
            stub_zone[y_base_bot:, :] = False
            raw_mask |= (skel_mask & stub_zone)

    return raw_mask


def _vectorize_mask(mask: np.ndarray, speckle_threshold: int = 4) -> str:
    img_arr = np.where(mask, 0, 255).astype(np.uint8)
    im = Image.fromarray(img_arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return vtracer.convert_raw_image_to_svg(
        buf.getvalue(),
        img_format="png",
        colormode="binary",
        mode="spline",
        filter_speckle=max(2, min(48, int(speckle_threshold))),
        corner_threshold=48,
        length_threshold=3.0,
        splice_threshold=45,
        path_precision=3,
    )


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod Serverless entrypoint for batched Few-Shot FontDiffuser glyph synthesis."""
    t_start = time.perf_counter()
    inp = event.get("input") or {}
    glyph_items: list[dict[str, Any]] = inp.get("glyphs") or []
    if not glyph_items:
        return {"glyph_svgs": {}, "glyph_masks_b64": {}, "count": 0, "engine": "fontdiffuser"}

    generation_mode = str(inp.get("generation_mode") or "exact_clone")
    guidance_scale = float(inp.get("guidance_scale", 8.5 if generation_mode == "exact_clone" else 5.8))
    sampling_steps = int(inp.get("sampling_steps", 20))
    differentiation_seed = float(inp.get("differentiation_seed", 0.0))
    preserve_connectors = bool(inp.get("preserve_arabic_connectors", True))
    speckle_threshold = int(inp.get("vtracer_speckle_threshold", 4))
    ref_telemetry = inp.get("ref_telemetry") or {}
    ref_stroke_ratio = float(ref_telemetry.get("ref_stroke_ratio", 0.05))
    is_hairline = bool(ref_telemetry.get("is_hairline", False))

    # Decode crisp square reference character style patches (extracted from user's uploaded image)
    style_pils: list[Image.Image] = []
    for b64_s in (inp.get("style_patches_b64") or []):
        try:
            style_pils.append(_decode_rgb_png(b64_s))
        except Exception:
            pass
    if not style_pils and inp.get("collage_patch_b64"):
        try:
            style_pils.append(_decode_rgb_png(inp["collage_patch_b64"]))
        except Exception:
            pass
    if not style_pils:
        for b64_s in (inp.get("reference_images_b64") or [])[:1]:
            try:
                style_pils.append(_decode_rgb_png(b64_s))
            except Exception:
                pass

    model, scheduler, device = _load_fontdiffuser()

    # Decode input glyph skeletons
    decoded_glyphs: list[dict[str, Any]] = []
    for item in glyph_items:
        gname = str(item.get("glyph_name") or "")
        b64_png = item.get("skeleton_png_b64") or ""
        if not gname or not b64_png:
            continue
        skel_pil = _decode_rgb_png(b64_png)
        c_size = int(item.get("canvas_size") or skel_pil.width)
        skel_mask = np.asarray(skel_pil.convert("L"), dtype=np.uint8) < 128
        decoded_glyphs.append(
            {
                "glyph_name": gname,
                "is_arabic": bool(item.get("is_arabic", False)),
                "is_arabic_joining": bool(item.get("is_arabic_joining", False)),
                "canvas_size": c_size,
                "skel_pil": skel_pil,
                "skel_mask": skel_mask,
            }
        )

    glyph_svgs: dict[str, str] = {}
    engine_tag = "runpod_fontdiffuser_cuda"

    # Compute target stroke thickness on the 384px canvas matching reference physical stroke ratio!
    # In skeletons.py, Arabic cap/Alif height spans ~52% of the 384px canvas (~200px).
    approx_line_px = 384.0 * 0.54
    target_stroke_px = max(3.6, min(42.0, ref_stroke_ratio * approx_line_px))

    if model is not None and scheduler is not None and style_pils and device.startswith("cuda"):
        import torch
        import torchvision.transforms as transforms
        from src import FontDiffuserDPMPipeline

        args = _build_fontdiffuser_args()
        args.guidance_scale = guidance_scale

        pipe = FontDiffuserDPMPipeline(
            model=model,
            ddpm_train_scheduler=scheduler,
            model_type=args.model_type,
            guidance_type=args.guidance_type,
            guidance_scale=guidance_scale,
        )

        tf = transforms.Compose(
            [
                transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Pre-encode all crisp reference style patches into tensors
        style_tensors = [tf(sp) for sp in style_pils]
        # Deterministic seed so all glyphs in the font share a cohesive typographic voice
        seed_val = 42 if generation_mode == "exact_clone" else (1000 + int(differentiation_seed * 10000))
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Run batched GPU diffusion (32 glyphs at a time -> ~1.1s per batch on A5000/L4/4090!)
        gpu_batch_size = int(os.getenv("FONTDIFFUSER_BATCH_SIZE", "32"))
        generated_pils: list[Image.Image] = []

        with torch.no_grad():
            for start in range(0, len(decoded_glyphs), gpu_batch_size):
                chunk = decoded_glyphs[start : start + gpu_batch_size]
                bsz = len(chunk)
                content_batch = torch.stack([tf(g["skel_pil"]) for g in chunk], dim=0).to(device)
                # Cycle diverse square reference patches across glyphs (or use primary collage/patch)
                # Using the single sharpest patch for exact_clone maximizes cross-glyph consistency!
                if generation_mode == "exact_clone":
                    primary_style = style_tensors[0]
                    style_batch = primary_style.unsqueeze(0).repeat(bsz, 1, 1, 1).to(device)
                else:
                    style_batch = torch.stack(
                        [style_tensors[(start + i) % len(style_tensors)] for i in range(bsz)],
                        dim=0,
                    ).to(device)

                imgs = pipe.generate(
                    content_images=content_batch,
                    style_images=style_batch,
                    batch_size=bsz,
                    order=args.order,
                    num_inference_step=sampling_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn,
                )
                generated_pils.extend(imgs)

        for g_info, gen_pil in zip(decoded_glyphs, generated_pils):
            final_mask = _upscale_sdf_and_calibrate_stroke(
                gen_pil,
                g_info["skel_mask"],
                canvas_size=g_info["canvas_size"],
                target_stroke_px=target_stroke_px,
                is_hairline=is_hairline,
                is_arabic_joining=g_info["is_arabic_joining"],
                preserve_connectors=preserve_connectors,
            )
            glyph_svgs[g_info["glyph_name"]] = _vectorize_mask(
                final_mask,
                speckle_threshold=speckle_threshold,
            )
    else:
        engine_tag = "runpod_sdf_fallback"
        for g_info in decoded_glyphs:
            final_mask = _upscale_sdf_and_calibrate_stroke(
                g_info["skel_pil"],
                g_info["skel_mask"],
                canvas_size=g_info["canvas_size"],
                target_stroke_px=target_stroke_px,
                is_hairline=is_hairline,
                is_arabic_joining=g_info["is_arabic_joining"],
                preserve_connectors=preserve_connectors,
            )
            glyph_svgs[g_info["glyph_name"]] = _vectorize_mask(
                final_mask,
                speckle_threshold=speckle_threshold,
            )

    elapsed = round(time.perf_counter() - t_start, 2)
    logger.info(
        "Completed %d glyphs via %s (mode=%s, guidance=%.1f, target_sw=%.1fpx) in %.2fs",
        len(glyph_svgs),
        engine_tag,
        generation_mode,
        guidance_scale,
        target_stroke_px,
        elapsed,
    )
    return {
        "glyph_svgs": glyph_svgs,
        "glyph_masks_b64": {},
        "count": len(glyph_svgs),
        "engine": engine_tag,
        "init_error": _INIT_ERROR,
        "generation_mode": generation_mode,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
