#!/usr/bin/env python3
"""
re_pairgen.py

Generate synthetic real-estate BEFORE/AFTER photo enhancement pairs.

Goals:
- Create NEW photorealistic real-estate scenes (text → image)
- Create a matched AFTER version that preserves the scene and applies only
  pro photo edits (exposure/WB/contrast/window pull/perspective/etc.)

Providers:
- Gemini image models via google-genai SDK (env: GEMINI_API_KEY)
- OpenAI image models via openai SDK (env: OPENAI_API_KEY)

Output format (compatible with this repo’s JSONL tooling):
- images/{id}_src.jpg   (BEFORE)
- images/{id}_tar.jpg   (AFTER)
- train.jsonl           (one JSON object per line with src/tar + metadata)
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

try:  # optional
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:  # optional
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


# ----------------------------
# Prompt system (scene recipes)
# ----------------------------

ROOM_TYPES = [
    "living room",
    "kitchen",
    "primary bedroom",
    "bathroom",
    "dining room",
    "home office",
    "entryway / foyer",
    "hallway",
    "laundry room",
    "exterior front of house",
    "backyard patio / deck",
]

HOME_STYLES = [
    "modern",
    "scandinavian",
    "transitional",
    "farmhouse",
    "mid-century modern",
    "coastal",
    "industrial loft",
    "traditional",
    "contemporary luxury",
]

LIGHTING_SITUATIONS = [
    "bright midday sun outside with a darker interior",
    "late afternoon sun outside creating strong highlights",
    "overcast exterior light with dim interior corners",
    "mixed lighting: daylight plus warm interior lamps",
]

WINDOW_SETUPS = [
    "large bright windows in the frame",
    "a sliding glass door with a backyard view",
    "multiple side windows with trees visible outside",
    "a bright window behind the main subject area",
]

CAMERA_NOTES_MLS = [
    "wide-angle interior real-estate photo, tripod-mounted, eye-level, 16mm lens look, level horizon",
    "wide-angle shot, straight-on, slightly elevated camera height, 18mm lens look, verticals mostly straight",
    "real-estate MLS style, wide lens, mostly level framing, mild perspective distortion typical of interiors",
]

CAMERA_NOTES_AMATEUR = [
    "handheld smartphone wide-angle photo (0.5x), taken by a homeowner, slightly tilted (~2–5 degrees), not perfectly centered, mild barrel distortion but NO fisheye look",
    "handheld phone photo from a doorway corner at chest height, slightly off-level and imperfect framing, a little too much floor or ceiling, mild barrel distortion but NO fisheye",
    "casual smartphone interior snapshot, slightly under-framed or over-framed, not symmetrical, not tripod, mild perspective convergence, NO fisheye",
]

FLOORS = [
    "light oak hardwood floors",
    "medium walnut hardwood floors",
    "light gray tile floors",
    "warm beige carpet",
    "polished concrete floors",
]

WALLS = [
    "white walls",
    "soft warm beige walls",
    "light gray walls",
    "greige walls",
]

DECOR = [
    "minimal staging with neutral decor",
    "tasteful decor with framed art and a few plants",
    "simple, clean decor with neutral textiles",
]

OUTDOOR_VIEWS = [
    "lush green trees and a grassy yard",
    "a suburban backyard with a fence and shrubs",
    "a quiet residential street with trees",
    "a patio with outdoor furniture and plants",
]


@dataclass(frozen=True)
class SceneRecipe:
    room_type: str
    home_style: str
    lighting: str
    window_setup: str
    camera: str
    floor: str
    walls: str
    decor: str
    outdoor_view: str
    severity: int  # 1..5 controls how strong the before/after gap is
    capture_style: str  # "amateur" | "mls"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def random_recipe(
    rng: random.Random,
    *,
    severity_min: int = 2,
    severity_max: int = 5,
    capture_style: str = "amateur",
) -> SceneRecipe:
    camera_notes = CAMERA_NOTES_AMATEUR if capture_style == "amateur" else CAMERA_NOTES_MLS
    return SceneRecipe(
        room_type=rng.choice(ROOM_TYPES),
        home_style=rng.choice(HOME_STYLES),
        lighting=rng.choice(LIGHTING_SITUATIONS),
        window_setup=rng.choice(WINDOW_SETUPS),
        camera=rng.choice(camera_notes),
        floor=rng.choice(FLOORS),
        walls=rng.choice(WALLS),
        decor=rng.choice(DECOR),
        outdoor_view=rng.choice(OUTDOOR_VIEWS),
        severity=rng.randint(severity_min, severity_max),
        capture_style=capture_style,
    )


def build_base_scene_prompt(r: SceneRecipe) -> str:
    return (
        f"Create a photorealistic smartphone photo of a {r.room_type} in a residential home. "
        "The photo is taken by an amateur homeowner during a listing walkthrough (NOT a professional photographer). "
        f"Home style: {r.home_style}. "
        f"{r.decor}. "
        f"The room has {r.floor} and {r.walls}. "
        f"Lighting situation: {r.lighting}. "
        f"Window situation: {r.window_setup}. "
        f"Outside the windows you can see {r.outdoor_view}. "
        f"Camera/composition: {r.camera}. "
        "No fisheye. Looks like a real photo (not CGI/3D). Not professionally staged. "
        "No people. No pets. No text. No logos. No watermarks."
    )


def before_generation_modifier(r: SceneRecipe) -> str:
    sev = r.severity
    if r.capture_style == "amateur":
        if sev <= 2:
            capture = (
                "Handheld smartphone snapshot quality: slight tilt, minor softness, light noise, "
                "and mild JPEG compression artifacts. "
            )
        elif sev == 3:
            capture = (
                "Handheld smartphone snapshot quality: slight tilt, minor handshake softness, moderate noise, "
                "and visible JPEG compression artifacts. "
            )
        else:
            capture = (
                "Handheld smartphone snapshot quality: slight tilt, a bit of softness from handshake, noticeable noise, "
                "and visible JPEG compression artifacts. "
            )
    else:
        capture = "Unedited camera JPEG (not HDR), slight noise, normal consumer lens distortion. "

    if sev <= 2:
        extra = (
            "Slightly underexposed interior, mild shadow heaviness, and mild mixed white balance. "
            "Windows are a bit too bright with limited exterior detail."
        )
    elif sev == 3:
        extra = (
            "Moderately underexposed interior, shadows are heavy in corners, "
            "mixed color temperature (warm lamps + cool daylight), "
            "windows are significantly overexposed and mostly blown out."
        )
    elif sev == 4:
        extra = (
            "Noticeably underexposed interior with heavy shadows, "
            "strong mixed white balance cast (warm/orange lamps and cool/blue daylight), "
            "windows are blown out with almost no exterior detail visible."
        )
    else:
        extra = (
            "Dark, moody single exposure with crushed shadows in corners, "
            "strong mixed white balance cast, "
            "windows are completely blown out (pure white) with zero exterior detail."
        )

    return (
        "This should look like an UNEDITED, single ambient exposure straight out of camera. "
        "Not HDR. Not flambient. No flash. "
        f"{capture}"
        "Dull contrast and muted colors. "
        + extra
    )


def before_degrade_modifier(r: SceneRecipe) -> str:
    """
    Edit a base image into a deliberately amateur-looking BEFORE without changing the scene.
    Strongly emphasize: NO geometry changes, NO content changes.
    """
    sev = r.severity

    if sev <= 2:
        quality = (
            "Slightly underexpose the interior (about 0.5–1 stop), add mild shadows in corners, "
            "introduce a mild warm/cool white balance mismatch, and add a small amount of high-ISO noise."
        )
    elif sev == 3:
        quality = (
            "Underexpose the interior (about 1–2 stops), make shadows heavier in corners, "
            "introduce noticeable mixed white balance (warm indoor light + cool daylight), "
            "add moderate high-ISO noise and a touch of motion blur."
        )
    elif sev == 4:
        quality = (
            "Heavily underexpose the interior (about 2–3 stops), crush shadow detail slightly, "
            "add strong mixed white balance and a mild green/magenta cast, "
            "add visible high-ISO noise and slight softness from missed focus or handshake."
        )
    else:
        quality = (
            "Make it clearly a bad amateur shot: very dark interior (about 3–4 stops under), "
            "strong color cast from mixed lighting, heavy noise, and noticeable softness/motion blur, "
            "while still keeping the scene recognizable."
        )

    return (
        "Using the provided image, create a BEFORE version that looks like a casual amateur smartphone photo. "
        "Keep the camera viewpoint/framing EXACTLY the same. "
        "Do NOT crop, rotate, straighten, warp, or change perspective. "
        "Do NOT add/remove/move/replace any objects (including anything visible through windows). "
        "Do NOT retouch or clean anything.\n\n"
        + quality
        + "\n\n"
        "Make highlights a bit clipped and interior a bit muddy; dull colors; mild JPEG compression artifacts."
    )


def after_edit_modifier(r: SceneRecipe) -> str:
    """
    Edit an image into a professional-looking AFTER without changing the scene.
    Strictly forbid geometry/content changes to keep pairs aligned.
    """
    sev = r.severity
    if sev <= 2:
        strength = (
            "Make a noticeable but natural improvement: lift exposure and shadows, neutralize white balance, "
            "improve contrast and color slightly, and reduce noise."
        )
    elif sev == 3:
        strength = (
            "Make a clear professional improvement: brighten the interior, lift shadows without banding, "
            "neutralize mixed white balance, improve local contrast, and reduce noise."
        )
    elif sev == 4:
        strength = (
            "Make a strong improvement: significantly brighten the interior, recover highlights where possible, "
            "neutralize strong mixed white balance, reduce haze, reduce noise, and add crisp (but realistic) clarity."
        )
    else:
        strength = (
            "Make a dramatic-but-realistic professional improvement: bring exposure to a bright, airy look, "
            "lift shadows substantially, correct strong color cast, reduce heavy noise, and restore clarity "
            "WITHOUT inventing details."
        )

    return (
        "Using the provided image, create an AFTER version that looks professionally edited for a real-estate listing. "
        "ABSOLUTE RULES:\n"
        "- Keep the camera viewpoint/framing EXACTLY the same.\n"
        "- Do NOT crop, rotate, straighten, warp, or change perspective.\n"
        "- Do NOT add/remove/move/replace any objects or textures.\n"
        "- Do NOT change anything visible through windows; do NOT invent exterior details.\n"
        "- Do NOT do virtual staging.\n"
        "- Do NOT retouch/erase stains, scuffs, cracks, cords, or clutter.\n\n"
        "Allowed edits ONLY: exposure and tone curve, highlight/shadow balancing, white balance correction, "
        "color correction, mild local contrast/clarity, mild dehaze, denoise, and gentle sharpening. "
        "If highlights are clipped (e.g., windows), only recover what is truly present—do not hallucinate.\n\n"
        + strength
    )


def after_edit_with_reference_prefix() -> str:
    return (
        "You will be provided multiple images.\n"
        "- Image 1: REFERENCE BEFORE (unedited)\n"
        "- Image 2: REFERENCE AFTER (professionally edited)\n"
        "- Image 3: TARGET image to edit\n"
        "Use Image 1→2 only to understand the *type and strength* of the edit. "
        "Do NOT recreate the reference scene. "
        "Apply the same kind of enhancement to Image 3 while preserving its content.\n"
        "Strictly keep Image 3's framing/geometry and all objects identical; no crop/rotate/warp; no hallucinated details.\n"
    )


# ----------------------------
# Reference dataset helpers
# ----------------------------


@dataclass(frozen=True)
class ReferencePair:
    before: Path
    after: Path


def load_reference_pairs(dataset_dir: Path, *, limit: Optional[int] = None) -> List[ReferencePair]:
    jsonl_path = dataset_dir / "train.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Expected {jsonl_path} to exist")

    pairs: List[ReferencePair] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            before = dataset_dir / entry["src"]
            after = dataset_dir / entry["tar"]
            if before.exists() and after.exists():
                pairs.append(ReferencePair(before=before, after=after))
            if limit is not None and len(pairs) >= limit:
                break

    if not pairs:
        raise RuntimeError(f"No valid reference pairs found in {jsonl_path}")
    return pairs


# ----------------------------
# Providers
# ----------------------------


class ProviderError(RuntimeError):
    pass


class BaseProvider:
    name: str

    def generate(self, prompt: str, *, out_path: Path, seed: Optional[int] = None) -> None:
        raise NotImplementedError

    def edit(
        self,
        prompt: str,
        *,
        image_paths: Sequence[Path],
        out_path: Path,
        seed: Optional[int] = None,
        edit_kind: Optional[str] = None,  # "before" | "after" (best-effort hint)
        recipe: Optional[SceneRecipe] = None,
    ) -> None:
        raise NotImplementedError


def _save_pil_as_jpeg(img: Image.Image, out_path: Path, *, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=quality, subsampling=0, optimize=True)


def _save_bytes_as_jpeg(image_bytes: bytes, out_path: Path, *, quality: int = 95) -> None:
    img = Image.open(io.BytesIO(image_bytes))
    _save_pil_as_jpeg(img, out_path, quality=quality)


def _lut_scale(scale: float) -> List[int]:
    scale = max(0.0, scale)
    return [min(255, max(0, int(round(i * scale)))) for i in range(256)]


def _lut_gamma(gamma: float) -> List[int]:
    # gamma > 1 darkens, < 1 brightens
    gamma = max(0.01, float(gamma))
    return [min(255, max(0, int(round(((i / 255.0) ** gamma) * 255.0)))) for i in range(256)]


def _apply_rgb_luts(img: Image.Image, *, r_lut: List[int], g_lut: List[int], b_lut: List[int]) -> Image.Image:
    r, g, b = img.convert("RGB").split()
    r = r.point(r_lut)
    g = g.point(g_lut)
    b = b.point(b_lut)
    return Image.merge("RGB", (r, g, b))


def _gray_world_scales(img: Image.Image) -> Tuple[float, float, float]:
    small = img.convert("RGB").resize((64, 64), Image.BILINEAR)
    pixels = list(small.getdata())
    if not pixels:
        return 1.0, 1.0, 1.0
    r_mean = sum(p[0] for p in pixels) / len(pixels)
    g_mean = sum(p[1] for p in pixels) / len(pixels)
    b_mean = sum(p[2] for p in pixels) / len(pixels)
    gray = (r_mean + g_mean + b_mean) / 3.0
    # clamp scaling to avoid wild shifts
    r_scale = max(0.6, min(1.6, gray / (r_mean + 1e-6)))
    g_scale = max(0.6, min(1.6, gray / (g_mean + 1e-6)))
    b_scale = max(0.6, min(1.6, gray / (b_mean + 1e-6)))
    return r_scale, g_scale, b_scale


def _jpeg_roundtrip(img: Image.Image, *, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=2, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _add_noise(img: Image.Image, *, sigma: float) -> Image.Image:
    noise = Image.effect_noise(img.size, sigma).convert("RGB")
    # center noise around 0 by offsetting
    return ImageChops.add(img.convert("RGB"), noise, scale=1.0, offset=-128)


def degrade_to_before(img: Image.Image, *, severity: int, rng: random.Random) -> Image.Image:
    severity = int(max(1, min(5, severity)))
    out = img.convert("RGB")

    # Exposure + gamma (make shadows muddy)
    gamma = {1: 1.18, 2: 1.30, 3: 1.45, 4: 1.60, 5: 1.80}[severity]
    out = _apply_rgb_luts(out, r_lut=_lut_gamma(gamma), g_lut=_lut_gamma(gamma), b_lut=_lut_gamma(gamma))
    out = ImageEnhance.Brightness(out).enhance({1: 0.94, 2: 0.88, 3: 0.80, 4: 0.72, 5: 0.66}[severity])
    out = ImageEnhance.Contrast(out).enhance({1: 0.97, 2: 0.95, 3: 0.93, 4: 0.91, 5: 0.89}[severity])

    # Mixed WB / color cast
    cast = rng.choice(["warm", "cool", "green", "magenta"])
    if cast == "warm":
        rs, gs, bs = (1.10, 1.00, 0.90)
    elif cast == "cool":
        rs, gs, bs = (0.92, 1.00, 1.10)
    elif cast == "green":
        rs, gs, bs = (0.95, 1.10, 0.95)
    else:  # magenta
        rs, gs, bs = (1.05, 0.92, 1.05)

    # stronger cast at higher severity
    strength = {1: 0.35, 2: 0.45, 3: 0.60, 4: 0.75, 5: 0.90}[severity]
    rs = 1.0 + (rs - 1.0) * strength
    gs = 1.0 + (gs - 1.0) * strength
    bs = 1.0 + (bs - 1.0) * strength
    out = _apply_rgb_luts(out, r_lut=_lut_scale(rs), g_lut=_lut_scale(gs), b_lut=_lut_scale(bs))

    # Dull colors
    out = ImageEnhance.Color(out).enhance({1: 0.95, 2: 0.92, 3: 0.88, 4: 0.82, 5: 0.78}[severity])

    # Blur / softness + noise
    blur_r = {1: 0.2, 2: 0.35, 3: 0.5, 4: 0.7, 5: 0.9}[severity]
    out = out.filter(ImageFilter.GaussianBlur(radius=blur_r))
    out = _add_noise(out, sigma={1: 6, 2: 9, 3: 13, 4: 18, 5: 24}[severity])

    # Mild JPEG artifacts
    out = _jpeg_roundtrip(out, quality={1: 85, 2: 78, 3: 70, 4: 62, 5: 55}[severity])
    return out


def enhance_to_after(img: Image.Image, *, severity: int) -> Image.Image:
    severity = int(max(1, min(5, severity)))
    out = img.convert("RGB")

    # White balance correction (gray world)
    rs, gs, bs = _gray_world_scales(out)
    out = _apply_rgb_luts(out, r_lut=_lut_scale(rs), g_lut=_lut_scale(gs), b_lut=_lut_scale(bs))

    # Brighten + contrast, proportional to severity
    out = ImageEnhance.Brightness(out).enhance({1: 1.10, 2: 1.18, 3: 1.28, 4: 1.40, 5: 1.55}[severity])
    out = ImageEnhance.Contrast(out).enhance({1: 1.05, 2: 1.08, 3: 1.12, 4: 1.16, 5: 1.20}[severity])
    out = ImageEnhance.Color(out).enhance({1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.10}[severity])

    # Denoise-ish: slight blur then unsharp
    out = out.filter(ImageFilter.GaussianBlur(radius={1: 0.2, 2: 0.3, 3: 0.4, 4: 0.45, 5: 0.5}[severity]))
    out = out.filter(
        ImageFilter.UnsharpMask(radius=2, percent={1: 120, 2: 140, 3: 160, 4: 175, 5: 190}[severity], threshold=3)
    )
    return out


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(
        self,
        *,
        model: str,
        aspect_ratio: str,
        image_size: Optional[str],
        jpeg_quality: int = 95,
    ):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ProviderError("Missing dependency: google-genai (pip install google-genai)") from e

        self._genai = genai
        self._types = types
        self._client = genai.Client()
        self._model = model
        self._aspect_ratio = aspect_ratio
        self._image_size = image_size
        self._jpeg_quality = jpeg_quality

    def _build_config(self, *, seed: Optional[int]) -> Any:
        kwargs: Dict[str, Any] = {
            "response_modalities": ["IMAGE"],
            "image_config": self._types.ImageConfig(
                aspect_ratio=self._aspect_ratio,
                **({"image_size": self._image_size} if self._image_size else {}),
            ),
        }
        if seed is not None:
            kwargs["seed"] = seed
        try:
            return self._types.GenerateContentConfig(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            return self._types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _iter_parts(resp: Any) -> Iterable[Any]:
        parts = getattr(resp, "parts", None)
        if parts is not None:
            return parts
        # fallback shape
        try:
            return resp.candidates[0].content.parts  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            return []

    def _save_first_image(self, resp: Any, out_path: Path) -> None:
        for part in self._iter_parts(resp):
            inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if not data:
                continue
            _save_bytes_as_jpeg(data, out_path, quality=self._jpeg_quality)
            return
        raise ProviderError("Gemini response did not contain an image (prompt may have been blocked).")

    def generate(self, prompt: str, *, out_path: Path, seed: Optional[int] = None) -> None:
        cfg = self._build_config(seed=seed)
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config=cfg,
        )
        self._save_first_image(resp, out_path)

    def edit(
        self,
        prompt: str,
        *,
        image_paths: Sequence[Path],
        out_path: Path,
        seed: Optional[int] = None,
        edit_kind: Optional[str] = None,
        recipe: Optional[SceneRecipe] = None,
    ) -> None:
        cfg = self._build_config(seed=seed)
        images: List[Image.Image] = []
        for p in image_paths:
            with Image.open(p) as img:
                images.append(img.copy())
        resp = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, *images],
            config=cfg,
        )
        self._save_first_image(resp, out_path)


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, *, model: str, size: str, jpeg_quality: int = 95):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ProviderError("Missing dependency: openai (pip install openai)") from e

        self._client = OpenAI()
        self._model = model
        self._size = size
        self._jpeg_quality = jpeg_quality
        self._local_edit_mode = False

    @staticmethod
    def _looks_like_org_verification_error(err: BaseException) -> bool:
        msg = str(err).lower()
        return "must be verified" in msg and "organization" in msg

    @staticmethod
    def _normalize_size(model: str, size: str) -> str:
        model = model.lower()
        if model == "dall-e-3":
            allowed = {"1024x1024", "1792x1024", "1024x1792"}
            if size in allowed:
                return size
            try:
                w_str, h_str = size.split("x")
                w, h = int(w_str), int(h_str)
            except Exception:
                return "1024x1024"
            if w > h:
                return "1792x1024"
            if h > w:
                return "1024x1792"
            return "1024x1024"
        # gpt-image-1 accepts "auto" and common rectangular sizes; we keep as-is.
        return size

    def generate(self, prompt: str, *, out_path: Path, seed: Optional[int] = None) -> None:
        # seed is best-effort (SDK/API may reject it); fall back cleanly.
        model = self._model
        size = self._normalize_size(model, self._size)
        params: Dict[str, Any] = {"model": model, "prompt": prompt, "n": 1, "size": size}
        if model.lower().startswith("dall-e"):
            params["response_format"] = "b64_json"
        if seed is not None:
            params["seed"] = seed

        try:
            rsp = self._client.images.generate(**params)
        except Exception as e:
            # If gpt-image-* is blocked (org verification), fall back to dall-e-3 generation.
            if model.lower().startswith("gpt-image") and self._looks_like_org_verification_error(e):
                print(
                    "[warn] OpenAI gpt-image-* unavailable (org not verified). Falling back to dall-e-3 generation "
                    "and local (non-AI) edits for strict no-drift pairs.",
                    file=sys.stderr,
                )
                self._local_edit_mode = True
                model = "dall-e-3"
                size = self._normalize_size(model, self._size)
                params = {"model": model, "prompt": prompt, "n": 1, "size": size, "response_format": "b64_json"}
                rsp = self._client.images.generate(**params)
            else:
                params.pop("seed", None)
                rsp = self._client.images.generate(**params)

        b64_data = getattr(rsp.data[0], "b64_json", None)
        if not b64_data:
            raise ProviderError("OpenAI images.generate did not return b64_json; set response_format=b64_json.")
        img_bytes = base64.b64decode(b64_data)
        _save_bytes_as_jpeg(img_bytes, out_path, quality=self._jpeg_quality)

    def edit(
        self,
        prompt: str,
        *,
        image_paths: Sequence[Path],
        out_path: Path,
        seed: Optional[int] = None,
        edit_kind: Optional[str] = None,
        recipe: Optional[SceneRecipe] = None,
    ) -> None:
        # If requested model can't do edits (or is blocked), do local deterministic edits
        # to guarantee exact geometry / no content drift.
        model = self._model.lower()
        if self._local_edit_mode or model == "dall-e-3":
            target_path = image_paths[-1]
            with Image.open(target_path) as img:
                base = img.copy()
            sev = recipe.severity if recipe else 4
            if edit_kind == "before" or ("before version" in prompt.lower()):
                rnd = random.Random((seed or 0) ^ hash(out_path.name))
                out_img = degrade_to_before(base, severity=sev, rng=rnd)
                q = min(self._jpeg_quality, 70)
            else:
                out_img = enhance_to_after(base, severity=sev)
                q = self._jpeg_quality
            _save_pil_as_jpeg(out_img, out_path, quality=q)
            return

        # API edit path (best-effort; may still drift)
        from contextlib import ExitStack

        size = self._normalize_size(self._model, self._size)
        params: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "output_format": "png",
            "input_fidelity": "high",
        }
        if seed is not None:
            params["seed"] = seed

        with ExitStack() as stack:
            # OpenAI images.edit expects a single file for `image` (even if you pass multiple inputs in prompt text).
            target_file = stack.enter_context(Path(image_paths[-1]).open("rb"))
            params["image"] = target_file
            try:
                rsp = self._client.images.edit(**params)
            except Exception as e:
                if model.startswith("gpt-image") and self._looks_like_org_verification_error(e):
                    print(
                        "[warn] OpenAI gpt-image-* edit unavailable (org not verified). Falling back to local edits.",
                        file=sys.stderr,
                    )
                    self._local_edit_mode = True
                    return self.edit(
                        prompt,
                        image_paths=image_paths,
                        out_path=out_path,
                        seed=seed,
                        edit_kind=edit_kind,
                        recipe=recipe,
                    )
                params.pop("seed", None)
                rsp = self._client.images.edit(**params)

        b64_data = rsp.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)
        _save_bytes_as_jpeg(img_bytes, out_path, quality=self._jpeg_quality)


# ----------------------------
# Orchestration
# ----------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def retry_call(fn, *, retries: int, base_sleep_s: float, retry_sleep_s: float) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001
            last_err = e
            if attempt >= retries:
                break
            sleep_s = base_sleep_s * (2**attempt) + retry_sleep_s * random.random()
            print(f"[warn] {e} (retrying in {sleep_s:.1f}s)", file=sys.stderr)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def build_provider(args: argparse.Namespace) -> BaseProvider:
    if args.provider == "gemini":
        return GeminiProvider(
            model=args.gemini_model,
            aspect_ratio=args.aspect,
            image_size=args.gemini_image_size,
            jpeg_quality=args.jpeg_quality,
        )
    if args.provider == "openai":
        return OpenAIProvider(
            model=args.openai_model,
            size=args.openai_size,
            jpeg_quality=args.jpeg_quality,
        )
    raise ValueError(f"Unknown provider: {args.provider}")


def make_output_paths(out_dir: Path, pair_id: int) -> Dict[str, Path]:
    images_dir = out_dir / "images"
    return {
        "before": images_dir / f"{pair_id:06d}_src.jpg",
        "after": images_dir / f"{pair_id:06d}_tar.jpg",
        "base": images_dir / f"{pair_id:06d}_base.jpg",
    }


def generate_pair(
    *,
    provider: BaseProvider,
    out_dir: Path,
    pair_id: int,
    recipe: SceneRecipe,
    pipeline: str,
    retries: int,
    sleep_s: float,
    reference: Optional[ReferencePair],
    save_base: bool,
    model_seed: Optional[int],
) -> Dict[str, Any]:
    paths = make_output_paths(out_dir, pair_id)
    before_path = paths["before"]
    after_path = paths["after"]
    base_path = paths["base"]

    base_scene_prompt = build_base_scene_prompt(recipe)

    meta: Dict[str, Any] = {
        "id": pair_id,
        "created_at": _now_iso(),
        "provider": provider.name,
        "pipeline": pipeline,
        "recipe": recipe.to_dict(),
        "src": f"images/{before_path.name}",
        "tar": f"images/{after_path.name}",
        "prompts": {},
    }

    def _gen(p: str, out_path: Path) -> None:
        retry_call(
            lambda: provider.generate(p, out_path=out_path, seed=model_seed),
            retries=retries,
            base_sleep_s=1.5,
            retry_sleep_s=1.0,
        )
        _sleep(sleep_s)

    def _edit(p: str, img_paths: Sequence[Path], out_path: Path, *, kind: str) -> None:
        retry_call(
            lambda: provider.edit(
                p,
                image_paths=img_paths,
                out_path=out_path,
                seed=model_seed,
                edit_kind=kind,
                recipe=recipe,
            ),
            retries=retries,
            base_sleep_s=1.5,
            retry_sleep_s=1.0,
        )
        _sleep(sleep_s)

    if pipeline == "before_to_after":
        before_prompt = base_scene_prompt + "\n\n" + before_generation_modifier(recipe)

        if reference is not None:
            after_prompt = after_edit_with_reference_prefix() + after_edit_modifier(recipe)
            after_inputs = [reference.before, reference.after, before_path]
            meta["reference"] = {"before": str(reference.before), "after": str(reference.after)}
        else:
            after_prompt = after_edit_modifier(recipe)
            after_inputs = [before_path]

        meta["prompts"]["before_generation"] = before_prompt
        meta["prompts"]["after_edit"] = after_prompt

        _gen(before_prompt, before_path)
        _edit(after_prompt, after_inputs, after_path, kind="after")

    elif pipeline == "base_to_both":
        neutral_base_prompt = base_scene_prompt + "\n\n" + (
            "Looks like an average unedited phone/consumer camera photo (single exposure). "
            "Not HDR, not flash, not professionally edited. Natural but imperfect."
        )
        before_prompt = before_degrade_modifier(recipe)

        if reference is not None:
            after_prompt = after_edit_with_reference_prefix() + after_edit_modifier(recipe)
            after_inputs = [reference.before, reference.after, base_path]
            meta["reference"] = {"before": str(reference.before), "after": str(reference.after)}
        else:
            after_prompt = after_edit_modifier(recipe)
            after_inputs = [base_path]

        meta["prompts"]["base_generation"] = neutral_base_prompt
        meta["prompts"]["before_degrade_edit"] = before_prompt
        meta["prompts"]["after_edit"] = after_prompt

        _gen(neutral_base_prompt, base_path)
        _edit(before_prompt, [base_path], before_path, kind="before")
        _edit(after_prompt, after_inputs, after_path, kind="after")

        if not save_base and base_path.exists():
            try:
                base_path.unlink()
            except OSError:
                pass
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    return meta


def generate_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta_path = out_dir / "run_meta.json"
    run_meta = {
        "created_at": _now_iso(),
        "cmd": "generate",
        "args": vars(args),
    }
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)

    reference_pairs: Optional[List[ReferencePair]] = None
    if args.ref_dataset:
        reference_pairs = load_reference_pairs(Path(args.ref_dataset))

    provider: Optional[BaseProvider]
    if args.dry_run:
        provider = None
    else:
        provider = build_provider(args)

    train_jsonl = out_dir / "train.jsonl"
    successes = 0
    failures = 0

    if args.resume and train_jsonl.exists() and not args.dry_run:
        pair_id = infer_next_pair_id(train_jsonl, default_start=args.start_index)
    else:
        pair_id = args.start_index

    iterator = range(args.num)
    if tqdm and args.progress:
        iterator = tqdm(iterator, total=args.num)

    for _ in iterator:
        while True:
            recipe = random_recipe(
                rng,
                severity_min=args.severity_min,
                severity_max=args.severity_max,
                capture_style=args.capture_style,
            )
            reference: Optional[ReferencePair] = None
            if reference_pairs and args.ref_pairs > 0:
                reference = rng.choice(reference_pairs)

            try:
                if args.dry_run:
                    preview = {
                        "id": pair_id,
                        "recipe": recipe.to_dict(),
                        "pipeline": args.pipeline,
                        "provider": args.provider,
                        "prompts": {
                            "base": build_base_scene_prompt(recipe),
                            "before_generation": before_generation_modifier(recipe),
                            "before_degrade_edit": before_degrade_modifier(recipe),
                            "after_edit": after_edit_modifier(recipe),
                        },
                        "reference": (
                            {"before": str(reference.before), "after": str(reference.after)} if reference else None
                        ),
                    }
                    write_jsonl_line(out_dir / "prompts_preview.jsonl", preview)
                    successes += 1
                    pair_id += 1
                    break

                assert provider is not None
                meta = generate_pair(
                    provider=provider,
                    out_dir=out_dir,
                    pair_id=pair_id,
                    recipe=recipe,
                    pipeline=args.pipeline,
                    retries=args.retries,
                    sleep_s=args.sleep,
                    reference=reference,
                    save_base=args.save_base,
                    model_seed=args.model_seed,
                )
                write_jsonl_line(train_jsonl, meta)
                successes += 1
                pair_id += 1
                break
            except BaseException as e:  # noqa: BLE001
                failures += 1
                print(f"[error] pair {pair_id}: {e}", file=sys.stderr)

                # best-effort cleanup of partial outputs for this id
                paths = make_output_paths(out_dir, pair_id)
                for p in [paths["before"], paths["after"], paths["base"]]:
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass

                if not args.continue_on_error:
                    raise
                if failures >= args.max_failures:
                    raise SystemExit(f"Too many failures ({failures}); aborting.")
                # try a new recipe (same pair_id)
                continue

    print(f"Done. Wrote {successes} pairs to {out_dir}")


def enhance_folder(args: argparse.Namespace) -> None:
    provider = build_provider(args)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    if not images:
        raise SystemExit(f"No images found in {in_dir}")

    generic_recipe = SceneRecipe(
        room_type="interior room",
        home_style="residential",
        lighting="mixed lighting",
        window_setup="windows present",
        camera="wide-angle MLS photo",
        floor="",
        walls="",
        decor="",
        outdoor_view="",
        severity=4,
        capture_style="amateur",
    )
    prompt = after_edit_modifier(generic_recipe)

    iterator: Iterable[Path] = images
    if tqdm and args.progress:
        iterator = tqdm(images, total=len(images))

    for img_path in iterator:
        out_path = out_dir / f"{img_path.stem}_after.jpg"
        retry_call(
            lambda: provider.edit(
                prompt,
                image_paths=[img_path],
                out_path=out_path,
                seed=args.model_seed,
                edit_kind="after",
                recipe=generic_recipe,
            ),
            retries=args.retries,
            base_sleep_s=1.5,
            retry_sleep_s=1.0,
        )
        _sleep(args.sleep)

    print(f"Done. Enhanced {len(images)} images into {out_dir}")


def warn_missing_keys(provider: str) -> None:
    if provider == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print(
            "[warn] GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export it or add it to a .env file.",
            file=sys.stderr,
        )
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "[warn] OPENAI_API_KEY is not set. Export it or add it to a .env file.",
            file=sys.stderr,
        )


_PAIR_ID_RE = re.compile(r"(\d+)_src\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def infer_next_pair_id(train_jsonl: Path, *, default_start: int) -> int:
    max_id: Optional[int] = None
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            candidate: Optional[int] = None
            if isinstance(entry.get("id"), int):
                candidate = int(entry["id"])
            else:
                src = entry.get("src")
                if isinstance(src, str):
                    m = _PAIR_ID_RE.search(Path(src).name)
                    if m:
                        candidate = int(m.group(1))

            if candidate is None:
                continue
            if max_id is None or candidate > max_id:
                max_id = candidate

    return (max_id + 1) if max_id is not None else default_start


def main() -> None:
    load_env_files()

    parser = argparse.ArgumentParser(description="Generate synthetic real-estate BEFORE/AFTER pairs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic BEFORE/AFTER pairs.")
    p_gen.add_argument("--provider", choices=["gemini", "openai"], required=True)
    p_gen.add_argument("--num", type=int, default=10, help="Number of PAIRS to generate (each pair = before+after).")
    p_gen.add_argument("--out", type=str, required=True, help="Output folder for this run.")
    p_gen.add_argument("--seed", type=int, default=1234, help="RNG seed for recipe sampling.")
    p_gen.add_argument("--pipeline", choices=["before_to_after", "base_to_both"], default="base_to_both")
    p_gen.add_argument("--retries", type=int, default=3, help="Retries per API call.")
    p_gen.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls.")
    p_gen.add_argument("--progress", action="store_true", help="Show progress bar (if tqdm is installed).")

    p_gen.add_argument("--start-index", type=int, default=0, help="Starting numeric id for naming images.")
    p_gen.add_argument("--resume", action="store_true", help="Resume from the highest id in existing train.jsonl.")
    p_gen.add_argument("--save-base", action="store_true", help="Keep base images when using base_to_both.")

    p_gen.add_argument("--severity-min", type=int, default=2)
    p_gen.add_argument("--severity-max", type=int, default=5)
    p_gen.add_argument(
        "--capture-style",
        choices=["amateur", "mls"],
        default="amateur",
        help="Controls camera/framing style for the generated scene (composition is part of the scene).",
    )

    p_gen.add_argument("--continue-on-error", action="store_true")
    p_gen.add_argument("--max-failures", type=int, default=25)

    p_gen.add_argument("--dry-run", action="store_true", help="Write prompts_preview.jsonl but do not call APIs.")

    p_gen.add_argument("--model-seed", type=int, default=None, help="Best-effort provider seed (if supported).")

    p_gen.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved images.")

    p_gen.add_argument("--ref-dataset", type=str, default=None, help="Dataset dir containing train.jsonl + images/")
    p_gen.add_argument("--ref-pairs", type=int, default=0, help="Use 1 reference BEFORE/AFTER pair for the AFTER edit.")

    p_gen.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-image")
    p_gen.add_argument("--aspect", type=str, default="16:9")
    p_gen.add_argument("--gemini-image-size", type=str, default=None, help="For gemini-3-pro-image-preview: 1K/2K/4K")

    p_gen.add_argument("--openai-model", type=str, default="gpt-image-1")
    p_gen.add_argument(
        "--openai-size",
        type=str,
        default="1536x1024",
        help='OpenAI size: "auto" or "1024x1024" / "1536x1024" / "1024x1536"',
    )

    p_enh = sub.add_parser("enhance-folder", help="Enhance a folder of BEFORE images into AFTER images.")
    p_enh.add_argument("--provider", choices=["gemini", "openai"], required=True)
    p_enh.add_argument("--in", dest="in_dir", type=str, required=True)
    p_enh.add_argument("--out", dest="out_dir", type=str, required=True)
    p_enh.add_argument("--retries", type=int, default=3)
    p_enh.add_argument("--sleep", type=float, default=0.0)
    p_enh.add_argument("--progress", action="store_true")
    p_enh.add_argument("--model-seed", type=int, default=None)
    p_enh.add_argument("--jpeg-quality", type=int, default=95)

    p_enh.add_argument("--gemini-model", type=str, default="gemini-2.5-flash-image")
    p_enh.add_argument("--aspect", type=str, default="16:9")
    p_enh.add_argument("--gemini-image-size", type=str, default=None)

    p_enh.add_argument("--openai-model", type=str, default="gpt-image-1")
    p_enh.add_argument("--openai-size", type=str, default="1536x1024")

    args = parser.parse_args()

    if args.cmd == "generate":
        if args.num <= 0:
            raise SystemExit("--num must be > 0")
        if not args.dry_run:
            warn_missing_keys(args.provider)
        generate_run(args)
        return

    if args.cmd == "enhance-folder":
        warn_missing_keys(args.provider)
        enhance_folder(args)
        return

    raise SystemExit("Unknown command")


def load_env_files() -> None:
    """
    Load API keys from .env files without relying on python-dotenv's stack
    inspection (which can be brittle across Python versions/environments).
    """
    if not load_dotenv:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            # override=False: prefer already-exported shell vars
            load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception as e:  # pragma: no cover
            print(f"[warn] Failed to load {env_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
