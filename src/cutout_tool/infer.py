from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


MODEL_INPUT_SIZE = (1024, 1024)
DEFAULT_MODEL_PATH = Path("models/birefnet-dis5k.onnx")
MODEL_SOURCE = "https://huggingface.co/onnx-community/BiRefNet-DIS5K-ONNX"
MATTING_MODEL_SOURCE = "https://huggingface.co/emrikol/birefnet-matting-onnx"
MODEL_REVISION = "main; file pinned by SHA-256"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class RunResult:
    input_path: str
    output_dir: str
    model_path: str
    model_source: str
    model_revision: str
    model_sha256: str
    model_input: list[int]
    model_content_box: list[int]
    original_size: list[int]
    output_size: list[int]
    providers: list[str]
    ort_device: str
    elapsed_ms: float
    output_files: dict[str, str]
    alpha_min: int
    alpha_max: int
    alpha_nonzero_pixels: int
    alpha_partial_pixels: int
    success: bool
    error: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_source_for_path(path: Path) -> str:
    if path.name == DEFAULT_MODEL_PATH.name:
        return MODEL_SOURCE
    if path.name == "birefnet-matting.onnx":
        return MATTING_MODEL_SOURCE
    return "unknown"


def prepare_model_image(
    image: Image.Image,
    size: tuple[int, int] = MODEL_INPUT_SIZE,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands():
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")

    source_width, source_height = image.size
    target_width, target_height = size
    scale = min(target_width / source_width, target_height / source_height)
    fitted_size = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
    fitted = image.resize(fitted_size, Image.Resampling.LANCZOS)
    left = (target_width - fitted_size[0]) // 2
    top = (target_height - fitted_size[1]) // 2
    fitted_array = np.asarray(fitted, dtype=np.uint8)
    padded_array = np.pad(
        fitted_array,
        ((top, target_height - fitted_size[1] - top),
         (left, target_width - fitted_size[0] - left),
         (0, 0)),
        mode="edge",
    )
    padded = Image.fromarray(padded_array)
    return padded, (left, top, left + fitted_size[0], top + fitted_size[1])


def _image_to_tensor(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def prepare_input(image: Image.Image, size: tuple[int, int] = MODEL_INPUT_SIZE) -> np.ndarray:
    prepared, _ = prepare_model_image(image, size)
    return _image_to_tensor(prepared)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def alpha_from_prediction(
    prediction: np.ndarray,
    size: tuple[int, int],
    content_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    values = np.asarray(prediction)
    if values.ndim == 4:
        values = values[0, 0]
    elif values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"unexpected model output shape: {prediction.shape}")
    alpha = np.clip(sigmoid(values) * 255.0, 0, 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha)
    if content_box is not None:
        alpha_image = alpha_image.crop(content_box)
    return alpha_image.resize(size, Image.Resampling.LANCZOS)


def composite(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    layer = Image.new("RGB", image.size, background)
    return Image.alpha_composite(layer.convert("RGBA"), image).convert("RGB")


def save_outputs(image: Image.Image, alpha: Image.Image, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    paths = {
        "transparent": output_dir / f"{stem}_transparent.png",
        "preview_black": output_dir / f"{stem}_preview_black.png",
        "preview_white": output_dir / f"{stem}_preview_white.png",
        "alpha": output_dir / f"{stem}_alpha.png",
    }
    rgba.save(paths["transparent"], format="PNG")
    composite(rgba, (0, 0, 0)).save(paths["preview_black"], format="PNG")
    composite(rgba, (255, 255, 255)).save(paths["preview_white"], format="PNG")
    alpha.save(paths["alpha"], format="PNG")
    return paths


def run(
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    session: ort.InferenceSession | None = None,
) -> RunResult:
    started = time.perf_counter()
    model_hash = sha256_file(model_path)
    with Image.open(input_path) as opened:
        input_format = opened.format
        source = ImageOps.exif_transpose(opened)
        source.load()
        original = source.convert("RGBA")
    original_size = original.size
    prepared, content_box = prepare_model_image(source if input_format == "JPEG" else original)
    input_tensor = _image_to_tensor(prepared)
    session = session or ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: input_tensor})[0]
    alpha = alpha_from_prediction(prediction, original_size, content_box)
    source_alpha = np.asarray(original.getchannel("A"), dtype=np.uint8)
    if source_alpha.min() < 255:
        alpha = Image.fromarray(np.minimum(np.asarray(alpha), source_alpha).astype(np.uint8))
    stem = input_path.stem
    paths = save_outputs(original, alpha, output_dir, stem)
    alpha_array = np.asarray(alpha)
    result = RunResult(
        input_path=str(input_path.resolve()),
        output_dir=str(output_dir.resolve()),
        model_path=str(model_path.resolve()),
        model_source=model_source_for_path(model_path),
        model_revision=MODEL_REVISION,
        model_sha256=model_hash,
        model_input=list(MODEL_INPUT_SIZE),
        model_content_box=list(content_box),
        original_size=list(original_size),
        output_size=list(original_size),
        providers=session.get_providers(),
        ort_device=ort.get_device(),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        output_files={key: str(path.resolve()) for key, path in paths.items()},
        alpha_min=int(alpha_array.min()),
        alpha_max=int(alpha_array.max()),
        alpha_nonzero_pixels=int(np.count_nonzero(alpha_array)),
        alpha_partial_pixels=int(np.count_nonzero((alpha_array > 0) & (alpha_array < 255))),
        success=True,
    )
    (output_dir / "run.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local BiRefNet ONNX inference.")
    parser.add_argument("--input", type=Path, required=True, help="path to an unprocessed input image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args.input, args.output_dir, args.model)
    except Exception as exc:  # noqa: BLE001 - CLI must persist a failed run record.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "success": False,
            "input_path": str(args.input.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "model_path": str(args.model.resolve()),
            "model_source": model_source_for_path(args.model),
            "model_revision": MODEL_REVISION,
            "error": f"{type(exc).__name__}: {exc}",
            "python": sys.version,
            "platform": platform.platform(),
        }
        (args.output_dir / "run.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0
