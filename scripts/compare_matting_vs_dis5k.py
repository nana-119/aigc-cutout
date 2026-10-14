"""Compare the two local BiRefNet ONNX models without changing the default pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cutout_tool.infer import (  # noqa: E402
    MODEL_INPUT_SIZE,
    _image_to_tensor,
    alpha_from_prediction,
    composite,
    prepare_model_image,
    sha256_file,
)

MODELS = {
    "matting": ROOT / "models" / "birefnet-matting.onnx",
    "dis5k": ROOT / "models" / "birefnet-dis5k.onnx",
}
DIS5K_SHA256 = "6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def checkerboard_composite(original: Image.Image, alpha: Image.Image) -> Image.Image:
    tile = Image.new("RGB", (64, 64), (235, 235, 235))
    dark = Image.new("RGB", (32, 32), (205, 205, 205))
    tile.paste(dark, (0, 0))
    tile.paste(dark, (32, 32))
    background = Image.new("RGB", original.size)
    for top in range(0, original.height, 64):
        for left in range(0, original.width, 64):
            background.paste(tile, (left, top))
    foreground = original.copy()
    foreground.putalpha(alpha)
    return Image.alpha_composite(background.convert("RGBA"), foreground).convert("RGB")


def run_model(name: str, model_path: Path, rows: list[dict]) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"{name}: CPUExecutionProvider was not selected")
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.shape != [1, 3, 1024, 1024] or model_input.type != "tensor(float)":
        raise ValueError(f"{name}: unexpected input shape or type")
    if model_output.shape != [1, 1, 1024, 1024] or model_output.type != "tensor(float)":
        raise ValueError(f"{name}: unexpected output shape or type")

    for row in rows:
        source_path = Path(row["source"])
        image_dir = Path(row["output_dir"])
        with Image.open(source_path) as opened:
            input_format = opened.format
            source = ImageOps.exif_transpose(opened)
            source.load()
            original = source.convert("RGBA")
        prepared, content_box = prepare_model_image(source if input_format == "JPEG" else original)
        tensor = _image_to_tensor(prepared)
        started = time.perf_counter()
        prediction = session.run([model_output.name], {model_input.name: tensor})[0]
        inference_ms = round((time.perf_counter() - started) * 1000, 2)
        alpha = alpha_from_prediction(prediction, original.size, content_box)
        source_alpha = np.asarray(original.getchannel("A"), dtype=np.uint8)
        if source_alpha.min() < 255:
            alpha = Image.fromarray(np.minimum(np.asarray(alpha), source_alpha).astype(np.uint8))
        alpha.save(image_dir / f"{name}_alpha.png")
        transparent = original.copy()
        transparent.putalpha(alpha)
        transparent.save(image_dir / f"{name}_transparent.png")
        composite(transparent, (0, 0, 0)).save(image_dir / f"{name}_preview_black.png")
        composite(transparent, (255, 255, 255)).save(image_dir / f"{name}_preview_white.png")
        checkerboard_composite(original, alpha).save(image_dir / f"{name}_composite.png")
        row[name] = {
            "success": True,
            "cpu_inference_ms": inference_ms,
            "raw_output_min": float(prediction.min()),
            "raw_output_max": float(prediction.max()),
            "postprocess": "sigmoid_logits",
            "error": None,
        }
        print(f"{source_path.name}: {name} {inference_ms:.2f} ms", flush=True)

    del session
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path, help="folder containing source images")
    args = parser.parse_args()
    folder = args.image_folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"not an image folder: {folder}")
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"no supported images in: {folder}")
    for model_path in MODELS.values():
        if not model_path.is_file():
            parser.error(f"model missing: {model_path}")
    hashes = {name: sha256_file(path) for name, path in MODELS.items()}
    if hashes["dis5k"] != DIS5K_SHA256:
        parser.error("DIS5K model SHA-256 mismatch; refusing inference")

    output_root = ROOT / "outputs" / f"compare-matting-dis5k-{datetime.now():%Y%m%d-%H%M%S}"
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, source_path in enumerate(images, 1):
        image_dir = output_root / f"{index:03d}-{source_path.stem}"
        image_dir.mkdir()
        shutil.copy2(source_path, image_dir / f"original{source_path.suffix.lower()}")
        rows.append({"source": str(source_path), "output_dir": str(image_dir)})

    record = {
        "success": False,
        "preprocessing": "infer.py: EXIF transpose, aspect fit + edge pad to 1024x1024, ImageNet normalize",
        "postprocessing": "sigmoid logits, crop content box, LANCZOS resize to original size",
        "model_input": list(MODEL_INPUT_SIZE),
        "providers": ["CPUExecutionProvider"],
        "runtime": {"onnxruntime": ort.__version__, "platform": platform.platform()},
        "model_source": {
            "matting": "https://huggingface.co/emrikol/birefnet-matting-onnx",
            "dis5k": "https://huggingface.co/onnx-community/BiRefNet-DIS5K-ONNX",
        },
        "model_revision": {"matting": "main; pinned by SHA-256", "dis5k": "main; pinned by SHA-256"},
        "model_license": {"matting": "MIT (repository label)", "dis5k": "MIT (repository label)"},
        "model_sha256": hashes,
        "images": rows,
        "error": None,
    }
    try:
        for name, model_path in MODELS.items():
            run_model(name, model_path, rows)
        record["success"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_root / "timings.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
