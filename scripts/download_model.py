from __future__ import annotations

import hashlib
import sys
import argparse
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/emrikol/birefnet-matting-onnx/resolve/main/birefnet-matting.onnx?download=true"
DEST = Path(__file__).resolve().parents[1] / "models" / "birefnet-matting.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(received: int, total_size: int) -> None:
    if total_size <= 0:
        return
    received = min(received, total_size)
    print(f"\r下载模型: {received / total_size:.1%}", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the pinned ONNX model.")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args(argv)

    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"模型已存在: {destination}")
        print(f"SHA-256: {sha256(destination)}")
        return 0
    temp = destination.with_suffix(".download")
    try:
        request = urllib.request.Request(args.url, headers={"User-Agent": "aigc-cutout/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response, temp.open("wb") as handle:
            total_size = int(response.headers.get("Content-Length", "0"))
            received = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                received += len(chunk)
                progress(received, total_size)
        print()
        if total_size and received != total_size:
            raise IOError(f"incomplete download: {received} of {total_size} bytes")
        temp.replace(destination)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    print(f"已保存: {destination}")
    print(f"SHA-256: {sha256(destination)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
