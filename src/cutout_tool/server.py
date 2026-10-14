from __future__ import annotations

import argparse
import json
import mimetypes
import os
import secrets
import shutil
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from .infer import DEFAULT_MODEL_PATH, run


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_RETRIES = 1
ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
FILE_KINDS = {
    "original": ("input", "attachment"),
    "transparent": ("transparent", "attachment"),
    "black": ("preview_black", "inline"),
    "white": ("preview_white", "inline"),
    "alpha": ("alpha", "inline"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def safe_error(exc: Exception, *private_paths: Path) -> str:
    message = str(exc) or "未提供更多信息"
    for path in private_paths:
        message = message.replace(str(path), "[内部路径]")
    return f"{type(exc).__name__}: {message[:500]}"


def safe_display_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.replace("\x00", "").strip()
    return (name or "未命名图片")[:200]


def parse_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("上传必须使用 multipart/form-data")
    message = BytesParser(policy=policy.default).parsebytes(
        (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n").encode() + body
    )
    for part in message.iter_attachments():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        payload = part.get_payload(decode=True)
        if payload:
            return safe_display_name(part.get_filename() or "未命名图片"), payload
    raise ValueError("缺少名为 file 的图片字段")


def validate_image(path: Path) -> tuple[str, tuple[int, int]]:
    try:
        with Image.open(path) as image:
            image_format = image.format or ""
            if image_format not in ALLOWED_FORMATS:
                raise ValueError("仅支持 PNG、JPEG、WEBP 图片")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f"图片像素超过上限 {MAX_PIXELS:,}")
            image.verify()
        with Image.open(path) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            size = oriented.size
    except UnidentifiedImageError as exc:
        raise ValueError("无法识别图片内容") from exc
    return image_format, size


class DevIdentityProvider:
    """Explicit local-only identity; never trusts a browser-supplied user id."""

    def __init__(self, token: str, owner_id: str):
        if not token or not owner_id:
            raise ValueError("开发预览必须显式设置 token 和 owner")
        self.token = token
        self.owner_id = owner_id

    def authenticate(self, provided_token: str | None) -> str | None:
        if provided_token and secrets.compare_digest(provided_token, self.token):
            return self.owner_id
        return None


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    retries INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 1,
                    result_json TEXT,
                    width INTEGER,
                    height INTEGER
                )"""
            )

    def recover_stale(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE tasks
                   SET status='failed', stage='failed',
                       error='服务重启时任务未完成，请重新上传或重试', updated_at=?
                 WHERE status IN ('queued', 'processing')""",
                (utc_now(),),
            )

    def create(self, task_id: str, owner_id: str, name: str, input_path: Path, output_dir: Path, size: tuple[int, int]) -> dict:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO tasks
                   (id, owner_id, original_name, input_path, output_dir, status, stage,
                    created_at, updated_at, max_retries, width, height)
                   VALUES (?, ?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?, ?)""",
                (task_id, owner_id, name, str(input_path), str(output_dir), now, now, MAX_RETRIES, size[0], size[1]),
            )
        return self.get(task_id)

    def get(self, task_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def get_owned(self, task_id: str, owner_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id=? AND owner_id=?", (task_id, owner_id)).fetchone()
        return dict(row) if row else None

    def list_owned(self, owner_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE owner_id=? ORDER BY created_at DESC LIMIT 100", (owner_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def update(self, task_id: str, **fields: object) -> None:
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        values = list(fields.values()) + [task_id]
        with self._connect() as connection:
            connection.execute(f"UPDATE tasks SET {assignments} WHERE id=?", values)

    def retry(self, task_id: str, owner_id: str) -> dict | None:
        task = self.get_owned(task_id, owner_id)
        if not task or task["status"] != "failed" or task["retries"] >= task["max_retries"]:
            return None
        self.update(task_id, status="queued", stage="queued", error=None, retries=task["retries"] + 1)
        return self.get(task_id)


class TaskManager:
    def __init__(self, store: TaskStore, model_path: Path, data_dir: Path):
        self.store = store
        self.model_path = model_path
        self.data_dir = data_dir
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutout")
        self._session = None
        self._session_lock = threading.Lock()

    def submit(self, task_id: str) -> None:
        self.executor.submit(self._process, task_id)

    def _get_session(self):
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    import onnxruntime as ort

                    self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session

    def _process(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if not task:
            return
        try:
            self.store.update(task_id, status="processing", stage="loading_model", error=None)
            session = self._get_session()
            self.store.update(task_id, stage="inference")
            result = run(Path(task["input_path"]), Path(task["output_dir"]), self.model_path, session=session)
            payload = {
                "original_size": result.original_size,
                "output_size": result.output_size,
                "elapsed_ms": result.elapsed_ms,
                "providers": result.providers,
                "model_sha256": result.model_sha256,
                "outputs": {key: Path(value).name for key, value in result.output_files.items()},
            }
            self.store.update(task_id, status="succeeded", stage="completed", result_json=json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - persist a user-safe failure state.
            self.store.update(task_id, status="failed", stage="failed", error=safe_error(exc, self.model_path, self.data_dir))


def public_task(task: dict) -> dict:
    result = json.loads(task["result_json"]) if task.get("result_json") else None
    return {
        "task_id": task["id"],
        "status": task["status"],
        "stage": task["stage"],
        "original_name": task["original_name"],
        "width": task["width"],
        "height": task["height"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "retries": task["retries"],
        "max_retries": task["max_retries"],
        "error": task["error"],
        "result": result,
    }


class CutoutHandler(BaseHTTPRequestHandler):
    server_version = "AIGCCutoutDev/0.1"

    @property
    def app(self):
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        print(f"[cutout] {self.address_string()} {format % args}")

    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: object) -> None:
        self._send(status, json_bytes(value), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _owner(self) -> str | None:
        return self.app.identity.authenticate(self.headers.get("X-Cutout-Dev-Token"))

    def _require_owner(self) -> str | None:
        owner = self._owner()
        if owner is None:
            self._error(401, "开发预览 token 无效或缺失")
        return owner

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            return self._serve_asset("index.html", "text/html; charset=utf-8")
        if path in {"/app.js", "/styles.css"}:
            content_type = "application/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8"
            return self._serve_asset(path.lstrip("/"), content_type)
        owner = self._require_owner()
        if owner is None:
            return
        if path == "/api/health":
            return self._json(200, {"ok": True, "model_exists": self.app.model_path.exists(), "model": str(self.app.model_path.name)})
        if path == "/api/tasks":
            return self._json(200, {"tasks": [public_task(task) for task in self.app.store.list_owned(owner)]})
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "tasks" and parts[2]:
            task = self.app.store.get_owned(parts[2], owner)
            return self._json(200, public_task(task)) if task else self._error(404, "任务不存在")
        if len(parts) == 5 and parts[:2] == ["api", "tasks"] and parts[3] == "files":
            task = self.app.store.get_owned(parts[2], owner)
            if not task or parts[4] not in FILE_KINDS:
                return self._error(404, "文件不存在")
            kind, disposition = FILE_KINDS[parts[4]]
            result = json.loads(task["result_json"]) if task.get("result_json") else {}
            if kind == "input":
                file_path = Path(task["input_path"])
            else:
                filename = result.get("outputs", {}).get(kind)
                file_path = Path(task["output_dir"]) / filename if filename else Path()
            if not file_path.is_file():
                return self._error(404, "文件尚未生成")
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            headers = {"Content-Disposition": f'{disposition}; filename="{file_path.name}"'}
            return self._send(200, file_path.read_bytes(), content_type, headers)
        self._error(404, "页面不存在")

    def _serve_asset(self, name: str, content_type: str) -> None:
        path = self.app.web_dir / name
        if not path.is_file():
            return self._error(404, "预览页面文件不存在")
        self._send(200, path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        owner = self._require_owner()
        if owner is None:
            return
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/tasks":
            return self._create_task(owner)
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "retry":
            task = self.app.store.retry(parts[2], owner)
            if not task:
                return self._error(409, "任务不可重试或已达到重试上限")
            self.app.manager.submit(task["id"])
            return self._json(202, public_task(task))
        self._error(404, "接口不存在")

    def _create_task(self, owner: str) -> None:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "-1")
        except ValueError:
            length = -1
        if length < 0 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
            return self._error(413, "上传大小超过限制或缺少 Content-Length")
        body = self.rfile.read(length)
        try:
            original_name, payload = parse_upload(self.headers.get("Content-Type", ""), body)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError(f"文件大小超过上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            task_id = uuid.uuid4().hex
            task_dir = self.app.data_dir / "tasks" / task_id
            task_dir.mkdir(parents=True, exist_ok=False)
            upload_path = task_dir / "upload.bin"
            upload_path.write_bytes(payload)
            image_format, size = validate_image(upload_path)
            input_path = task_dir / f"input{ALLOWED_FORMATS[image_format]}"
            upload_path.replace(input_path)
            output_dir = task_dir / "outputs"
            task = self.app.store.create(task_id, owner, original_name, input_path, output_dir, size)
            self.app.manager.submit(task_id)
        except Exception as exc:  # noqa: BLE001 - convert validation errors to a clear API response.
            if "task_dir" in locals() and task_dir.exists():
                shutil.rmtree(task_dir)
            return self._error(400, str(exc))
        self._json(202, public_task(task))


class CutoutApp:
    def __init__(self, root: Path, data_dir: Path, model_path: Path, identity: DevIdentityProvider):
        self.root = root
        self.data_dir = data_dir
        self.model_path = model_path
        self.identity = identity
        self.web_dir = root / "src" / "cutout_tool" / "web"
        self.store = TaskStore(data_dir / "cutout.db")
        self.store.recover_stale()
        self.manager = TaskManager(self.store, model_path, data_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Start the local cutout development preview server.")
    parser.add_argument("--host", default="127.0.0.1", help="development-only bind address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dev-token", default=os.environ.get("CUTOUT_DEV_TOKEN"))
    parser.add_argument("--dev-owner", default=os.environ.get("CUTOUT_DEV_OWNER"))
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--model", type=Path, default=root / DEFAULT_MODEL_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dev_token or not args.dev_owner:
        raise SystemExit("必须显式提供 --dev-token/CUTOUT_DEV_TOKEN 和 --dev-owner/CUTOUT_DEV_OWNER；不会使用默认测试身份")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"模型不存在: {model_path}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("开发预览只允许绑定本机回环地址")
    root = Path(__file__).resolve().parents[2]
    app = CutoutApp(root, args.data_dir.resolve(), model_path, DevIdentityProvider(args.dev_token, args.dev_owner))
    server = ThreadingHTTPServer((args.host, args.port), CutoutHandler)
    server.app = app  # type: ignore[attr-defined]
    print(f"开发预览: http://{args.host}:{args.port}/")
    print(f"数据目录: {app.data_dir}")
    print(f"模型: {model_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n开发预览已停止")
    finally:
        server.shutdown()
        app.manager.executor.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
