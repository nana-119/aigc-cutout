import http.client
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutout_tool.server import (  # noqa: E402
    MAX_INFLIGHT_PER_OWNER,
    CutoutApp,
    CutoutHandler,
    DevIdentityProvider,
    TaskStore,
    parse_args,
    parse_upload,
    validate_image,
)


class ServerTest(unittest.TestCase):
    def test_default_model_and_optional_matting(self):
        self.assertEqual(parse_args([]).model.name, "birefnet-dis5k.onnx")
        matting = Path("models/birefnet-matting.onnx")
        self.assertEqual(parse_args(["--model", str(matting)]).model, matting)

    def test_explicit_dev_identity_and_multipart_upload(self):
        identity = DevIdentityProvider("secret", "local-owner")
        self.assertIsNone(identity.authenticate("wrong"))
        self.assertEqual(identity.authenticate("secret"), "local-owner")
        boundary = "cutout-test"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n"
            "Content-Type: image/png\r\n\r\nhello\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        name, payload = parse_upload(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(name, "x.png")
        self.assertEqual(payload, b"hello")

    def test_image_validation_and_owner_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "input.png"
            Image.new("RGBA", (12, 8), (1, 2, 3, 0)).save(image_path)
            image_format, size = validate_image(image_path)
            self.assertEqual((image_format, size), ("PNG", (12, 8)))

            store = TaskStore(root / "cutout.db")
            task = store.create("task-1", "owner-a", "x.png", image_path, root / "outputs", size, "hash-test")
            self.assertEqual(store.get_owned(task["id"], "owner-a")["id"], "task-1")
            self.assertIsNone(store.get_owned(task["id"], "owner-b"))
            store.recover_stale()
            self.assertEqual(store.get(task["id"])["status"], "failed")
            retry = store.retry(task["id"], "owner-a")
            self.assertEqual(retry["status"], "queued")
            self.assertEqual(retry["retries"], 1)
            self.assertIsNone(store.retry(task["id"], "owner-a"))


class StoreIdempotencyTest(unittest.TestCase):
    def test_legacy_db_gets_hash_column_via_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / "legacy.db"
            connection = sqlite3.connect(legacy)
            try:
                connection.execute(
                    """CREATE TABLE tasks (
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
                connection.commit()
            finally:
                connection.close()
            store = TaskStore(legacy)
            connection = sqlite3.connect(legacy)
            try:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(tasks)")]
            finally:
                connection.close()
            self.assertIn("image_sha256", columns)
            # legacy rows keep NULL hash and are not idempotency candidates
            self.assertIsNone(store.find_duplicate("owner-a", "anything"))

    def test_find_duplicate_scope_and_inflight_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "in.png"
            Image.new("RGBA", (12, 8), (1, 2, 3, 255)).save(image_path)
            store = TaskStore(root / "cutout.db")
            store.create("t1", "owner-a", "x.png", image_path, root / "o1", (12, 8), "hash-abc")
            store.create("t2", "owner-a", "y.png", image_path, root / "o2", (12, 8), "hash-other")
            store.create("t3", "owner-b", "z.png", image_path, root / "o3", (12, 8), "hash-abc")
            self.assertEqual(store.find_duplicate("owner-a", "hash-abc")["id"], "t1")
            self.assertEqual(store.find_duplicate("owner-a", "hash-other")["id"], "t2")
            self.assertIsNone(store.find_duplicate("owner-a", "missing"))
            self.assertEqual(store.find_duplicate("owner-b", "hash-abc")["id"], "t3")
            self.assertEqual(store.count_inflight("owner-a"), 2)  # both queued
            store.update("t1", status="succeeded", stage="completed")
            self.assertEqual(store.count_inflight("owner-a"), 1)
            store.update("t2", status="failed", stage="failed", error="boom")
            self.assertEqual(store.count_inflight("owner-a"), 0)


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_image")]

    def get_outputs(self):
        return [SimpleNamespace(name="output_image")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, inputs):
        return [np.zeros((1, 1, 1024, 1024), dtype="float32")]


class MultiTokenIdentity(DevIdentityProvider):
    def __init__(self, tokens):
        self.tokens = tokens

    def authenticate(self, provided_token):
        return self.tokens.get(provided_token)


def png_bytes(seed: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (16, 16), (seed % 255, (seed * 7) % 255, (seed * 13) % 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class HttpPreviewServer:
    def __init__(self, root: Path, data_dir: Path, model_path: Path, identity):
        self.app = CutoutApp(root, data_dir, model_path, identity)
        self.app.manager._get_session = lambda: FakeSession()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), CutoutHandler)
        self.httpd.app = self.app
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.manager.executor.shutdown(wait=False, cancel_futures=True)


def upload(server: HttpPreviewServer, token: str, payload: bytes, filename: str = "x.png") -> tuple[int, dict]:
    boundary = "cutout-test-boundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    connection.request(
        "POST", "/api/tasks", body=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Cutout-Dev-Token": token,
            "Content-Length": str(len(body)),
        },
    )
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, data


def get_json(server: HttpPreviewServer, token: str, path: str) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    connection.request("GET", path, headers={"X-Cutout-Dev-Token": token})
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, data


def post_retry(server: HttpPreviewServer, token: str, task_id: str) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    connection.request("POST", f"/api/tasks/{task_id}/retry", headers={"X-Cutout-Dev-Token": token})
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, data


def wait_succeeded(server: HttpPreviewServer, token: str, task_id: str) -> dict:
    for _ in range(100):
        status, task = get_json(server, token, f"/api/tasks/{task_id}")
        if task["status"] in ("succeeded", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not settle")


class HttpIdempotencyAndIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        model_path = self.root / "model.onnx"
        model_path.write_bytes(b"test")
        repo_root = Path(__file__).resolve().parents[1]
        identity = MultiTokenIdentity({"token-a": "owner-a", "token-b": "owner-b"})
        self.server = HttpPreviewServer(repo_root, self.data_dir, model_path, identity)

    def tearDown(self):
        self.server.close()
        self.temp.cleanup()

    def test_duplicate_upload_reuses_task_without_new_inference(self):
        payload = png_bytes(1)
        status, first = upload(self.server, "token-a", payload, "a.png")
        self.assertEqual(status, 202)
        self.assertFalse(first.get("reused", False))
        settled = wait_succeeded(self.server, "token-a", first["task_id"])
        self.assertEqual(settled["status"], "succeeded")

        status, second = upload(self.server, "token-a", payload, "renamed.png")
        self.assertEqual(status, 200)
        self.assertTrue(second["reused"])
        self.assertEqual(second["task_id"], first["task_id"])
        self.assertEqual(second["status"], "succeeded")

        different = png_bytes(2)
        status, third = upload(self.server, "token-a", different, "b.png")
        self.assertEqual(status, 202)
        self.assertNotEqual(third["task_id"], first["task_id"])

    def test_inflight_quota_rejects_new_task_per_owner(self):
        payload = png_bytes(10)
        status, task = upload(self.server, "token-a", payload)
        self.assertEqual(status, 202)
        wait_succeeded(self.server, "token-a", task["task_id"])

        self.server.app.store.update(task["task_id"], status="processing", stage="processing")
        status, body = upload(self.server, "token-a", png_bytes(11))
        self.assertEqual(status, 409)
        self.assertIn("进行中", body["error"])
        # other owners are not blocked
        status, _ = upload(self.server, "token-b", png_bytes(12))
        self.assertEqual(status, 202)

    def test_cross_owner_read_and_retry_are_rejected(self):
        payload = png_bytes(20)
        status, task = upload(self.server, "token-a", payload)
        self.assertEqual(status, 202)
        wait_succeeded(self.server, "token-a", task["task_id"])

        status, body = get_json(self.server, "token-b", f"/api/tasks/{task['task_id']}")
        self.assertEqual(status, 404)
        status, _ = get_json(self.server, "token-b", f"/api/tasks/{task['task_id']}/files/transparent")
        self.assertEqual(status, 404)
        status, _ = post_retry(self.server, "token-b", task["task_id"])
        self.assertEqual(status, 409)


if __name__ == "__main__":
    unittest.main()
