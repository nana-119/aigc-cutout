import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutout_tool.server import (  # noqa: E402
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
            task = store.create("task-1", "owner-a", "x.png", image_path, root / "outputs", size)
            self.assertEqual(store.get_owned(task["id"], "owner-a")["id"], "task-1")
            self.assertIsNone(store.get_owned(task["id"], "owner-b"))
            store.recover_stale()
            self.assertEqual(store.get(task["id"])["status"], "failed")
            retry = store.retry(task["id"], "owner-a")
            self.assertEqual(retry["status"], "queued")
            self.assertEqual(retry["retries"], 1)
            self.assertIsNone(store.retry(task["id"], "owner-a"))


if __name__ == "__main__":
    unittest.main()
