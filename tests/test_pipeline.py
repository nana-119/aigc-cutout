import sys
import unittest
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutout_tool.infer import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    MATTING_MODEL_SOURCE,
    MODEL_SOURCE,
    alpha_from_prediction,
    composite,
    model_source_for_path,
    parse_args,
    prepare_input,
    prepare_model_image,
    run,
)


class PipelineTest(unittest.TestCase):
    def test_default_model_and_optional_matting_source(self):
        self.assertEqual(parse_args(["--input", "input.png", "--output-dir", "out"]).model, DEFAULT_MODEL_PATH)
        self.assertEqual(DEFAULT_MODEL_PATH, Path("models/birefnet-dis5k.onnx"))
        self.assertEqual(model_source_for_path(DEFAULT_MODEL_PATH), MODEL_SOURCE)
        matting = Path("models/birefnet-matting.onnx")
        self.assertEqual(parse_args(["--input", "input.png", "--output-dir", "out", "--model", str(matting)]).model, matting)
        self.assertEqual(model_source_for_path(matting), MATTING_MODEL_SOURCE)
        self.assertEqual(model_source_for_path(Path("models/custom.onnx")), "unknown")

    def test_prepare_input_shape_and_alpha_outputs(self):
        image = Image.new("RGB", (80, 40), (120, 80, 40))
        tensor = prepare_input(image, size=(32, 32))
        self.assertEqual(tensor.shape, (1, 3, 32, 32))

        prepared, content_box = prepare_model_image(image, size=(32, 32))
        self.assertEqual(prepared.size, (32, 32))
        self.assertEqual(content_box, (0, 8, 32, 24))

        alpha = alpha_from_prediction(__import__("numpy").zeros((1, 1, 32, 32), dtype="float32"), image.size)
        self.assertEqual(alpha.size, image.size)
        self.assertEqual(alpha.getextrema(), (127, 127))

        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        preview = composite(rgba, (255, 255, 255))
        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, image.size)

    def test_existing_transparency_is_preserved(self):
        image = Image.new("RGBA", (16, 16), (255, 255, 255, 0))
        image.putpixel((8, 8), (240, 180, 20, 255))
        alpha = alpha_from_prediction(__import__("numpy").zeros((1, 1, 32, 32), dtype="float32"), image.size)
        combined = __import__("numpy").minimum(__import__("numpy").asarray(alpha), __import__("numpy").asarray(image.getchannel("A")))
        self.assertEqual(int(combined[0, 0]), 0)
        self.assertEqual(int(combined[8, 8]), 127)

    def test_alpha_restore_crops_padding_before_resize(self):
        import numpy as np

        prediction = np.full((1, 1, 32, 32), -80.0, dtype="float32")
        prediction[:, :, 8:24, :] = 80.0
        alpha = alpha_from_prediction(prediction, (80, 40), (0, 8, 32, 24))
        self.assertEqual(alpha.size, (80, 40))
        self.assertEqual(alpha.getpixel((40, 20)), 255)
        self.assertEqual(alpha.getpixel((40, 0)), 255)

    def test_palette_png_transparency_still_uses_black_composite(self):
        import tempfile
        from types import SimpleNamespace

        import numpy as np

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="image")]

            def get_outputs(self):
                return [SimpleNamespace(name="mask")]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, names, inputs):
                self.tensor = inputs["image"]
                return [np.zeros((1, 1, 1024, 1024), dtype=np.float32)]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new("P", (2, 1))
            image.putpalette([255, 255, 255, 255, 0, 0] + [0] * 762)
            image.putdata([0, 1])
            image.info["transparency"] = 0
            source = root / "palette.png"
            image.save(source)
            model = root / "model.onnx"
            model.write_bytes(b"test")
            session = Session()
            run(source, root / "out", model, session=session)
            self.assertTrue(np.all(session.tensor[0, :, 0, 0] < -1.0))
            with Image.open(root / "out" / "palette_transparent.png") as result:
                self.assertEqual(result.getchannel("A").getpixel((0, 0)), 0)


if __name__ == "__main__":
    unittest.main()
