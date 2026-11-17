"""Fast persistence tests; no Slicer, model weights or GPU required."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "nnunet_model_registry", Path(__file__).parents[1] / "SlicerNNUNetLib/ModelRegistry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ModelRegistry = module.ModelRegistry


class FileSettings:
    def __init__(self, path):
        self.path = path
        self.values = json.loads(path.read_text()) if path.exists() else {}

    def value(self, key, default=""):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.path.write_text(json.dumps(self.values))


class Parameters:
    def __init__(self, **values):
        self.values = values

    def toSettings(self, settings, key):
        settings.setValue(key, json.dumps(self.values))
        settings.sync()

    @classmethod
    def fromSettings(cls, settings, key):
        return cls(**json.loads(settings.value(key)))


class ModelRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporaryDirectory = tempfile.TemporaryDirectory(prefix="nnunet registry ")
        self.path = Path(self.temporaryDirectory.name) / "settings.json"
        self.settings = FileSettings(self.path)
        self.registry = ModelRegistry(self.settings, Parameters)

    def tearDown(self):
        self.temporaryDirectory.cleanup()

    def test_empty_slots_do_not_invent_a_model(self):
        self.assertEqual(self.registry.selected, "Manual")
        self.assertIsNone(self.registry.load("CNN"))
        self.assertIsNone(self.registry.load("ViT"))

    def test_models_and_manual_restore_independently_after_restart(self):
        cnn = {"modelPath": "/weights with spaces/CNN", "folds": "all", "checkPointName": "best.pth", "disableTta": False}
        vit = {"modelPath": "/weights with spaces/ViT", "folds": "0,1", "checkPointName": "final.safetensors", "disableTta": True}
        self.registry.save("CNN", Parameters(**cnn))
        self.registry.save("ViT", Parameters(**vit))
        self.registry.save("Manual", Parameters(modelPath="/manual"))
        self.registry.select("CNN")
        restarted = ModelRegistry(FileSettings(self.path), Parameters)
        self.assertEqual(restarted.selected, "CNN")
        self.assertEqual(restarted.load("CNN").values, cnn)
        self.assertEqual(restarted.load("ViT").values, vit)
        self.assertEqual(restarted.load("Manual").values, {"modelPath": "/manual"})
        # Editing a restored object and selecting a slot never overwrites saved settings.
        restarted.load("CNN").values["folds"] = "4"
        restarted.select("ViT")
        self.assertEqual(restarted.load("CNN").values, cnn)

    def test_update_moved_weights_is_explicit(self):
        self.registry.save("CNN", Parameters(modelPath="/old weights"))
        self.registry.select("CNN")
        self.registry.select("ViT")
        self.assertEqual(self.registry.load("CNN").values["modelPath"], "/old weights")
        self.registry.save("CNN", Parameters(modelPath="/new weights"))
        self.assertEqual(ModelRegistry(FileSettings(self.path), Parameters).load("CNN").values["modelPath"], "/new weights")

    def test_malformed_index_and_entry_fail_safely(self):
        self.settings.setValue(ModelRegistry.settingsKey, "not json")
        self.settings.setValue(ModelRegistry.settingsKey + "/CNN", "[]")
        registry = ModelRegistry(self.settings, Parameters)
        self.assertEqual(registry.selected, "Manual")
        self.assertIsNone(registry.load("CNN"))
        self.assertEqual(len(registry.warnings), 2)

    def test_unknown_slot_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.select("Other")
        with self.assertRaises(ValueError):
            self.registry.save("Other", Parameters())


if __name__ == "__main__":
    unittest.main()
