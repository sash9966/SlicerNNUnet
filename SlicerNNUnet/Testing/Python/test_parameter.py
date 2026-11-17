"""Fast metadata/argument tests. Run with Python; neither Slicer nor model weights are required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def load_parameter():
    """Supply only import-time Slicer/Qt types; production validation and argument code remain real."""
    qt = types.ModuleType("qt")
    qt.QSettings = object
    wrapper = types.ModuleType("slicer.parameterNodeWrapper")

    def parameter_pack(cls):
        def set_value(instance, name, value):
            if name not in cls.__dataclass_fields__:
                raise ValueError(name)
            setattr(instance, name, value)
        cls.setValue = set_value
        return cls

    wrapper.parameterPack = parameter_pack
    wrapper.Choice = lambda *args: object()
    wrapper.WithinRange = lambda *args: object()
    source = Path(__file__).parents[2] / "SlicerNNUNetLib" / "Parameter.py"
    spec = importlib.util.spec_from_file_location("parameter_under_test", source)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module while decorating the class.
    with patch.dict(sys.modules, {"qt": qt, "slicer.parameterNodeWrapper": wrapper,
                                 spec.name: module}):
        spec.loader.exec_module(module)
    return module.Parameter


Parameter = load_parameter()


class Settings:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


class ParameterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model settings with spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.create_model()
        self.parameter = Parameter(modelPath=self.model, device="cpu")

    def create_model(self, trainer="nnUNetTrainer", plans="nnUNetPlans", configuration="3d_fullres"):
        path = self.root / "Dataset123_Example" / f"{trainer}__{plans}__{configuration}"
        path.mkdir(parents=True)
        (path / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "Left ventricle": 1, "Aorta": 7},
            "file_ending": ".nii.gz",
        }))
        (path / "plans.json").write_text(json.dumps({
            "plans_name": plans,
            "dataset_name": "Dataset123_Example",
            "configurations": {configuration: {}},
        }))
        for fold in (0, 1, 2, 3, 4, "all"):
            (path / f"fold_{fold}").mkdir()
            (path / f"fold_{fold}" / "checkpoint_final.pth").touch()
        return path

    def edit_dataset(self, **fields):
        path = self.model / "dataset.json"
        dataset = json.loads(path.read_text())
        dataset.update(fields)
        path.write_text(json.dumps(dataset))

    def test_standard_configuration_and_default_fold(self):
        self.assertEqual(self.parameter.isValid(), (True, ""))
        self.assertEqual(self.parameter._foldsAsList(), [0])
        args = self.parameter.asArgList(self.root / "input", self.root / "output")
        self.assertEqual(args[args.index("-d") + 1], "Dataset123_Example")
        self.assertEqual(args[args.index("-tr") + 1], "nnUNetTrainer")
        self.assertEqual(args[args.index("-p") + 1], "nnUNetPlans")
        self.assertEqual(args[args.index("-c") + 1], "3d_fullres")

    def test_explicit_fold_selections(self):
        for selection, expected in (("0", [0]), (" 0, 1,2,3,4 ", [0, 1, 2, 3, 4]), ("all", ["all"])):
            with self.subTest(selection=selection):
                self.parameter.folds = selection
                self.assertEqual(self.parameter._foldsAsList(), expected)
                self.assertTrue(self.parameter.isValid()[0])
                args = self.parameter.asArgList(self.root / "input", self.root / "output")
                start = args.index("-f") + 1
                self.assertEqual(args[start:args.index("-npp")], [str(fold) for fold in expected])

    def test_invalid_folds_are_rejected_without_throwing_from_validation(self):
        for selection in ("all,0", "0,all", "All", "-1", "1.0", "0,,1", ",0", "0,", "0 1", "0,0", "٠"):
            with self.subTest(selection=selection):
                self.parameter.folds = selection
                valid, reason = self.parameter.isValid()
                self.assertFalse(valid)
                self.assertIn("fold", reason.lower())

    def test_missing_all_does_not_use_numbered_fold(self):
        self.parameter.folds = "all"
        (self.model / "fold_all" / "checkpoint_final.pth").unlink()
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("all", self.parameter.isValid()[1])

    def test_missing_fold_does_not_fallback(self):
        self.parameter.folds = "5"
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("5", self.parameter.isValid()[1])

    def test_every_selected_fold_requires_the_exact_checkpoint(self):
        self.parameter.folds = "0,1"
        self.parameter.checkPointName = "checkpoint_best.pth"
        (self.model / "fold_0" / "checkpoint_best.pth").touch()
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("checkpoint_best.pth", self.parameter.isValid()[1])
        (self.model / "fold_1" / "checkpoint_best.pth").touch()
        self.assertTrue(self.parameter.isValid()[0])

    def test_safetensors_is_validated_exactly(self):
        self.parameter.checkPointName = "selected.safetensors"
        self.assertFalse(self.parameter.isValid()[0])
        (self.model / "fold_0" / self.parameter.checkPointName).touch()
        self.assertTrue(self.parameter.isValid()[0])

    def test_checkpoint_directory_does_not_count_as_file(self):
        self.parameter.checkPointName = "folder.pth"
        (self.model / "fold_0" / self.parameter.checkPointName).mkdir()
        self.assertFalse(self.parameter.isValid()[0])

    def test_unsafe_or_unsupported_checkpoint_names(self):
        for filename in ("../checkpoint_final.pth", "nested/checkpoint_final.pth", "..\\checkpoint_final.pth", "/checkpoint_final.pth", "weights.txt"):
            with self.subTest(filename=filename):
                self.parameter.checkPointName = filename
                self.assertFalse(self.parameter.isValid()[0])
                self.assertIn("filename", self.parameter.isValid()[1])

    def test_full_identifiers_with_double_underscores_are_preserved(self):
        model = self.create_model(trainer="Original__Trainer", plans="ptPlans__cnn____Spacing__1.00", configuration="3d_fullres")
        self.parameter.modelPath = model
        self.assertTrue(self.parameter.isValid()[0])
        self.assertEqual(self.parameter._configurationNameParts,
                         ["Original__Trainer", "ptPlans__cnn____Spacing__1.00", "3d_fullres"])
        args = self.parameter.asArgList(self.root / "input", self.root / "output")
        self.assertEqual(args[args.index("-tr") + 1], "Original__Trainer")
        self.assertEqual(args[args.index("-p") + 1], "ptPlans__cnn____Spacing__1.00")

    def test_wrong_plans_or_configuration_does_not_reconstruct_identifiers(self):
        plans = self.model / "plans.json"
        data = json.loads(plans.read_text())
        data["plans_name"] = "AnotherPlans"
        plans.write_text(json.dumps(data))
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("unchanged", self.parameter.isValid()[1])

    def test_inconsistent_dataset_identity_is_rejected(self):
        plans = self.model / "plans.json"
        data = json.loads(plans.read_text())
        data["dataset_name"] = "Dataset999_Other"
        plans.write_text(json.dumps(data))
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("dataset_name", self.parameter.isValid()[1])

    def test_specific_fold_path_resolves_to_same_model_parent(self):
        self.parameter.modelPath = self.model / "fold_all"
        self.parameter.folds = "all"
        self.assertEqual(self.parameter.configurationFolder, self.model)
        self.assertTrue(self.parameter.isValid()[0])

    def test_unambiguous_parent_resolves(self):
        self.parameter.modelPath = self.model.parent
        self.assertTrue(self.parameter.isValid()[0])
        self.assertEqual(self.parameter.configurationFolder, self.model)

    def test_multiple_models_require_explicit_selection(self):
        self.create_model(trainer="DifferentTrainer")
        self.parameter.modelPath = self.model.parent
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("Multiple model", self.parameter.isValid()[1])
        self.assertEqual(self.parameter.readChannelNamesFromFile(), [])

    def test_missing_metadata(self):
        for name in ("dataset.json", "plans.json"):
            with self.subTest(name=name):
                path = self.model / name
                original = path.read_bytes()
                path.unlink()
                self.assertFalse(self.parameter.isValid()[0])
                self.assertIn(name, self.parameter.isValid()[1])
                path.write_bytes(original)

    def test_malformed_metadata_is_reported(self):
        for name in ("dataset.json", "plans.json"):
            with self.subTest(name=name):
                path = self.model / name
                original = path.read_bytes()
                path.write_text("{broken")
                self.assertFalse(self.parameter.isValid()[0])
                self.assertIn(name, self.parameter.isValid()[1])
                path.write_bytes(original)

    def test_channel_order_is_numeric(self):
        channels = {str(index): f"modality {index}" for index in reversed(range(12))}
        self.edit_dataset(channel_names=channels)
        self.assertTrue(self.parameter.isValid()[0])
        self.assertEqual(self.parameter.readChannelNamesFromFile(), [f"modality {index}" for index in range(12)])

    def test_missing_channel_definition_is_rejected(self):
        self.edit_dataset(channel_names={"1": "CT"})
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("channel", self.parameter.isValid()[1])

    def test_label_names_and_ids_exclude_background_and_ignore(self):
        self.edit_dataset(labels={"background": 0, "Left ventricle": 1, "Aorta": 7, "ignore": 255})
        self.assertEqual(self.parameter.readSegmentIdsAndLabelsFromDatasetFile(),
                         [("Segment_1", "Left ventricle"), ("Segment_7", "Aorta")])

    def test_region_labels_use_saved_output_class_order(self):
        self.edit_dataset(labels={"background": 0, "Whole": [1, 2], "Core": 2}, regions_class_order=[1, 2])
        self.assertTrue(self.parameter.isValid()[0])
        self.assertEqual(self.parameter.readSegmentIdsAndLabelsFromDatasetFile(), [("Segment_1", "Whole"), ("Segment_2", "Core")])

    def test_invalid_labels_are_reported(self):
        self.edit_dataset(labels={"background": 0, "Heart": "1"})
        self.assertFalse(self.parameter.isValid()[0])
        self.assertIn("label", self.parameter.isValid()[1])

    def test_paths_with_spaces_are_single_arguments(self):
        input_path, output_path = self.root / "input with spaces", self.root / "output with spaces"
        args = self.parameter.asArgList(input_path, output_path)
        self.assertEqual(args[args.index("-i") + 1], str(input_path))
        self.assertEqual(args[args.index("-o") + 1], str(output_path))
        self.assertIn(" ", str(self.parameter.modelFolder))

    def test_selected_device_is_not_silently_changed(self):
        self.parameter.device = "cuda"
        with patch.object(self.parameter, "isSelectedDeviceAvailable", return_value=False):
            args = self.parameter.asArgList(self.root / "input", self.root / "output")
        self.assertEqual(args[args.index("-device") + 1], "cuda")

    def test_mac_sequential_workers_are_preserved(self):
        with patch.object(sys, "platform", "darwin"):
            args = self.parameter.asArgList(self.root / "input", self.root / "output")
        self.assertEqual(args[args.index("-npp") + 1], 0)
        self.assertEqual(args[args.index("-nps") + 1], 0)

    def test_native_worker_counts_elsewhere(self):
        self.parameter.nProcessPreprocessing = 2
        self.parameter.nProcessSegmentationExport = 3
        with patch.object(sys, "platform", "linux"):
            args = self.parameter.asArgList(self.root / "input", self.root / "output")
        self.assertEqual(args[args.index("-npp") + 1], 2)
        self.assertEqual(args[args.index("-nps") + 1], 3)

    def test_inference_defaults_round_trip_in_settings(self):
        settings = Settings()
        self.parameter.folds = "all"
        self.parameter.checkPointName = "selected.safetensors"
        self.parameter.disableTta = False
        self.parameter.toSettings(settings)
        restored = Parameter.fromSettings(settings)
        self.assertEqual(restored, self.parameter)

    def test_zero_step_size_is_invalid(self):
        self.parameter.stepSize = 0
        self.assertFalse(self.parameter.isValid()[0])


if __name__ == "__main__":
    unittest.main()
