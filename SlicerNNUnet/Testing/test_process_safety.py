"""Fast process/import safety checks. No Slicer, models or GPU are required.

Run with: python3 -m unittest discover -s SlicerNNUnet/Testing -p 'test_*.py'
These fakes check failure boundaries; actual image geometry is tested in Slicer.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


LIB = Path(__file__).resolve().parents[1] / "SlicerNNUNetLib"


class _Signal:
    def __init__(self, *_):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def __call__(self, *args):
        for slot in list(self.slots):
            slot(*args)

    emit = __call__


class _ByteArray:
    def __init__(self, value=b""):
        self.value = value

    def data(self):
        return self.value


class _QProcess:
    MergedChannels = 0
    Unbuffered = 1
    ReadOnly = 2
    NotRunning = 0
    Running = 2
    NormalExit = 0
    CrashExit = 1
    ProcessChannelMode = int

    def __init__(self):
        self.finished = _Signal()
        self.errorOccurred = _Signal()
        self.readyRead = _Signal()
        self.output = b""
        self.stderr = b""
        self.status = self.NotRunning
        self.setProcessChannelMode = Mock()
        self.start = Mock(side_effect=self._start)
        self.waitForFinished = Mock()

    def _start(self, *_):
        self.status = self.Running

    def state(self):
        return self.status

    def kill(self):
        self.status = self.NotRunning
        self.finished(-1, self.CrashExit)

    def readAll(self):
        value, self.output = self.output, b""
        return _ByteArray(value)

    def readAllStandardError(self):
        value, self.stderr = self.stderr, b""
        return _ByteArray(value)

    def errorString(self):
        return "Failed to start prediction executable"


class _TemporaryDir:
    def __init__(self):
        self.directory = tempfile.mkdtemp(prefix="slicer model safety ")

    def path(self):
        return self.directory

    def isValid(self):
        return True

    def __del__(self):
        shutil.rmtree(self.directory, ignore_errors=True)


class _Parameter:
    def __init__(self, **values):
        self.channelNames = values.pop("channelNames", {"0": "CT"})
        self.device = values.pop("device", "cpu")
        self.modelFolder = Path(values.pop("modelFolder", "/ordinary model folder"))
        self.__dict__.update(values)

    def asDict(self):
        return dict(self.__dict__)

    def asArgList(self, inDir, outDir):
        return ["-i", str(inDir), "-o", str(outDir)]

    def readChannelNamesFromFile(self):
        return self.channelNames

    def isSelectedDeviceAvailable(self):
        return True

    def readFileEndingFromDatasetFile(self):
        return ".nii.gz"

    def readSegmentIdsAndLabelsFromDatasetFile(self):
        return [("Segment_1", "LV"), ("Segment_2", "RV")]

    def debugString(self):
        return "{}"


class _InferenceProcess:
    def __init__(self):
        self.finished = _Signal()
        self.errorOccurred = _Signal()
        self.readInfo = _Signal()
        self.start = Mock()
        self.stop = Mock()
        self.waitForFinished = Mock()


def _load_logic():
    name = "_slicer_process_safety_test"
    package = types.ModuleType(name)
    package.__path__ = [str(LIB)]
    parameter = types.ModuleType(name + ".Parameter")
    parameter.Parameter = _Parameter
    slicer = types.ModuleType("slicer")
    slicer.vtkMRMLScalarVolumeNode = type("Volume", (), {"GetTransformNodeID": lambda self: None})
    slicer.util = types.SimpleNamespace(loadSegmentation=Mock(), exportNode=Mock())
    slicer.mrmlScene = types.SimpleNamespace(RemoveNode=Mock())
    qt = types.ModuleType("qt")
    qt.QTemporaryDir = _TemporaryDir
    qt.QProcess = _QProcess
    qt.QIODevice = types.SimpleNamespace(OpenMode=int)
    with patch.dict(sys.modules, {name: package, name + ".Parameter": parameter,
                                  "slicer": slicer, "qt": qt}):
        for filename in ("Signal", "SegmentationLogic"):
            spec = importlib.util.spec_from_file_location(name + "." + filename, LIB / (filename + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return module, slicer


class SegmentationSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.slicer = _load_logic()

    def setUp(self):
        self.slicer.util.loadSegmentation.reset_mock()
        self.slicer.mrmlScene.RemoveNode.reset_mock()
        self.slicer.util.exportNode = Mock(side_effect=lambda node, path, *args: Path(path).write_bytes(b"scan"))
        self.process = _InferenceProcess()
        self.logic = self.module.SegmentationLogic(self.process)
        self.logic.setParameter(_Parameter())
        self.errors = []
        self.done = []
        self.logic.errorOccurred.connect(self.errors.append)
        self.logic.inferenceFinished.connect(lambda: self.done.append(True))
        self.logic._findUNetPredictPath = lambda: Path("/runtime with spaces/nnUNetv2_predict")
        self.volume = self.slicer.vtkMRMLScalarVolumeNode()
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def start(self):
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_called_once()

    def test_failed_run_does_not_import_existing_output(self):
        self.start()
        Path(self.logic._outFile).write_bytes(b"partial")
        self.process.errorOccurred("CUDA out of memory")
        self.process.finished()
        with self.assertRaisesRegex(RuntimeError, "successful|failed|cancelled"):
            self.logic.loadSegmentation()
        self.slicer.util.loadSegmentation.assert_not_called()
        self.assertEqual(self.done, [])

    def test_cancelled_run_does_not_import_output_or_emit_success(self):
        self.start()
        Path(self.logic._outFile).write_bytes(b"partial")
        self.logic.stopSegmentation()
        self.process.finished()
        with self.assertRaises(RuntimeError):
            self.logic.loadSegmentation()
        self.slicer.util.loadSegmentation.assert_not_called()
        self.assertEqual(self.done, [])

    def test_cancel_during_export_progress_does_not_start_process(self):
        # Widget progress reporting processes Qt events, so Stop may run here.
        self.logic.progressInfo.connect(
            lambda message: self.logic.stopSegmentation() if message.startswith("Transferring volume") else None)
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_not_called()
        self.assertFalse(self.logic._running)
        with self.assertRaises(RuntimeError):
            self.logic.loadSegmentation()

    def test_cancel_during_start_progress_does_not_start_process(self):
        self.logic.progressInfo.connect(
            lambda message: self.logic.stopSegmentation() if message.startswith("Starting nnUNet") else None)
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_not_called()
        self.assertFalse(self.logic._running)
        with self.assertRaises(RuntimeError):
            self.logic.loadSegmentation()

    def test_cancel_during_preprocessing_progress_does_not_start_process(self):
        self.logic.progressInfo.connect(
            lambda message: self.logic.stopSegmentation() if message.startswith("nnUNet preprocessing") else None)
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_not_called()
        self.assertFalse(self.logic._running)

    def test_cancel_and_restart_during_progress_does_not_resume_old_run(self):
        # A Qt event batch may contain Stop followed immediately by Apply.
        restarted = []

        def restart_once(message):
            if message.startswith("Transferring volume") and not restarted:
                restarted.append(True)
                self.logic.stopSegmentation()
                self.logic.startSegmentation(self.volume)

        self.logic.progressInfo.connect(restart_once)
        self.logic.startSegmentation(self.volume)
        self.assertEqual(restarted, [True])
        self.process.start.assert_called_once()
        self.assertTrue(self.logic._running)
        self.assertFalse(self.logic._cancelled)

    def test_only_exact_case_output_is_accepted(self):
        self.start()
        nested = self.logic.nnUNetOutDir / "old run"
        nested.mkdir()
        (nested / "volume.nii.gz").write_bytes(b"old")
        (self.logic.nnUNetOutDir / "unrelated.nii.gz").write_bytes(b"other")
        self.process.finished()
        with self.assertRaisesRegex(RuntimeError, "expected"):
            self.logic.loadSegmentation()
        self.slicer.util.loadSegmentation.assert_not_called()

    def test_success_imports_expected_file_and_applies_metadata_and_reference(self):
        self.start()
        Path(self.logic._outFile).write_bytes(b"complete")
        segments = {"Segment_1": Mock(), "Segment_2": Mock()}
        node = Mock()
        node.GetSegmentation.return_value.GetSegment.side_effect = segments.get
        self.slicer.util.loadSegmentation.return_value = node
        self.process.finished()
        self.assertIs(self.logic.loadSegmentation(), node)
        self.slicer.util.loadSegmentation.assert_called_once_with(self.logic._outFile)
        segments["Segment_1"].SetName.assert_called_once_with("LV")
        segments["Segment_2"].SetName.assert_called_once_with("RV")
        node.SetReferenceImageGeometryParameterFromVolumeNode.assert_called_once_with(self.volume)
        node.SetAndObserveTransformNodeID.assert_called_once_with(None)
        self.slicer.mrmlScene.RemoveNode.assert_not_called()

    def test_failed_import_removes_only_new_node(self):
        self.start()
        Path(self.logic._outFile).write_bytes(b"complete")
        node = Mock()
        node.SetReferenceImageGeometryParameterFromVolumeNode.side_effect = RuntimeError("invalid geometry")
        self.slicer.util.loadSegmentation.return_value = node
        self.process.finished()
        with self.assertRaisesRegex(RuntimeError, "invalid geometry"):
            self.logic.loadSegmentation()
        self.slicer.mrmlScene.RemoveNode.assert_called_once_with(node)

    def test_new_run_gets_a_fresh_directory(self):
        self.start()
        firstDir = self.logic.nnUNetOutDir
        Path(self.logic._outFile).write_bytes(b"old output")
        self.process.finished()
        self.process.start.reset_mock()
        self.start()
        self.assertNotEqual(firstDir, self.logic.nnUNetOutDir)
        self.assertFalse(Path(self.logic._outFile).exists())

    def test_wrong_channel_count_fails_before_export_or_process(self):
        self.logic.setParameter(_Parameter(channelNames={"0": "CT", "1": "Other"}))
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_not_called()
        self.slicer.util.exportNode.assert_not_called()
        self.assertIn("2 input channel", str(self.errors[-1]))

    def test_missing_channel_fails_before_export_or_process(self):
        self.logic.startSegmentation([None])
        self.process.start.assert_not_called()
        self.slicer.util.exportNode.assert_not_called()
        self.assertTrue(self.errors)

    def test_unavailable_device_fails_without_changing_device(self):
        self.logic._nnUNetParam.device = "cuda"
        self.logic._nnUNetParam.isSelectedDeviceAvailable = lambda: False
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_not_called()
        self.assertEqual(self.logic._nnUNetParam.device, "cuda")
        self.assertIn("unavailable", str(self.errors[-1]))

    def test_paths_with_spaces_are_separate_arguments(self):
        self.start()
        program, args, _ = self.process.start.call_args.args
        self.assertIn(" ", program)
        self.assertEqual(args[0], "-i")
        self.assertEqual(args[1], str(self.logic.nnUNetInDir))
        self.assertIn(" ", args[1])

    def test_predictor_discovery_does_not_pick_another_matching_command(self):
        with tempfile.TemporaryDirectory(prefix="runtime with spaces ") as tmp:
            scripts = Path(tmp) / "bin"
            scripts.mkdir()
            (scripts / "nnUNetv2_predict_from_modelfolder").write_text("different CLI")
            with patch("sysconfig.get_path", return_value=str(scripts)), \
                    patch.object(self.module.SegmentationLogic, "_nnUNetPythonDir", return_value=Path(tmp)):
                self.assertIsNone(self.module.SegmentationLogic._findUNetPredictPath())
                command = scripts / ("nnUNetv2_predict.exe" if sys.platform == "win32" else "nnUNetv2_predict")
                command.write_text("supported CLI")
                self.assertEqual(self.module.SegmentationLogic._findUNetPredictPath(), command)


class ProcessSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _ = _load_logic()

    def setUp(self):
        self.process = self.module.Process(_QProcess.MergedChannels)
        self.errors = []
        self.done = []
        self.info = []
        self.process.errorOccurred.connect(self.errors.append)
        self.process.finished.connect(lambda: self.done.append(True))
        self.process.readInfo.connect(self.info.append)

    def test_zero_normal_exit_emits_success(self):
        self.process.start("predict", [], 0)
        self.process.process.finished(0, _QProcess.NormalExit)
        self.assertEqual(self.done, [True])
        self.assertEqual(self.errors, [])

    def test_single_argument_finished_queries_native_exit_status(self):
        self.process.process.exitStatus = lambda: _QProcess.NormalExit
        self.process.start("predict", [], 0)
        self.process.process.finished(0)
        self.assertEqual(self.done, [True])
        self.assertEqual(self.errors, [])

    def test_nonzero_exit_emits_error_without_success(self):
        self.process.start("predict", [], 0)
        self.process.process.finished(1, _QProcess.NormalExit)
        self.assertEqual(self.done, [])
        self.assertTrue(self.errors)

    def test_crash_emits_error_without_success(self):
        self.process.start("predict", [], 0)
        self.process.process.finished(0, _QProcess.CrashExit)
        self.assertEqual(self.done, [])
        self.assertTrue(self.errors)

    def test_start_error_emits_actionable_error_even_with_empty_stderr(self):
        self.process.process.errorOccurred(0)
        self.assertTrue(self.errors)
        self.assertIn("start", str(self.errors[-1]).lower())

    def test_cancellation_never_emits_success(self):
        self.process.start("predict", [], 0)
        self.process.stop()
        self.assertEqual(self.done, [])

    def test_invalid_utf8_output_is_reported_without_crashing(self):
        self.process.process.output = b"progress \xff\n"
        self.process.process.readyRead()
        self.assertTrue(self.info)
        self.assertIn("progress", self.info[-1])

    def test_memory_error_preserves_original_diagnostic(self):
        self.process.start("predict", [], 0)
        self.process.process.output = b"CUDA out of memory: allocation failed\n"
        self.process.process.readyRead()
        self.process.process.finished(1, _QProcess.NormalExit)
        self.assertIn("CUDA out of memory", self.errors[-1])
        self.assertIn("memory", self.errors[-1].lower())
        self.assertEqual(self.done, [])

    def test_error_followed_by_exit_emits_only_one_error(self):
        self.process.start("predict", [], 0)
        self.process.process.errorOccurred(0)
        self.process.process.finished(1, _QProcess.CrashExit)
        self.assertEqual(len(self.errors), 1)
        self.assertEqual(self.done, [])


if __name__ == "__main__":
    unittest.main()
