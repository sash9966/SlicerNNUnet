"""Exercise the real Qt process wrapper without model weights, images or dependencies."""
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock

import qt
import slicer

from SlicerNNUNetLib.SegmentationLogic import Process


class ProcessTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suffix = ".exe" if sys.platform == "win32" else ""
        candidates = [Path(slicer.app.applicationDirPath()) / f"PythonSlicer{suffix}"]
        executable = Path(sys.executable)
        if "python" in executable.name.lower():
            candidates.append(executable)
        cls.pythonExecutable = next((path for path in candidates if path.is_file()), None)
        if cls.pythonExecutable is None:
            raise unittest.SkipTest("Slicer's PythonSlicer executable is unavailable for real QProcess tests.")

    def setUp(self):
        self.process = Process(qt.QProcess.MergedChannels)
        self.finished = MagicMock()
        self.errors = MagicMock()
        self.importResults = MagicMock()
        self.output = []
        self.process.finished.connect(self.finished)
        self.process.finished.connect(self.importResults)
        self.process.errorOccurred.connect(self.errors)
        self.process.readInfo.connect(self.output.append)

    def tearDown(self):
        self.process.stop()
        self.process.waitForFinished(1000)
        slicer.app.processEvents()

    def _start(self, script):
        self.process.start(str(self.pythonExecutable), ["-c", script],
                           qt.QProcess.Unbuffered | qt.QProcess.ReadOnly)
        self.assertTrue(self.process.process.waitForStarted(3000),
                        f"Cannot start {self.pythonExecutable}: {self.process.process.errorString()}")

    def _waitForExit(self):
        self.process.waitForFinished(3000)
        slicer.app.processEvents()
        self.assertEqual(self.process.process.state(), qt.QProcess.NotRunning,
                         "Test subprocess did not finish within three seconds")

    def test_zero_exit_emits_success_and_streams_output(self):
        self._start("import sys; print(sys.executable); print('native-process-ok', flush=True)")
        self._waitForExit()
        self.finished.assert_called_once_with()
        self.importResults.assert_called_once_with()
        self.errors.assert_not_called()
        self.assertIn("native-process-ok", "".join(self.output))

    def test_nonzero_exit_preserves_memory_diagnostic_and_never_emits_success(self):
        self._start("import sys; sys.stderr.write('Traceback (most recent call last):\\n"
                    "RuntimeError: CUDA out of memory\\n'); sys.exit(2)")
        self._waitForExit()
        self.finished.assert_not_called()
        self.importResults.assert_not_called()
        self.errors.assert_called_once()
        diagnostic = self.errors.call_args[0][0]
        self.assertIn("code 2", diagnostic)
        self.assertIn("memory was exhausted", diagnostic)
        self.assertIn("Traceback", diagnostic)
        self.assertIn("CUDA out of memory", diagnostic)

    def test_nonzero_exit_reports_unavailable_trainer_dependency(self):
        self._start("import sys; sys.stderr.write('ModuleNotFoundError: No module named original_trainer\\n'); sys.exit(2)")
        self._waitForExit()
        self.finished.assert_not_called()
        self.importResults.assert_not_called()
        self.errors.assert_called_once()
        diagnostic = self.errors.call_args[0][0]
        self.assertIn("original trainer, architecture and compatible runtime", diagnostic)
        self.assertIn("original_trainer", diagnostic)

    def test_cancelled_process_never_emits_success_or_imports_output(self):
        self._start("import time; print('cancel-ready', flush=True); time.sleep(30)")
        deadline = time.monotonic() + 3
        while "cancel-ready" not in "".join(self.output) and time.monotonic() < deadline:
            self.process.process.waitForReadyRead(20)
            slicer.app.processEvents()
        self.assertIn("cancel-ready", "".join(self.output), "Child must be running before cancellation")
        self.process.stop()
        self._waitForExit()
        self.finished.assert_not_called()
        self.importResults.assert_not_called()
        self.errors.assert_not_called()
