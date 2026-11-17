import os
import shlex
import sys
from pathlib import Path
from typing import Protocol, List, Optional, Callable, Union

import qt
import slicer

from .Parameter import Parameter
from .Signal import Signal


class SegmentationLogicProtocol(Protocol):
    """
    Segmentation interface class.
    This class is used as interface which is expected by the NNUnet Widget.
    Any instance implementing this interface is compatible with the NNUnet Widget.
    """
    inferenceFinished: Signal
    errorOccurred: Signal
    progressInfo: Signal

    def setParameter(self, nnUNetParam: Parameter) -> None:
        pass

    def startSegmentation(
            self,
            volumeNodes: Union[slicer.vtkMRMLScalarVolumeNode, list[slicer.vtkMRMLScalarVolumeNode]]
    ) -> None:
        pass

    def stopSegmentation(self):
        pass

    def waitForSegmentationFinished(self):
        pass

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        pass


class ProcessProtocol(Protocol):
    """
    Interface for NNUnet process runner.
    Process is responsible for running the NNUnet prediction and reporting progress / errors.
    """

    errorOccurred: Signal
    finished: Signal
    readInfo: Signal

    def start(self, program: str, args: List[str]) -> None:
        pass

    def stop(self) -> None:
        pass

    def waitForFinished(self) -> None:
        pass


class SegmentationLogic:
    r"""
    Segmentation logic for nnUNet based segmentations.

    This class is responsible for writing a volume file with correct nnUNet formatting, calling the nnUNet detection
    on the given folder with user inputs and user model and allowing to load the generated segmentation afterwards.

    At the start of the nnUNet detection, a temporary folder is created with the selected input volume.
    The nnUNet detection QProcess is called with user parameters.

    After a normal zero process exit, the Segmentation Logic triggers inferenceFinished.
    Failures are reported through errorOccurred and cannot be imported.

    During execution, the standard output is returned using the progressInfo signal. If any error occurs, the
    errorOccurred sigal is used.

    The loadSegmentation method can be called to load the segmentation results. If the segmentation failed, this method
    call will raise a RuntimeError Exception.

    Usage example :
    >>> from SlicerNNUNetLib import SegmentationLogic, Parameter
    >>>
    >>> # Create instance
    >>> logic = SegmentationLogic()
    >>>
    >>> # Connect progress and error logging
    >>> logic.progressInfo.connect(print)
    >>> logic.errorOccurred.connect(slicer.util.errorDisplay)
    >>>
    >>> # Connect processing done signal
    >>> logic.inferenceFinished.connect(logic.loadSegmentation)
    >>>
    >>> # Start segmentation on given volume node on 5 folds
    >>> param = Parameter(modelPath=Path(r"C:\<PATH_TO>\NNUnetModel\Dataset123_456MRI"), folds = "0,1,2,3,4")
    >>> import SampleData
    >>> volumeNode = SampleData.downloadSample("MRHead")
    >>> logic.setParameter(param)
    >>> logic.startSegmentation(volumeNode)
    """

    def __init__(self, process: Optional[ProcessProtocol] = None):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")

        self.inferenceProcess = process or Process(qt.QProcess.MergedChannels)
        self.inferenceProcess.finished.connect(self._onProcessFinished)
        self.inferenceProcess.errorOccurred.connect(self._onProcessError)
        self.inferenceProcess.readInfo.connect(self.progressInfo)

        self._nnUNet_predict_path = None
        self._nnUNetParam: Optional[Parameter] = None
        self._tmpDir = qt.QTemporaryDir()
        self._running = False
        self._completedSuccessfully = False
        self._cancelled = False
        self._referenceVolume = None
        self._runGeneration = 0

    def __del__(self):
        self.stopSegmentation()

    def setParameter(self, nnUnetConf: Parameter):
        # Keep a run independent of edits to the widget's parameter node.
        self._nnUNetParam = Parameter(**nnUnetConf.asDict())

    def startSegmentation(
            self,
            volumeNodes: Union[slicer.vtkMRMLScalarVolumeNode, list[slicer.vtkMRMLScalarVolumeNode]]
    ) -> None:
        """ Run segmentation on a single Slicer volume node or a list of volume nodes.

        If a list of volume nodes is provided, the list must be in the input order expected by the nnUNet model.
        """
        if not isinstance(volumeNodes, list):
            volumeNodes = [volumeNodes]
        self.stopSegmentation()
        self.waitForSegmentationFinished()
        generation = self._runGeneration
        self._cancelled = False
        self._completedSuccessfully = False
        # Check the nnUNet parameters are correct
        try:
            self._getNNUNetParamArgsOrRaise()
            expected = len(self._nnUNetParam.readChannelNamesFromFile())
            if len(volumeNodes) != expected or any(node is None for node in volumeNodes):
                raise RuntimeError(f"Model requires {expected} input channel(s), in dataset.json order; "
                                   f"received {len(volumeNodes)}. Select every input volume.")
            if not self._nnUNetParam.isSelectedDeviceAvailable():
                raise RuntimeError(f"Selected device '{self._nnUNetParam.device}' is unavailable. "
                                   "Check the PyTorch build and device driver, or explicitly select another device.")
        except (RuntimeError, ImportError) as e:
            self.errorOccurred(f"Cannot start inference: {e}. Check the selected model and its compatible PyTorch/nnU-Net runtime.")
            return

        # Prepare the inference directory
        try:
            prepared = self._prepareInferenceDir(volumeNodes, generation)
            if self._cancelled or generation != self._runGeneration:
                return
            if not prepared:
                raise RuntimeError(f"Failed to export input volumes to {self.nnUNetInDir}")
        except Exception as e:
            self.errorOccurred(str(e))
            return

        # Launch the nnUNet processing
        self._startInferenceProcess(generation)

    def stopSegmentation(self):
        self._runGeneration += 1
        self._cancelled = True
        self._completedSuccessfully = False
        self._running = False
        self.inferenceProcess.stop()

    def _onProcessError(self, message):
        self._running = False
        self._completedSuccessfully = False
        self.errorOccurred(str(message))

    def _onProcessFinished(self, *_):
        # Process.finished is emitted only after a normal zero exit.
        if not self._running or self._cancelled:
            return
        self._running = False
        self._completedSuccessfully = True
        self.inferenceFinished()

    def waitForSegmentationFinished(self):
        self.inferenceProcess.waitForFinished()

    def loadSegmentation(self) -> "slicer.vtkMRMLSegmentationNode":
        if not self._completedSuccessfully or self._cancelled:
            raise RuntimeError("No successful completed inference is available. Failed or cancelled output is not imported.")
        if not Path(self._outFile).is_file():
            raise RuntimeError(f"Inference did not produce the expected result: {self._outFile}")
        segmentationNode = None
        try:
            segmentationNode = slicer.util.loadSegmentation(self._outFile)
            if segmentationNode is None:
                raise RuntimeError("Slicer could not read the prediction as an editable segmentation.")
            self._renameSegments(segmentationNode)
            segmentationNode.SetAndObserveTransformNodeID(self._referenceVolume.GetTransformNodeID())
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self._referenceVolume)
            return segmentationNode
        except Exception as e:
            if segmentationNode is not None:
                slicer.mrmlScene.RemoveNode(segmentationNode)
            raise RuntimeError(f"Failed to import segmentation: {e}") from e

    def _renameSegments(self, segmentationNode: "slicer.vtkMRMLSegmentationNode") -> None:
        """
        Rename loaded segments with dataset file labels dict.
        """
        labels = self._nnUNetParam.readSegmentIdsAndLabelsFromDatasetFile()
        if labels is None:
            return

        for segmentId, label in labels:
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            if segment is None:
                continue
            segment.SetName(label)

    @staticmethod
    def _nnUNetPythonDir():
        return Path(sys.executable).parent.joinpath("..", "lib", "Python")

    @classmethod
    def _findUNetPredictPath(cls):
        # nnUNet install dir depends on OS. For Windows, install will be done in the Scripts dir.
        # For Linux and MacOS, install will be done in the bin folder.
        # The scripts dir reported by sysconfig is checked first, as it is also correct in a Slicer build tree.
        import sysconfig
        scriptDirs = [Path(sysconfig.get_path("scripts"))]
        scriptDirs += [cls._nnUNetPythonDir().joinpath(path) for path in ["Scripts", "bin"]]
        for scriptDir in scriptDirs:
            # Other nnUNetv2_predict* commands accept different arguments.
            name = "nnUNetv2_predict.exe" if sys.platform == "win32" else "nnUNetv2_predict"
            candidate = scriptDir / name
            if candidate.is_file():
                return candidate

        return None

    def _startInferenceProcess(self, generation=None):
        """
        Run the nnU-Net V2 inference script
        """
        # Check the nnUNet predict script is correct
        nnUnetPredictPath = self._findUNetPredictPath()
        if not nnUnetPredictPath:
            self.errorOccurred("Cannot find the nnUNetv2_predict entry point in Slicer's Python runtime. "
                               "Use nnUNet Install explicitly to install a model-compatible runtime, then restart Slicer.")
            return

        # Get the nnUNet parameters as arg list
        try:
            args = self._getNNUNetParamArgsOrRaise()
        except RuntimeError as e:
            self.errorOccurred(e)
            return

        # setup environment variables
        # not needed, just needs to be an existing directory
        os.environ['nnUNet_preprocessed'] = self._nnUNetParam.modelFolder.as_posix()

        # not needed, just needs to be an existing directory
        os.environ['nnUNet_raw'] = self._nnUNetParam.modelFolder.as_posix()
        os.environ['nnUNet_results'] = self._nnUNetParam.modelFolder.as_posix()

        # Prevent nnUNet from querying hostname via subprocess in get_allowed_n_proc_DA which may lead
        # to decode errors on non-Windows machines.
        cpu_count = os.cpu_count() or 1
        os.environ['nnUNet_n_proc_DA'] = str(min(12, cpu_count))
        os.environ['nnUNet_def_n_proc'] = str(min(8, cpu_count))

        argListStr = shlex.join([str(a) for a in args])
        self.progressInfo(
            "Starting nnUNet with the following parameters:\n"
            f"\n{nnUnetPredictPath} {argListStr}\n\n"
            "JSON parameters :\n"
            f"{self._nnUNetParam.debugString()}\n"
        )
        self.progressInfo("nnUNet preprocessing...\n")
        # Progress callbacks may process UI events, including Stop/scene close.
        if self._cancelled or (generation is not None and generation != self._runGeneration):
            return
        self._running = True
        try:
            self.inferenceProcess.start(str(nnUnetPredictPath), [str(a) for a in args],
                                        qt.QProcess.Unbuffered | qt.QProcess.ReadOnly)
        except Exception as e:
            self._onProcessError(e)

    def _getNNUNetParamArgsOrRaise(self):
        if self._nnUNetParam is None:
            raise RuntimeError("Select a model before starting inference.")
        return self._nnUNetParam.asArgList(self.nnUNetInDir, self.nnUNetOutDir)

    @property
    def _outFile(self) -> str:
        return (self.nnUNetOutDir / f"volume{self._fileEnding}").as_posix()

    def _prepareInferenceDir(self, volumeNodes, generation=None) -> bool:
        if not volumeNodes:
            return False
        # A new run cannot reuse partial output from an earlier invocation.
        self._tmpDir = qt.QTemporaryDir()
        if not self._tmpDir.isValid():
            raise RuntimeError("Cannot create a temporary inference directory. Check available disk space.")
        self.nnUNetOutDir.mkdir(parents=True)
        self.nnUNetInDir.mkdir(parents=True)
        self._referenceVolume = volumeNodes[0]

        # Name of the volume should match expected nnUNet conventions
        self.progressInfo(f"Transferring volume to nnUNet in {self._tmpDir.path()}\n")
        volumePaths = []
        for idx, volumeNode in enumerate(volumeNodes):
            if self._cancelled or (generation is not None and generation != self._runGeneration):
                return False
            volumePaths.append(
                volumePath := self.nnUNetInDir.joinpath(f"volume_{idx:04d}{self._fileEnding}")
            )
            if not slicer.util.exportNode(volumeNode, str(volumePath), {}):
                return False

        return all(volumePath.exists() for volumePath in volumePaths)

    @property
    def _fileEnding(self):
        return self._nnUNetParam.readFileEndingFromDatasetFile() if self._nnUNetParam else ".nii.gz"

    @property
    def nnUNetOutDir(self):
        return Path(self._tmpDir.path()).joinpath("output")

    @property
    def nnUNetInDir(self):
        return Path(self._tmpDir.path()).joinpath("input")


class Process:
    """
    Convenience wrapper around QProcess run.
    Forwards the process read / error events when read.
    Kills process on stop if process is running.
    """

    def __init__(self, channelMode: qt.QProcess.ProcessChannelMode):
        self.errorOccurred = Signal("str")
        self.finished = Signal()
        self.readInfo = Signal("str")

        self.process = qt.QProcess()
        self._cancelled = False
        self._failed = False
        self._logTail = ""
        self.process.setProcessChannelMode(channelMode)
        self.process.finished.connect(self._onFinished)
        self.process.errorOccurred.connect(self._onErrorOccurred)
        self.process.readyRead.connect(self._onReadyRead)

    def stop(self):
        if self.process.state() != self.process.NotRunning:
            self._cancelled = True
            self.readInfo("Killing process.")
            self.process.kill()

    def start(self, program, args, openMode: qt.QIODevice.OpenMode):
        self.stop()
        self._cancelled = False
        self._failed = False
        self._logTail = ""
        self.process.start(program, args, openMode)

    def waitForFinished(self, timeOut_ms: Optional[int] = None):
        self.process.waitForFinished(timeOut_ms if timeOut_ms is not None else -1)

    def _onReadyRead(self):
        info = bytes(self.process.readAll().data()).decode("utf-8", errors="replace")
        if info:
            self._logTail = (self._logTail + info)[-12000:]
            self.readInfo(info)

    def _onErrorOccurred(self, *_):
        self._onReadyRead()
        if not self._cancelled and not self._failed:
            self._failed = True
            self.errorOccurred(f"Could not run nnU-Net: {self.process.errorString()}\n{self._logTail}")

    def _onFinished(self, exitCode, exitStatus=None):
        # PythonQt may bind QProcess.finished(int) rather than its two-argument overload.
        if exitStatus is None:
            exitStatus = self.process.exitStatus()
        self._onReadyRead()
        if self._cancelled or self._failed:
            return
        if exitCode != 0 or exitStatus != qt.QProcess.NormalExit:
            self._failed = True
            diagnostic = self._logTail
            lower = diagnostic.lower()
            if "out of memory" in lower or "cannot allocate memory" in lower:
                hint = "Device or host memory was exhausted. Free memory or choose suitable hardware explicitly. "
            elif "modulenotfounderror" in lower or "importerror" in lower or "trainer" in lower:
                hint = "Check that this model's original trainer, architecture and compatible runtime are installed. "
            else:
                hint = "Check the model metadata, original trainer/architecture and runtime versions. "
            self.errorOccurred(f"nnU-Net exited with code {exitCode} (status {exitStatus}). "
                               f"{hint}No result was imported.\n{diagnostic}")
            return
        self.finished()

    @staticmethod
    def _report(stream: "qt.QByteArray", outSignal: Callable[[str], None]) -> None:
        # QTextCodec is not available in Qt6. Invalid UTF-8 bytes are replaced instead of raising an error.
        info = bytes(stream.data()).decode("utf-8", errors="replace")
        if info:
            outSignal(info)
