import traceback
from pathlib import Path
from typing import Optional

import qt
import slicer
from slicer.parameterNodeWrapper import parameterNodeWrapper

from .InstallLogic import InstallLogic, InstallLogicProtocol
from .Parameter import Parameter
from .ModelRegistry import ModelRegistry
from .SegmentationLogic import SegmentationLogic, SegmentationLogicProtocol


@parameterNodeWrapper
class WidgetParameterNode:
    parameter: Parameter


class Widget(qt.QWidget):
    """
    nnUNet widget containing an install and run settings collapsible areas.
    Allows to run nnUNet model and displays results in the UI.
    Saves the used settings to QSettings for reloading.
    """

    def __init__(
            self,
            segmentationLogic: Optional[SegmentationLogicProtocol] = None,
            installLogic: Optional[InstallLogicProtocol] = None,
            doShowInfoWindows: bool = True,
            parent=None,
            settings=None
    ):
        super().__init__(parent)
        self.logic = segmentationLogic or SegmentationLogic()
        self.installLogic = installLogic or InstallLogic()

        self.additionalInputSelectors = []
        self.modelRegistry = ModelRegistry(settings)
        self._selectedModel = self.modelRegistry.selected
        self._inferenceRunning = False
        self._inferenceFailed = False
        self._applyGeneration = 0

        # Instantiate widget UI
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        uiWidget = slicer.util.loadUI(self.resourcePath().joinpath("UI/SlicerNNUNet.ui").as_posix())
        uiWidget.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputsChanged)
        self.ui.installButton.clicked.connect(self.onInstall)
        self.ui.registeredModelComboBox.addItems(list(ModelRegistry.names))
        self.ui.registeredModelComboBox.setCurrentIndex(ModelRegistry.names.index(self._selectedModel))
        self.ui.registeredModelComboBox.currentIndexChanged.connect(self.onRegisteredModelChanged)
        self.ui.saveModelButton.clicked.connect(self.onSaveModel)

        self.ui.applyButton.setIcon(self.icon("start_icon.png"))
        self.ui.applyButton.clicked.connect(self.onApply)

        self.ui.stopButton.setIcon(self.icon("stop_icon.png"))
        self.ui.stopButton.clicked.connect(self.onStopClicked)

        # Logic connection
        self.logic.inferenceFinished.connect(self.onInferenceFinished)
        self.logic.errorOccurred.connect(self.onInferenceError)
        self.logic.progressInfo.connect(self.onProgressInfo)
        self.installLogic.progressInfo.connect(self.onProgressInfo)
        self.isStopping = False

        self.sceneCloseObserver = slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onSceneChanged)
        self._doShowErrorWindows = doShowInfoWindows

        # Create parameter node and connect GUI
        self._parameterNode = self._createParameterNode()
        savedParameter = self.modelRegistry.load(self._selectedModel)
        self._parameterNode.parameter = savedParameter or (
            Parameter.fromSettings(self.modelRegistry.settings) if self._selectedModel == "Manual" else Parameter()
        )
        self._parameterNode.connectParametersToGui(
            {
                "parameter.modelPath": self.ui.nnUNetModelPathEdit,
                "parameter.device": self.ui.deviceComboBox,
                "parameter.stepSize": self.ui.stepSizeSlider,
                "parameter.checkPointName": self.ui.checkPointNameLineEdit,
                "parameter.folds": self.ui.foldsLineEdit,
                "parameter.nProcessPreprocessing": self.ui.nProcessPreprocessingSpinBox,
                "parameter.nProcessSegmentationExport": self.ui.nProcessSegmentationExportSpinBox,
                "parameter.disableTta": self.ui.disableTtaCheckBox
            }
        )

        self.ui.nnUNetModelPathEdit.connect("currentPathChanged(QString)", self.onModelPathChanged)
        self.ui.foldsLineEdit.textChanged.connect(self.onInputsChanged)
        self.ui.checkPointNameLineEdit.textChanged.connect(self.onInputsChanged)
        self.ui.stepSizeSlider.valueChanged.connect(self.onInputsChanged)

        # Configure UI
        self.updateInstalledVersion()
        self._setApplyVisible(True)

        self.onModelPathChanged()
        self._updateModelSelectionStatus(savedParameter is not None)
        for warning in self.modelRegistry.warnings:
            self.onLogMessage(warning)

    def _updateModelSelectionStatus(self, registered):
        self.ui.saveModelButton.setEnabled(self._selectedModel != "Manual" and not self._inferenceRunning)
        if self._selectedModel == "Manual":
            self.ui.modelSelectionStatusLabel.setText("Choose a model folder in nnUNet Run Settings.")
        elif registered:
            self.ui.modelSelectionStatusLabel.setText("To move weights, update the model path and click Save/Update.")
        else:
            self.ui.modelSelectionStatusLabel.setText("Set this model's path, folds and checkpoint in Run Settings, then Save/Update.")
            self.ui.nnUNetSettingsCollapsibleButton.collapsed = False

    def onRegisteredModelChanged(self, index):
        name = ModelRegistry.names[int(index)]
        if name == self._selectedModel:
            return
        if self._selectedModel == "Manual":
            self.modelRegistry.save("Manual", self._parameterNode.parameter)
        self._selectedModel = name
        self.modelRegistry.select(name)
        savedParameter = self.modelRegistry.load(name)
        self._parameterNode.parameter = savedParameter or Parameter()
        self.onModelPathChanged()
        self._updateModelSelectionStatus(savedParameter is not None)
        if self.modelRegistry.warnings:
            self.onLogMessage(self.modelRegistry.warnings[-1])

    def onSaveModel(self, *_):
        if self._selectedModel == "Manual" or self._inferenceRunning:
            return
        parameter = self._parameterNode.parameter
        valid, reason = parameter.isValid()
        if not valid:
            self._reportError(f"Cannot save {self._selectedModel}: {reason}", doTraceback=False)
            return
        # Store explicit native selections, including the original manual defaults.
        parameter.folds = parameter.folds.strip() or "0"
        parameter.checkPointName = parameter._getCheckpointName()
        self.modelRegistry.save(self._selectedModel, parameter)
        self._updateModelSelectionStatus(True)
        self.onLogMessage(f"Saved {self._selectedModel} settings.")

    def onUpdateAdditionalInputSelectors(self):
        channels = self._parameterNode.parameter.readChannelNamesFromFile()

        formLayout = self.ui.additionalInputsFrame.layout()
        if len(self.additionalInputSelectors) > 0:
            for i in reversed(range(formLayout.rowCount())):
                formLayout.removeRow(i)
            self.additionalInputSelectors.clear()

        for idx, channel in enumerate(channels):
            if idx == 0:
                continue
            inputSelector = slicer.qMRMLNodeComboBox()
            inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
            inputSelector.noneEnabled = True
            inputSelector.showHidden = False
            inputSelector.renameEnabled = True
            inputSelector.setMRMLScene(slicer.mrmlScene)
            inputSelector.currentNodeChanged.connect(self.onInputsChanged)
            formLayout.addRow(f"Input volume {idx + 1}: ", inputSelector)
            self.additionalInputSelectors.append(inputSelector)

    def onModelPathChanged(self):
        nnUNetParam = self._parameterNode.parameter
        isValid, messages = nnUNetParam.isValid()
        self.ui.logTextEdit.clear()
        if messages:
            self.onLogMessage(messages)

        self.onUpdateAdditionalInputSelectors()
        self.onInputsChanged()

    @staticmethod
    def _createParameterNode() -> WidgetParameterNode:
        moduleName = "SlicerNNUNet"
        parameterNode = slicer.mrmlScene.GetSingletonNode(moduleName, "vtkMRMLScriptedModuleNode")
        if not parameterNode:
            parameterNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            parameterNode.SetName(slicer.mrmlScene.GenerateUniqueName(moduleName))

        parameterNode.SetAttribute("ModuleName", moduleName)
        return WidgetParameterNode(parameterNode)

    @staticmethod
    def resourcePath() -> Path:
        return Path(__file__).parent.joinpath("..", "Resources")

    @classmethod
    def icon(cls, icon_name) -> "qt.QIcon":
        return qt.QIcon(cls.resourcePath().joinpath("Icons", icon_name).as_posix())

    def __del__(self):
        if getattr(self, "sceneCloseObserver", None):
            slicer.mrmlScene.RemoveObserver(self.sceneCloseObserver)

    def _setButtonsEnabled(self, isEnabled):
        self.ui.installButton.setEnabled(isEnabled)
        self.ui.applyButton.setEnabled(isEnabled)
        self.ui.inputSelector.setEnabled(isEnabled)
        self.ui.registeredModelComboBox.setEnabled(isEnabled)
        self.ui.nnUNetSettingsCollapsibleButton.setEnabled(isEnabled)
        self.ui.saveModelButton.setEnabled(isEnabled and self._selectedModel != "Manual")
        for selector in self.additionalInputSelectors:
            selector.setEnabled(isEnabled)

    def onInstall(self, *, doReportFinished=True):
        self._setButtonsEnabled(False)

        if doReportFinished:
            self.ui.logTextEdit.clear()

        success = self.installLogic.setupPythonRequirements(f"nnunetv2{self.ui.toInstallLineEdit.text}")
        if doReportFinished:
            if success:
                self._reportFinished("Install finished correctly.")
            else:
                self._reportError("Install failed.")
        self.updateInstalledVersion()
        self._setButtonsEnabled(True)
        self.onInputsChanged()
        return success

    def _reportError(self, msg, doTraceback=True):
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            all_msgs = (msg,) if not doTraceback else (msg, traceback.format_exc())
            slicer.util.errorDisplay(*all_msgs)

    def _reportFinished(self, msg):
        self.onProgressInfo("*" * 80)
        self.onProgressInfo(msg)
        if self._doShowErrorWindows:
            slicer.util.infoDisplay(msg)

    def onLogMessage(self, msg):
        self.ui.logTextEdit.insertPlainText(msg + "\n")

    def updateInstalledVersion(self):
        self.ui.currentVersionLabel.setText(str(self.installLogic.getInstalledNNUnetVersion()))

    def onSceneChanged(self, *_):
        self.onStopClicked()
        self.onModelPathChanged()

    def onStopClicked(self):
        self._applyGeneration += 1
        self._inferenceFailed = True
        self.isStopping = True
        self.logic.stopSegmentation()
        self.logic.waitForSegmentationFinished()
        slicer.app.processEvents()
        self.isStopping = False
        self._setApplyVisible(True)

    def onApply(self, *_):
        if self._inferenceRunning:
            return
        volumes = self.getCurrentVolumeNodes()
        channels = self._parameterNode.parameter.readChannelNamesFromFile()
        valid, reason = self._parameterNode.parameter.isValid()
        if not valid:
            self._reportError(f"Invalid model settings: {reason}", doTraceback=False)
            return
        if not channels or len(volumes) != len(channels) or any(volume is None for volume in volumes):
            self._reportError("Select one input volume for each model channel.", doTraceback=False)
            return

        if self.installLogic.getInstalledNNUnetVersion() is None:
            self._reportError(
                "nnU-Net is not installed. Use nnUNet Install to install the model's compatible runtime first.",
                doTraceback=False
            )
            return

        if self.installLogic.needsRestart:
            self._reportFinished("Please restart 3D Slicer to proceed with segmentation.")
            return

        self._setApplyVisible(False)
        self._inferenceFailed = False
        self._applyGeneration += 1
        generation = self._applyGeneration
        try:
            self.ui.logTextEdit.clear()
            self.onProgressInfo("Start")
            self.onProgressInfo("*" * 80)
            if generation == self._applyGeneration:
                self._runSegmentation()
        except Exception as error:
            self.onInferenceError(error)

    def _setApplyVisible(self, isVisible):
        self._inferenceRunning = not isVisible
        self.ui.applyButton.setVisible(isVisible)
        self.ui.stopButton.setVisible(not isVisible)
        self._setButtonsEnabled(isVisible)
        if isVisible:
            self.onInputsChanged()

    def _runSegmentation(self):
        if not self._inferenceRunning or self._inferenceFailed:
            return
        if self._selectedModel == "Manual":
            self._parameterNode.parameter.toSettings(self.modelRegistry.settings)
            self.modelRegistry.save("Manual", self._parameterNode.parameter)
        self.logic.setParameter(self._parameterNode.parameter)
        self.logic.startSegmentation(self.getCurrentVolumeNodes())

    def onInputsChanged(self, *_):
        nChannels = len(self._parameterNode.parameter.readChannelNamesFromFile())
        volumeNodes = [v for v in self.getCurrentVolumeNodes() if v is not None]
        valid, reason = self._parameterNode.parameter.isValid()
        self.ui.applyButton.setEnabled(bool(not self._inferenceRunning and valid and nChannels and len(volumeNodes) == nChannels))
        self.ui.applyButton.setToolTip(reason if not valid else "Run nnUNet segmentation on the selected volumes.")

    def getCurrentVolumeNodes(self):
        return [self.ui.inputSelector.currentNode()] + [s.currentNode() for s in self.additionalInputSelectors]

    def onInferenceFinished(self, *_):
        if self.isStopping or self._inferenceFailed:
            self._setApplyVisible(True)
            return

        try:
            self.onProgressInfo("Loading inference results...")
            segmentation = self.logic.loadSegmentation()
            segmentation.SetName(self.ui.inputSelector.currentNode().GetName() + "Segmentation")
            self._reportFinished("Inference ended successfully.")
        except Exception as e:
            self._reportError(f"Inference ended in error:\n{e}")
        finally:
            self._setApplyVisible(True)

    def onInferenceError(self, errorMsg):
        self._inferenceFailed = True
        if self.isStopping:
            return

        self._setApplyVisible(True)
        if isinstance(errorMsg, Exception):
            errorMsg = str(errorMsg)
        self._reportError("Encountered error during inference :\n" + errorMsg, doTraceback=False)

    def onProgressInfo(self, infoMsg):
        self.ui.logTextEdit.insertPlainText(self._formatMsg(infoMsg) + "\n")
        self.moveTextEditToEnd(self.ui.logTextEdit)
        slicer.app.processEvents()

    @staticmethod
    def _formatMsg(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines()])

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)
