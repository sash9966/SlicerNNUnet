import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import qt
import slicer
from packaging.version import parse

from SlicerNNUNetLib import Widget, Signal


class MockSegmentationLogic:
    def __init__(self):
        self.inferenceFinished = Signal()
        self.errorOccurred = Signal("str")
        self.progressInfo = Signal("str")
        self.setParameter = MagicMock()
        self.startSegmentation = MagicMock()
        self.stopSegmentation = MagicMock()
        self.waitForSegmentationFinished = MagicMock()
        self.loadSegmentation = MagicMock()
        self.loadSegmentation.side_effect = self.load_segmentation

    @staticmethod
    def load_segmentation():
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")


class MockInstallLogic:
    def __init__(self):
        self.progressInfo = Signal()
        self.needsRestart = False
        self.setupPythonRequirements = MagicMock(return_value=True)
        self.getInstalledNNUnetVersion = MagicMock(return_value=parse("2.2.3"))


class WidgetTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.segmentationLogic = MockSegmentationLogic()
        self.installLogic = MockInstallLogic()
        self.temporaryDirectory = tempfile.TemporaryDirectory(prefix="nnunet widget ")
        self.root = Path(self.temporaryDirectory.name)
        self.settings = qt.QSettings(str(self.root / "settings.ini"), qt.QSettings.IniFormat)
        self.widget = Widget(
            segmentationLogic=self.segmentationLogic,
            installLogic=self.installLogic,
            doShowInfoWindows=False,
            settings=self.settings
        )
        self.node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "WidgetTestInput")
        self.modelPath = self.makeModel("Dataset001_Widget", "0")
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath(str(self.modelPath))
        self.widget.ui.inputSelector.setCurrentNode(self.node)
        self.widget.show()
        self.createdSegmentations = []

    def tearDown(self):
        slicer.mrmlScene.RemoveObserver(self.widget.sceneCloseObserver)
        self.widget.sceneCloseObserver = None
        self.widget.close()
        self.widget.deleteLater()
        slicer.mrmlScene.RemoveNode(self.node)
        for node in self.createdSegmentations:
            slicer.mrmlScene.RemoveNode(node)
        self.temporaryDirectory.cleanup()
        super().tearDown()

    def makeModel(self, dataset, fold, checkpoint="checkpoint_final.pth"):
        model = self.root / dataset / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        (model / f"fold_{fold}").mkdir(parents=True)
        (model / f"fold_{fold}" / checkpoint).touch()
        (model / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CT"}, "labels": {"background": 0, "heart": 1}, "file_ending": ".nii.gz"
        }))
        (model / "plans.json").write_text(json.dumps({
            "dataset_name": dataset, "plans_name": "nnUNetPlans", "configurations": {"3d_fullres": {}}
        }))
        return model

    def assertButtonsEnabled(self):
        self.assertTrue(self.widget.ui.inputSelector.isEnabled())
        self.assertTrue(self.widget.ui.installButton.isEnabled())
        self.assertTrue(self.widget.ui.registeredModelComboBox.isEnabled())
        self.assertTrue(self.widget.ui.applyButton.isVisible())
        self.assertFalse(self.widget.ui.stopButton.isVisible())

    def assertButtonsDisabled(self):
        self.assertFalse(self.widget.ui.inputSelector.isEnabled())
        self.assertFalse(self.widget.ui.installButton.isEnabled())
        self.assertFalse(self.widget.ui.registeredModelComboBox.isEnabled())
        self.assertFalse(self.widget.ui.nnUNetSettingsCollapsibleButton.isEnabled())
        self.assertFalse(self.widget.ui.applyButton.isVisible())
        self.assertTrue(self.widget.ui.stopButton.isVisible())

    def test_widget_starts_segmentation_without_installing_on_apply(self):
        self.installLogic.needsRestart = False

        self.widget.ui.applyButton.click()

        self.installLogic.setupPythonRequirements.assert_not_called()
        self.segmentationLogic.setParameter.assert_called_once()
        self.segmentationLogic.startSegmentation.assert_called_once_with([self.node])

        self.assertButtonsDisabled()
        self.segmentationLogic.inferenceFinished()
        self.segmentationLogic.loadSegmentation.assert_called_once()
        self.assertButtonsEnabled()

        # Check segmentation is loaded and contains the input volume name on load
        segmentations = [node for node in slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
                         if node.GetName() == self.node.GetName() + "Segmentation"]
        self.createdSegmentations.extend(segmentations)
        self.assertEqual(len(segmentations), 1)
        self.assertIn(self.node.GetName(), segmentations[0].GetName())

    def test_widget_doesnt_call_segmentation_if_restart_is_pending(self):
        self.installLogic.needsRestart = True

        self.widget.ui.applyButton.click()
        self.widget.ui.applyButton.click()

        self.segmentationLogic.startSegmentation.assert_not_called()
        self.assertButtonsEnabled()

    def test_missing_runtime_doesnt_install_or_start_segmentation(self):
        self.installLogic.getInstalledNNUnetVersion.return_value = None
        self.installLogic.needsRestart = False

        self.widget.ui.applyButton.click()
        self.segmentationLogic.startSegmentation.assert_not_called()
        self.installLogic.setupPythonRequirements.assert_not_called()
        self.assertButtonsEnabled()

    def test_nnunet_path_is_forwarded_to_segmentation_logic(self):
        self.widget.ui.applyButton.click()
        self.segmentationLogic.setParameter.assert_called_once()
        self.assertEqual(self.segmentationLogic.setParameter.call_args[0][0].modelPath, self.modelPath)

    def test_synchronous_start_error_restores_controls_and_does_not_import(self):
        self.segmentationLogic.startSegmentation.side_effect = RuntimeError("Trainer unavailable")
        self.widget.ui.applyButton.click()
        self.assertButtonsEnabled()
        self.widget.onInferenceFinished()
        self.segmentationLogic.loadSegmentation.assert_not_called()

    def test_failed_or_cancelled_run_does_not_import(self):
        self.widget.ui.applyButton.click()
        self.segmentationLogic.errorOccurred("GPU out of memory")
        self.segmentationLogic.inferenceFinished()
        self.segmentationLogic.loadSegmentation.assert_not_called()
        self.assertButtonsEnabled()
        self.widget.ui.applyButton.click()
        self.widget.onStopClicked()
        self.segmentationLogic.inferenceFinished()
        self.segmentationLogic.loadSegmentation.assert_not_called()
        self.assertButtonsEnabled()

    def test_saved_models_restore_independent_parameters_and_selection(self):
        self.widget.ui.registeredModelComboBox.setCurrentIndex(1)
        cnn = self.makeModel("Dataset002_CNN", "all", "checkpoint_best.pth")
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath(str(cnn))
        self.widget.ui.foldsLineEdit.setText("all")
        self.widget.ui.checkPointNameLineEdit.setText("checkpoint_best.pth")
        self.widget.ui.disableTtaCheckBox.setChecked(False)
        self.widget.ui.saveModelButton.click()
        self.widget.ui.registeredModelComboBox.setCurrentIndex(2)
        self.assertEqual(self.widget._parameterNode.parameter.modelPath, Path())
        vit = self.makeModel("Dataset003_ViT", "0")
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath(str(vit))
        self.widget.ui.disableTtaCheckBox.setChecked(True)
        self.widget.ui.saveModelButton.click()
        self.widget.ui.registeredModelComboBox.setCurrentIndex(1)
        parameter = self.widget._parameterNode.parameter
        self.assertEqual(parameter.modelPath, cnn)
        self.assertEqual(parameter.folds, "all")
        self.assertEqual(parameter.checkPointName, "checkpoint_best.pth")
        self.assertFalse(parameter.disableTta)
        # Reconstruct the widget from the same INI file, as after Slicer restart.
        slicer.mrmlScene.RemoveObserver(self.widget.sceneCloseObserver)
        self.widget.sceneCloseObserver = None
        self.widget.close()
        self.widget.deleteLater()
        self.widget = Widget(self.segmentationLogic, self.installLogic, False,
                             settings=qt.QSettings(str(self.root / "settings.ini"), qt.QSettings.IniFormat))
        self.assertEqual(self.widget._selectedModel, "CNN")
        self.assertEqual(self.widget._parameterNode.parameter.modelPath, cnn)
        self.widget.ui.registeredModelComboBox.setCurrentIndex(2)
        self.assertEqual(self.widget._parameterNode.parameter.modelPath, vit)
        self.assertEqual(self.widget._parameterNode.parameter.folds, "0")
        self.assertTrue(self.widget._parameterNode.parameter.disableTta)
        self.segmentationLogic.startSegmentation.assert_not_called()
        self.installLogic.setupPythonRequirements.assert_not_called()

    def test_moved_weights_update_only_after_save(self):
        from SlicerNNUNetLib.ModelRegistry import ModelRegistry

        self.widget.ui.registeredModelComboBox.setCurrentIndex(1)
        original = self.makeModel("Dataset004_MovedCNN", "0")
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath(str(original))
        self.widget.ui.saveModelButton.click()
        movedParent = self.root / "moved weights" / original.parent.name
        movedParent.parent.mkdir()
        original.parent.rename(movedParent)
        moved = movedParent / original.name
        self.widget.ui.nnUNetModelPathEdit.setCurrentPath(str(moved))
        self.assertEqual(ModelRegistry(self.settings).load("CNN").modelPath, original)
        self.widget.ui.saveModelButton.click()
        self.widget.ui.registeredModelComboBox.setCurrentIndex(2)
        self.widget.ui.registeredModelComboBox.setCurrentIndex(1)
        self.assertEqual(self.widget._parameterNode.parameter.modelPath, moved)
        self.assertTrue(self.widget._parameterNode.parameter.isValid()[0])
        self.assertEqual(ModelRegistry(self.settings).load("CNN").modelPath, moved)
        self.segmentationLogic.startSegmentation.assert_not_called()
        self.installLogic.setupPythonRequirements.assert_not_called()

    def test_stop_during_startup_progress_prevents_process_start(self):
        originalProgress = self.widget.onProgressInfo
        def progress(message):
            originalProgress(message)
            if message == "Start":
                self.widget.onStopClicked()
        self.widget.onProgressInfo = progress
        self.widget.onApply()
        self.segmentationLogic.startSegmentation.assert_not_called()
        self.assertButtonsEnabled()

    def test_stop_then_apply_during_progress_starts_only_the_new_run(self):
        originalProgress = self.widget.onProgressInfo
        restarted = False
        def progress(message):
            nonlocal restarted
            originalProgress(message)
            if message == "Start" and not restarted:
                restarted = True
                self.widget.onStopClicked()
                self.widget.onApply()
        self.widget.onProgressInfo = progress
        self.widget.onApply()
        self.segmentationLogic.startSegmentation.assert_called_once_with([self.node])
