import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import qt
import slicer

from SlicerNNUNetLib import SegmentationLogic, Signal
from SlicerNNUNetLib.Parameter import Parameter
from .Utils import NNUNetTestCase, load_test_CT_volume


class MockProcess:
    def __init__(self):
        self.errorOccurred = Signal()
        self.finished = Signal()
        self.readInfo = Signal()
        self.start = MagicMock()
        self.stop = MagicMock()
        self.waitForFinished = MagicMock()


class SegmentationLogicTestCase(NNUNetTestCase):
    def setUp(self):
        super().setUp()
        self._clearScene()
        self._tmp_dir = qt.QTemporaryDir()

        self.mockError = MagicMock()
        self.mockInfo = MagicMock()

        self.process = MockProcess()
        self.logic = SegmentationLogic(self.process)
        self.logic.setParameter(Parameter(modelPath=self.get_tmp_dataset_folder()))
        self.logic.errorOccurred.connect(print)
        self.logic.errorOccurred.connect(self.mockError)
        self.logic.progressInfo.connect(print)
        self.logic.progressInfo.connect(self.mockInfo)
        self.devicePatch = patch.object(Parameter, "isSelectedDeviceAvailable", return_value=True)
        self.devicePatch.start()
        self.addCleanup(self.devicePatch.stop)
        self.predictPatch = patch.object(self.logic, "_findUNetPredictPath", return_value=Path("nnUNetv2_predict"))
        self.predictPatch.start()
        self.addCleanup(self.predictPatch.stop)

        self.volume = load_test_CT_volume()

    def get_tmp_dataset_folder(self) -> Path:
        dataset_folder = Path(self._tmp_dir.path()) / "Dataset111_453CT" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        dataset_folder.mkdir(parents=True, exist_ok=True)
        return dataset_folder

    def create_dataset_file(self, file_ending=None):
        dataset_dict = {
            "channel_names": {
                "0": "channel name"
            },
            "labels": {
                "background": 0,
                "Test Label 1": 1,
                "Test Label 2": 2,
                "Test Label 3": 3,
                "Test Label 4": 4,
                "Test Label 5": 5,
            },
            "numTraining": 42,
            "file_ending": file_ending or ".nii.gz"
        }

        dataset_path = self.get_tmp_dataset_folder() / "dataset.json"

        with open(dataset_path, "w") as f:
            f.write(json.dumps(dataset_dict, indent=4))

        plans_path = self.get_tmp_dataset_folder() / "plans.json"
        plans_path.write_text(json.dumps({
            "plans_name": "nnUNetPlans",
            "dataset_name": "Dataset111_453CT",
            "configurations": {"3d_fullres": {}}
        }))
        return dataset_path

    def create_fake_segmentation(self, file_ending=None):
        import numpy as np
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        volumeId = shNode.GetItemByDataNode(self.volume)
        clonedId = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, volumeId)
        clonedVolume = shNode.GetItemDataNode(clonedId)
        array = slicer.util.arrayFromVolume(clonedVolume)
        array = np.random.randint(0, 6, array.shape, dtype=np.int32)
        slicer.util.updateVolumeFromArray(clonedVolume, array)

        file_ending = file_ending or ".nii.gz"
        slicer.util.exportNode(clonedVolume, self.logic.nnUNetOutDir.joinpath(f"volume{file_ending}"))
        slicer.mrmlScene.RemoveNode(clonedVolume)

    def create_folds_folders(self, folds=(0, 1, 2, 3, 4), chkpt_name="checkpoint_final.pth"):
        for fold in folds:
            fold_path = self.get_tmp_dataset_folder() / f"fold_{fold}"
            fold_path.mkdir()
            chkpt_file = fold_path.joinpath(chkpt_name)
            with open(chkpt_file, "w"):
                pass

    def create_fake_model_dir(self, file_ending=None):
        self.create_dataset_file(file_ending)
        self.create_folds_folders()

    def test_setups_temporary_volume_for_nn_unet_runner(self):
        mockInferenceFinished = MagicMock()

        self.create_fake_model_dir()
        self.logic.inferenceFinished.connect(mockInferenceFinished)
        self.logic.startSegmentation(self.volume)

        self.process.start.assert_called_once()

        self.assertTrue(self.logic.nnUNetInDir.exists())
        self.assertTrue(self.logic.nnUNetOutDir.exists())
        self.assertEqual(len(list(self.logic.nnUNetInDir.rglob("*.nii.gz"))), 1)

        self.process.finished()
        mockInferenceFinished.assert_called_once()

    def test_exported_volume_file_ending_changes_depending_on_dataset_file(self):
        self.create_fake_model_dir(file_ending=".nrrd")
        self.logic.inferenceFinished.connect(MagicMock())
        self.logic.startSegmentation(self.volume)
        self.assertTrue(self.logic.nnUNetInDir.exists())
        self.assertEqual(len(list(self.logic.nnUNetInDir.rglob("*.nrrd"))), 1)
        self.process.finished()

    def test_segmentation_forwards_process_information(self):
        self.process.readInfo("READ")
        self.process.errorOccurred("ERROR")

        self.mockInfo.assert_called_once_with("READ")
        self.mockError.assert_called_once_with("ERROR")

    def test_informs_error_occurred_if_invalid_model_path(self):
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.mockError.reset_mock()

        self.create_folds_folders()
        self.logic.startSegmentation(self.volume)
        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_informs_error_occurred_if_fold_is_outside_created_ones(self):
        self.create_fake_model_dir()
        self.logic.setParameter(Parameter(modelPath=self.get_tmp_dataset_folder(), folds="5"))
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_informs_error_occurred_if_dataset_name_doesnt_start_with_dataset(self):
        self.create_fake_model_dir()
        model_path = os.path.join(self._tmp_dir.path(), "StringWithoutDatasetPrefix")
        os.rename(os.path.join(self._tmp_dir.path(), "Dataset111_453CT"), model_path)

        self.logic.setParameter(Parameter(modelPath=Path(model_path)))
        self.logic.startSegmentation(self.volume)

        self.mockError.assert_called_once()
        self.process.start.assert_not_called()

    def test_loads_segmentation_names_from_data_set(self):
        self.create_fake_model_dir()
        self.logic.startSegmentation(self.volume)
        self.create_fake_segmentation()
        self.process.finished()
        self.logic.loadSegmentation()

        segmentations = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))
        self.assertEqual(len(segmentations), 1)

        segmentation = segmentations[0].GetSegmentation()
        s4 = segmentation.GetNthSegmentID(4)
        self.assertIn("Test Label", segmentation.GetSegment(s4).GetName())

    def test_loads_segmentations_based_on_dataset_file_ending(self):
        self.create_fake_model_dir(file_ending=".nrrd")
        self.logic.startSegmentation(self.volume)
        self.create_fake_segmentation(file_ending=".nrrd")
        self.process.finished()
        self.logic.loadSegmentation()
        self.assertEqual(self.mockError.call_count, 0)

    def test_parameters_can_be_stored_to_and_from_settings(self):
        param = Parameter(
            folds="0,1,2",
            device="cpu",
            stepSize=0.42,
            disableTta=False,
            nProcessSegmentationExport=2,
            nProcessPreprocessing=4,
            checkPointName="custom",
            modelPath=Path("PATH")
        )

        param.toSettings()
        p2 = Parameter.fromSettings()
        self.assertEqual(param, p2)


class SyntheticGeometryTestCase(NNUNetTestCase):
    """Actual Slicer image export/import with synthetic labels; no weights, GPU or downloads.

    These tests verify geometry and editable segmentation handling, not model inference parity.
    """
    def setUp(self):
        super().setUp()
        import numpy as np
        import vtk

        self._tmp_dir = qt.QTemporaryDir()
        self.model = Path(self._tmp_dir.path()) / "Dataset123_Synthetic" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        self.model.mkdir(parents=True)
        (self.model / "dataset.json").write_text(json.dumps({
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "Left ventricle": 1, "Aorta": 7},
            "file_ending": ".nii.gz"
        }))
        (self.model / "plans.json").write_text(json.dumps({
            "plans_name": "nnUNetPlans",
            "dataset_name": "Dataset123_Synthetic",
            "configurations": {"3d_fullres": {}}
        }))
        (self.model / "fold_all").mkdir()
        (self.model / "fold_all" / "checkpoint_final.pth").touch()
        self.parameter = Parameter(modelPath=self.model, folds="all", device="cpu")
        self.process = MockProcess()
        self.logic = SegmentationLogic(self.process)
        self.logic.setParameter(self.parameter)
        self.errors = MagicMock()
        self.finished = MagicMock()
        self.logic.errorOccurred.connect(self.errors)
        self.logic.inferenceFinished.connect(self.finished)
        devicePatch = patch.object(Parameter, "isSelectedDeviceAvailable", return_value=True)
        devicePatch.start()
        self.addCleanup(devicePatch.stop)
        predictPatch = patch.object(self.logic, "_findUNetPredictPath", return_value=Path("nnUNetv2_predict"))
        predictPatch.start()
        self.addCleanup(predictPatch.stop)

        self.volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Synthetic oblique scan")
        image = np.arange(7 * 11 * 13, dtype=np.float32).reshape((7, 11, 13))
        slicer.util.updateVolumeFromArray(self.volume, image)
        theta, tilt = np.deg2rad(31), np.deg2rad(19)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(tilt), np.sin(theta) * np.sin(tilt)],
            [np.sin(theta), np.cos(theta) * np.cos(tilt), -np.cos(theta) * np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)]
        ])
        self.matrix = vtk.vtkMatrix4x4()
        self.matrix.Identity()
        spacing = (0.7, 1.3, 2.6)
        for row in range(3):
            for column in range(3):
                self.matrix.SetElement(row, column, rotation[row, column] * spacing[column])
        for row, origin in enumerate((12.4, -32.8, 7.1)):
            self.matrix.SetElement(row, 3, origin)
        self.volume.SetIJKToRASMatrix(self.matrix)
        self.transform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "Existing scan transform")
        transformMatrix = vtk.vtkMatrix4x4()
        transformMatrix.Identity()
        transformMatrix.SetElement(0, 3, 17.0)
        self.transform.SetMatrixTransformToParent(transformMatrix)
        self.volume.SetAndObserveTransformNodeID(self.transform.GetID())
        self.existing = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Existing segmentation")
        self.existing.GetSegmentation().AddEmptySegment("ExistingSegment", "Keep me")
        self.expectedLabels = np.zeros(image.shape, dtype=np.uint8)
        self.expectedLabels[1:4, 2:6, 3:8] = 1
        self.expectedLabels[4:6, 6:9, 8:11] = 7
        self.originalImage = image.copy()

    def _writeSyntheticPrediction(self):
        """Mimic only native output I/O by replacing pixels in a reloaded exported scan."""
        exported = slicer.util.loadVolume(str(self.logic.nnUNetInDir / "volume_0000.nii.gz"))
        try:
            self._assertGeometry(exported)
            slicer.util.updateVolumeFromArray(exported, self.expectedLabels)
            self.assertTrue(slicer.util.exportNode(exported, str(self.logic.nnUNetOutDir / "volume.nii.gz")))
        finally:
            slicer.mrmlScene.RemoveNode(exported)

    def _assertGeometry(self, node):
        import vtk
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        self.assertEqual(tuple(node.GetImageData().GetDimensions()), (13, 11, 7))
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(matrix.GetElement(row, column), self.matrix.GetElement(row, column), places=5)

    def _assertExistingNodesPreserved(self):
        import numpy as np
        self.assertIs(slicer.mrmlScene.GetNodeByID(self.volume.GetID()), self.volume)
        self.assertIs(slicer.mrmlScene.GetNodeByID(self.existing.GetID()), self.existing)
        self.assertEqual(self.existing.GetSegmentation().GetNumberOfSegments(), 1)
        self.assertEqual(self.existing.GetSegmentation().GetSegment("ExistingSegment").GetName(), "Keep me")
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(self.volume), self.originalImage)
        self._assertGeometry(self.volume)
        self.assertEqual(self.volume.GetTransformNodeID(), self.transform.GetID())

    def test_oblique_anisotropic_roundtrip_preserves_geometry_labels_and_parent_transform(self):
        import numpy as np
        import vtk
        self.logic.startSegmentation(self.volume)
        self.process.start.assert_called_once()
        self._writeSyntheticPrediction()
        self.process.finished()
        self.finished.assert_called_once()
        self.errors.assert_not_called()
        segmentation = self.logic.loadSegmentation()
        self.assertEqual(segmentation.GetTransformNodeID(), self.transform.GetID())
        self.assertEqual(segmentation.GetSegmentation().GetNumberOfSegments(), 2)
        for label, name in ((1, "Left ventricle"), (7, "Aorta")):
            segmentId = f"Segment_{label}"
            segment = segmentation.GetSegmentation().GetSegment(segmentId)
            self.assertIsNotNone(segment, f"Exported label {label} must retain its identity")
            self.assertEqual(segment.GetName(), name)
            restored = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, self.volume)
            np.testing.assert_array_equal(restored, self.expectedLabels == label)
        output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "Roundtrip labels")
        self.assertTrue(slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentation, output, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY))
        # Compare physical world geometry, accounting for the output's parent.
        sourceToWorld = vtk.vtkMatrix4x4()
        self.transform.GetMatrixTransformToWorld(sourceToWorld)
        expectedWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(sourceToWorld, self.matrix, expectedWorld)
        outputMatrix = vtk.vtkMatrix4x4()
        output.GetIJKToRASMatrix(outputMatrix)
        outputToWorld = vtk.vtkMatrix4x4()
        outputToWorld.Identity()
        if output.GetParentTransformNode() is not None:
            output.GetParentTransformNode().GetMatrixTransformToWorld(outputToWorld)
        actualWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(outputToWorld, outputMatrix, actualWorld)
        self.assertEqual(tuple(output.GetImageData().GetDimensions()), (13, 11, 7))
        for row in range(4):
            for column in range(4):
                self.assertAlmostEqual(actualWorld.GetElement(row, column), expectedWorld.GetElement(row, column), places=5)
        self._assertExistingNodesPreserved()

    def test_failed_run_refuses_partial_output_and_preserves_scene(self):
        self.logic.startSegmentation(self.volume)
        self._writeSyntheticPrediction()
        self.process.errorOccurred("Synthetic inference failed")
        self.process.finished()
        self.errors.assert_called_once()
        self.finished.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "Failed or cancelled"):
            self.logic.loadSegmentation()
        self.assertEqual(len(list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))), 1)
        self._assertExistingNodesPreserved()

    def test_cancelled_run_refuses_partial_output_and_preserves_scene(self):
        self.logic.startSegmentation(self.volume)
        self._writeSyntheticPrediction()
        self.logic.stopSegmentation()
        self.process.finished()
        self.finished.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "Failed or cancelled"):
            self.logic.loadSegmentation()
        self.assertEqual(len(list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))), 1)
        self._assertExistingNodesPreserved()

    def test_wrong_channel_count_does_not_start_or_modify_existing_nodes(self):
        self.logic.startSegmentation([self.volume, self.volume])
        self.process.start.assert_not_called()
        self.errors.assert_called_once()
        self.assertIn("requires 1 input channel", str(self.errors.call_args))
        self._assertExistingNodesPreserved()
