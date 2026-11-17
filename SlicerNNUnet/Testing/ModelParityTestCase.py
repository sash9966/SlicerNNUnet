"""Opt-in native-model parity checks. No weights or reference images are bundled.

Run in the target Slicer runtime with pytest -m slow and set, separately for CNN
and VIT, NNFOUNDATION_<name>_MODEL, _INPUT, _REFERENCE and _DEVICE. INPUT is one
image path or a JSON array of channel paths in dataset.json order. Missing
environment values skip that model before loading Slicer, data or weights.

Optional _PARAMETERS points to a JSON object with folds, checkPointName, stepSize
and disableTta, plus optional nProcessPreprocessing/nProcessSegmentationExport.
Without it, the recorded reference settings are all, checkpoint_final.pth,
step 0.5 and disabled TTA. These defaults must match the selected reference;
supplying a reference from a different inference recipe is a test error.

Labels must match exactly. Geometry tolerance is 1e-5 mm per IJK-to-RAS matrix
element, to allow image-header floating-point serialization. No prediction
tolerance or automatic setting changes are permitted.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import unittest

try:
    import pytest
    slow = pytest.mark.slow
except ImportError:
    # Slicer's unittest runner can report missing prerequisites without pytest.
    slow = lambda testClass: testClass


@slow
class ModelParityTestCase(unittest.TestCase):
    def test_cnn_native_inference_matches_reference(self):
        self._checkModel("CNN")

    def test_vit_native_inference_matches_reference(self):
        self._checkModel("VIT")

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _checkModel(self, name):
        prefix = f"NNFOUNDATION_{name}_"
        required = ("MODEL", "INPUT", "REFERENCE", "DEVICE")
        missing = [prefix + key for key in required if not os.environ.get(prefix + key)]
        if missing:
            self.skipTest("Actual model parity unrun; missing " + ", ".join(missing))

        # Keep fast collection and missing-prerequisite checks independent of Slicer.
        import numpy as np
        import slicer
        import vtk
        from SlicerNNUNetLib import Parameter, SegmentationLogic

        model = Path(os.environ[prefix + "MODEL"]).expanduser()
        referencePath = Path(os.environ[prefix + "REFERENCE"]).expanduser()
        inputValue = os.environ[prefix + "INPUT"]
        inputValues = json.loads(inputValue) if inputValue.lstrip().startswith("[") else [inputValue]
        self.assertIsInstance(inputValues, list, "INPUT must be an image path or JSON array of channel paths")
        self.assertTrue(inputValues, "INPUT must contain at least one channel")
        self.assertTrue(all(isinstance(value, str) and value for value in inputValues), "Every channel needs an explicit image path")
        inputPaths = [Path(value).expanduser() for value in inputValues]
        self.assertTrue(model.is_dir(), "MODEL must be an existing unchanged nnU-Net run directory")
        for path in inputPaths + [referencePath]:
            self.assertTrue(path.is_file(), f"Required reference/input file is missing: {path}")

        values = {"folds": "all", "checkPointName": "checkpoint_final.pth", "stepSize": 0.5, "disableTta": True}
        parametersPath = os.environ.get(prefix + "PARAMETERS")
        if parametersPath:
            parametersPath = Path(parametersPath).expanduser()
            self.assertTrue(parametersPath.is_file(), "PARAMETERS JSON file is missing")
            values = json.loads(parametersPath.read_text(encoding="utf-8"))
            self.assertIsInstance(values, dict, "PARAMETERS must contain a JSON object")
            self.assertTrue({"folds", "checkPointName", "stepSize", "disableTta"}.issubset(values),
                            "PARAMETERS must freeze folds, checkPointName, stepSize and disableTta")
            self.assertFalse(set(values) - {"folds", "checkPointName", "stepSize", "disableTta",
                                           "nProcessPreprocessing", "nProcessSegmentationExport"},
                             "MODEL and DEVICE come from their explicit environment values; unknown PARAMETERS keys are forbidden")
        parameter = Parameter(modelPath=model, device=os.environ[prefix + "DEVICE"], **values)
        self.assertTrue(isinstance(parameter.folds, str) and parameter.folds.strip(), "Parity requires explicit folds")
        self.assertTrue(isinstance(parameter.checkPointName, str) and parameter.checkPointName,
                        "Parity requires the exact explicit checkpoint filename")
        valid, reason = parameter.isValid()
        self.assertTrue(valid, reason)
        self.assertEqual(len(inputPaths), len(parameter.readChannelNamesFromFile()), "INPUT channel count must match dataset.json")
        self.assertTrue(parameter.isSelectedDeviceAvailable(), "Explicit DEVICE is unavailable in this runtime")

        sourcePaths = inputPaths + [referencePath, parameter.configurationFolder / "dataset.json",
                                    parameter.configurationFolder / "plans.json"]
        fingerprint = parameter.configurationFolder / "dataset_fingerprint.json"
        if fingerprint.is_file():
            sourcePaths.append(fingerprint)
        if parametersPath:
            sourcePaths.append(parametersPath)
        sourcePaths.extend(parameter.configurationFolder / f"fold_{fold}" / parameter.checkPointName
                           for fold in parameter._foldsAsList())
        originalHashes = {path: self._hash(path) for path in sourcePaths}
        ownedNodes = {}
        logic = SegmentationLogic()  # Real process, export, native inference and import.
        errors, finished = [], []
        logic.errorOccurred.connect(lambda message: errors.append(str(message)))
        logic.inferenceFinished.connect(lambda: finished.append(True))
        logic.progressInfo.connect(print)
        editor = None
        runtimeEnvironment = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_n_proc_DA", "nnUNet_def_n_proc")
        originalEnvironment = {key: os.environ.get(key) for key in runtimeEnvironment}

        def own(node):
            self.assertIsNotNone(node, "Slicer failed to load a required volume or segmentation")
            ownedNodes[node.GetID()] = node
            if hasattr(node, "GetStorageNode") and node.GetStorageNode():
                storage = node.GetStorageNode()
                ownedNodes[storage.GetID()] = storage
            if hasattr(node, "GetNumberOfDisplayNodes"):
                for index in range(node.GetNumberOfDisplayNodes()):
                    display = node.GetNthDisplayNode(index)
                    if display:
                        ownedNodes[display.GetID()] = display
            return node

        def geometry(node):
            matrix = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(matrix)
            return np.array([[matrix.GetElement(row, column) for column in range(4)] for row in range(4)])

        try:
            inputs = [own(slicer.util.loadVolume(str(path))) for path in inputPaths]
            originalArrays = [slicer.util.arrayFromVolume(node).copy() for node in inputs]
            originalGeometry = [geometry(node) for node in inputs]
            reference = own(slicer.util.loadLabelVolume(str(referencePath)))
            referenceArray = slicer.util.arrayFromVolume(reference).copy()
            logic.setParameter(parameter)
            logic.startSegmentation(inputs)
            deadline = time.monotonic() + 3600
            while not errors and not finished and time.monotonic() < deadline:
                logic.inferenceProcess.waitForFinished(1000)
                slicer.app.processEvents()
            slicer.app.processEvents()
            self.assertFalse(errors, "Native inference failed: " + "\n".join(errors))
            self.assertTrue(finished, "Native inference did not complete successfully within one hour")
            self.assertTrue(logic._completedSuccessfully, "Native process did not report successful completion")
            freshPath = Path(logic._outFile)
            self.assertTrue(freshPath.is_file(), "Fresh output volume is missing")
            fresh = own(slicer.util.loadLabelVolume(str(freshPath)))
            freshArray = slicer.util.arrayFromVolume(fresh).copy()
            np.testing.assert_array_equal(freshArray, referenceArray,
                                          err_msg="Prediction labels differ from matched reference; investigate before claiming parity")
            np.testing.assert_allclose(geometry(fresh), geometry(reference), rtol=0, atol=1e-5)
            np.testing.assert_allclose(geometry(fresh), originalGeometry[0], rtol=0, atol=1e-5)

            segmentation = own(logic.loadSegmentation())
            segmentation.CreateBinaryLabelmapRepresentation()
            labelNames = dict(parameter.readSegmentIdsAndLabelsFromDatasetFile())
            for value in np.unique(freshArray):
                if value == 0:
                    continue
                self.assertEqual(float(value), int(value), "Prediction contains a non-integer label")
                segmentId = f"Segment_{int(value)}"
                self.assertIn(segmentId, labelNames, "Prediction contains a label absent from selected model metadata")
                segment = segmentation.GetSegmentation().GetSegment(segmentId)
                self.assertIsNotNone(segment, f"Imported segmentation lost label {value}")
                self.assertEqual(segment.GetName(), labelNames[segmentId])
                importedMask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segmentId, inputs[0])
                np.testing.assert_array_equal(importedMask, freshArray == value)

            # Verify the native result can be assigned to Segment Editor without changing it.
            editorNode = own(slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode", "Model parity editor"))
            editor = slicer.qMRMLSegmentEditorWidget()
            editor.setMRMLScene(slicer.mrmlScene)
            editor.setMRMLSegmentEditorNode(editorNode)
            editor.setSegmentationNode(segmentation)
            editor.setSourceVolumeNode(inputs[0])
            self.assertEqual(editor.segmentationNode(), segmentation)
            for node, original, matrix in zip(inputs, originalArrays, originalGeometry):
                np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), original)
                np.testing.assert_allclose(geometry(node), matrix, rtol=0, atol=0)
        finally:
            logic.stopSegmentation()
            logic.waitForSegmentationFinished()
            if editor is not None:
                editor.setSegmentationNode(None)
                editor.setMRMLSegmentEditorNode(None)
                editor.deleteLater()
            # Remove only nodes created by this test; preserve the existing scene.
            for node in reversed(list(ownedNodes.values())):
                if slicer.mrmlScene.GetNodeByID(node.GetID()) is not None:
                    slicer.mrmlScene.RemoveNode(node)
            for key, value in originalEnvironment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for path, originalHash in originalHashes.items():
                self.assertEqual(self._hash(path), originalHash, f"Source file changed during inference: {path}")
