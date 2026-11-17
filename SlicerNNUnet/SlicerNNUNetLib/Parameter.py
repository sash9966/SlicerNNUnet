import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Tuple, Optional, Dict, Union

import qt
from slicer.parameterNodeWrapper import parameterPack, Choice, WithinRange


@lru_cache(maxsize=None)
def _isNvidiaDriverInstalled() -> bool:
    """
    Checks whether the NVIDIA driver, which is required for CUDA, is installed by loading its library.
    Does not require PyTorch.
    """
    import ctypes
    import sys
    libraryNames = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
    for libraryName in libraryNames:
        try:
            ctypes.CDLL(libraryName)
            return True
        except OSError:
            continue
    return False


def availableDevices() -> List[str]:
    """
    Inference devices that can be used on this platform, the preferred one first.
    Determined without PyTorch, as it may not be installed yet when this is called:
    CUDA is not available on macOS, and MPS (Apple GPU) is only available on Apple Silicon.
    On other platforms CUDA is preferred if the NVIDIA driver is installed. CUDA is offered
    even if the driver is not detected, so that it can still be selected manually.
    """
    import platform
    import sys
    if sys.platform == "darwin":
        return ["mps", "cpu"] if platform.machine() == "arm64" else ["cpu"]
    return ["cuda", "cpu"] if _isNvidiaDriverInstalled() else ["cpu", "cuda"]


@parameterPack
@dataclass
class Parameter:
    """
    Parameters storing the NNUNet config in Python formats.
    Sets the default values as used by the Segmentation Logic.
    Parameters are compatible with 3D Slicer parameterNodeWrapper.

    Provides method to convert to nnUNet process arg list.
    """
    folds: str = ""
    device: Annotated[str, Choice(availableDevices())] = availableDevices()[0]
    stepSize: Annotated[float, WithinRange(0., 1.0)] = 0.5
    disableTta: bool = True
    nProcessPreprocessing: Annotated[int, WithinRange(1, 999)] = 1
    nProcessSegmentationExport: Annotated[int, WithinRange(1, 999)] = 1
    checkPointName: str = ""
    modelPath: Path = Path()

    def asDict(self) -> Dict:
        return asdict(self)

    def asJSon(self, indent=None):
        return json.dumps(self.asDict(), cls=_PathEncoder, indent=indent)

    def debugString(self):
        return self.asJSon(indent=4)

    def asArgList(self, inDir: Path, outDir: Path) -> List:
        import sys

        isValid, reason = self.isValid()
        if not isValid:
            raise RuntimeError(f"Invalid nnUNet configuration. {reason}")

        npp = self.nProcessPreprocessing
        nps = self.nProcessSegmentationExport

        # Force sequential on macOS
        if sys.platform == "darwin":
            # nnU-net hangs on macOS at the end of inference (when saving the results)
            # due to multiprocessing. Disable multiprocessing to prevent the hang.
            npp = 0
            nps = 0

        args = [
            "-i", inDir.as_posix(),
            "-o", outDir.as_posix(),
            "-d", self._datasetName,
            "-tr", self._configurationNameParts[0],
            "-p", self._configurationNameParts[1],
            "-c", self._configurationNameParts[-1],
            "-f", *[str(f) for f in self._foldsAsList()],
            "-npp", npp,
            "-nps", nps,
            "-step_size", self.stepSize,
            "-device", self._getDevice(),
            "-chk", self._getCheckpointName()
        ]

        if self.disableTta:
            args.append("--disable_tta")

        return args

    def isSelectedDeviceAvailable(self) -> bool:
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            return False
        if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
        return True

    def _getDevice(self):
        """
        Keep the explicitly selected device. Runtime failures must not change inference settings.
        """
        return self.device

    def toSettings(self, settings: Optional[qt.QSettings] = None, key: str = "") -> None:
        """
        Saves the current Parameters to QSettings.
        If settings is not provided, saves to the default .ini file.
        """
        key = key or self._defaultSettingsKey()
        settings = settings or qt.QSettings()
        settings.setValue(key, self.asJSon())
        settings.sync()

    @classmethod
    def fromSettings(cls, settings: Optional[qt.QSettings] = None, key: str = ""):
        """
        Creates Parameters from the saved settings.
        If settings is not provided, loads from the default .ini file.
        If Parameter is not found or contains partial data, loads available and defaults the rest.
        """
        key = key or cls._defaultSettingsKey()
        settings = settings or qt.QSettings()
        val = settings.value(key, "")
        val_dict = json.loads(val, object_hook=_PathEncoder.decodePath) if val else {}

        instance = cls()
        for k, v in val_dict.items():
            try:
                instance.setValue(k, v)
            except (TypeError, ValueError):
                # ValueError: value is not valid anymore, for example a saved device that is not available on this platform
                continue
        return instance

    @classmethod
    def _defaultSettingsKey(cls):
        return "SlicerNNUNet/Parameter"

    def _getCheckpointName(self):
        return self.checkPointName or "checkpoint_final.pth"

    def isValid(self) -> Tuple[bool, str]:
        """Validate local metadata and every explicitly selected fold/checkpoint, without loading weights."""
        try:
            if not self._isDatasetPathValid():
                return False, ("dataset.json is missing. Select the existing nnU-Net model run folder "
                               "containing dataset.json, plans.json and fold_<number> or fold_all. "
                               f"Provided model directory: {self.modelPath}")
            self._readDatasetMetadata()
            self._configurationNameParts
            if not self._isDatasetNameValid():
                return False, f"Invalid Dataset folder name: {self._datasetName}. Preserve the exported Dataset folder."
            self._foldsAsList()
            checkpoint = self._getCheckpointName()
            if (Path(checkpoint).name != checkpoint or "/" in checkpoint or "\\" in checkpoint
                    or Path(checkpoint).suffix not in (".pth", ".safetensors")):
                return False, "Checkpoint must be a filename ending in .pth or .safetensors inside each selected fold."
            missingFolds = self._getMissingFolds()
            if missingFolds:
                return False, f"Selected folds are missing: {missingFolds}. No other fold will be substituted."
            invalidWeights = self._getFoldsWithInvalidWeights()
            if invalidWeights:
                return False, f"Selected checkpoint {checkpoint} is missing in folds {invalidWeights}. Select the actual filename."
            if not 0 < self.stepSize <= 1:
                return False, "Sliding-window step size must be greater than 0 and at most 1."
            return True, ""
        except (OSError, ValueError, TypeError, KeyError) as error:
            return False, str(error)

    def _readDatasetMetadata(self) -> Dict:
        path = self._datasetFilePath
        if path is None:
            raise ValueError("dataset.json is missing. Select a specific model run folder.")
        dataset = self._readMetadata(path)
        channels = dataset.get("channel_names")
        if isinstance(channels, dict):
            if set(channels) != {str(index) for index in range(len(channels))}:
                raise ValueError("dataset.json channel_names must use consecutive channel IDs starting at 0.")
            channelValues = list(channels.values())
        elif isinstance(channels, list):
            channelValues = channels
        else:
            raise ValueError("dataset.json is missing channel_names. Restore the original model metadata.")
        if not channelValues or any(not isinstance(name, str) or not name for name in channelValues):
            raise ValueError("dataset.json channel_names must contain nonempty modality names.")
        labels = dataset.get("labels")
        if not isinstance(labels, dict) or not labels:
            raise ValueError("dataset.json is missing labels. Restore the original model metadata.")
        if labels.get("background") != 0:
            raise ValueError("dataset.json labels must define background as 0.")
        for name, value in labels.items():
            values = value if isinstance(value, list) else [value]
            if (not isinstance(name, str) or not name or not values
                    or any(type(label) is not int or label < 0 for label in values)):
                raise ValueError("dataset.json labels must contain nonnegative integer label IDs or regions.")
        regions = [value for name, value in labels.items() if name not in ("background", "ignore")]
        if any(isinstance(value, list) for value in regions):
            order = dataset.get("regions_class_order")
            if (not isinstance(order, list) or len(order) != len(regions)
                    or any(type(label) is not int or label <= 0 for label in order)):
                raise ValueError("Region-based labels require the original regions_class_order in dataset.json.")
        ending = dataset.get("file_ending", ".nii.gz")
        if not isinstance(ending, str) or not ending.startswith(".") or "/" in ending or "\\" in ending:
            raise ValueError("dataset.json file_ending must be an image file suffix.")
        return dataset

    @staticmethod
    def _readMetadata(path: Path) -> Dict:
        if not path.is_file():
            raise ValueError(f"{path.name} is missing. Restore the original model metadata in {path.parent}.")
        try:
            with path.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read {path.name}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"{path.name} must contain a JSON object.")
        return metadata

    @staticmethod
    def _isConvertibleToInt(dataSetFolderName):
        try:
            int(dataSetFolderName)
            return True
        except ValueError:
            return False

    def _isDatasetNameValid(self):
        return self._datasetName.startswith("Dataset") or self._isConvertibleToInt(self._datasetName)

    def readSegmentIdsAndLabelsFromDatasetFile(self) -> Optional[List[Tuple[str, str]]]:
        """Return exported label identities, excluding background and the training-only ignore label."""
        try:
            dataset = self._readDatasetMetadata()
        except (OSError, ValueError, TypeError):
            return None
        labels = [(name, value) for name, value in dataset["labels"].items()
                  if name not in ("background", "ignore")]
        if any(isinstance(value, list) for _, value in labels):
            labels = [(name, value) for (name, _), value in zip(labels, dataset["regions_class_order"])]
        return [(f"Segment_{value}", name) for name, value in labels]

    def readFileEndingFromDatasetFile(self) -> str:
        try:
            return self._readDatasetMetadata().get("file_ending", ".nii.gz")
        except (OSError, ValueError, TypeError):
            return ".nii.gz"

    def readChannelNamesFromFile(self) -> List[str]:
        try:
            channels = self._readDatasetMetadata()["channel_names"]
        except (OSError, ValueError, TypeError):
            return []
        return [channels[str(index)] for index in range(len(channels))] if isinstance(channels, dict) else channels

    @property
    def _datasetFilePath(self) -> Optional[Path]:
        path = Path(self.modelPath)
        if path.name.startswith("fold_") and path.parent.joinpath("dataset.json").is_file():
            return path.parent / "dataset.json"
        return self._getFirstFolderWithDatasetFile(path)

    @staticmethod
    def _getFirstFolderWithDatasetFile(path: Path, searchDepth: int = 3) -> Optional[Path]:
        """Keep unambiguous parent-folder convenience without selecting an arbitrary model."""
        direct = path / "dataset.json"
        if direct.is_file():
            return direct
        candidates = []
        pattern = "dataset.json"
        for _ in range(1, searchDepth):
            pattern = f"*/{pattern}"
            candidates.extend(candidate for candidate in path.glob(pattern) if candidate.is_file())
        if len(candidates) > 1:
            raise ValueError("Multiple model directories contain dataset.json. Select the specific trainer/plans/configuration folder.")
        return candidates[0] if candidates else None

    def _foldsAsList(self) -> List[Union[int, str]]:
        value = self.folds.strip()
        if not value:
            return [0]
        if value == "all":
            return ["all"]
        parts = [part.strip() for part in value.split(",")]
        if any(not part or not part.isascii() or not part.isdigit() for part in parts):
            raise ValueError("Folds must be a numbered fold, comma-separated numbered folds, or exactly all. Do not mix all with numbered folds.")
        folds = [int(part) for part in parts]
        if len(set(folds)) != len(folds):
            raise ValueError("Each numbered fold must be selected only once.")
        return folds

    def _getFoldPaths(self) -> List[Tuple[Union[int, str], Path]]:
        return [(fold, self._configurationFolder / f"fold_{fold}") for fold in self._foldsAsList()]

    def _getMissingFolds(self) -> List[Union[int, str]]:
        return [fold for fold, path in self._getFoldPaths() if not path.is_dir()]

    def _getFoldsWithInvalidWeights(self) -> List[Union[int, str]]:
        return [fold for fold, path in self._getFoldPaths() if not (path / self._getCheckpointName()).is_file()]

    @property
    def configurationFolder(self) -> Path:
        return self._configurationFolder

    @property
    def _configurationFolder(self) -> Path:
        path = self._datasetFilePath
        if path is None:
            raise ValueError("dataset.json is missing. Select a specific model run folder.")
        return path.parent

    @property
    def _datasetFolder(self) -> Path:
        return self._configurationFolder.parent

    @property
    def _datasetName(self) -> str:
        return self._datasetFolder.name

    @property
    def modelFolder(self) -> Path:
        return self._datasetFolder.parent

    @property
    def _configurationNameParts(self) -> List[str]:
        """Recover complete identifiers from exact saved plans/configuration suffixes, never split on __."""
        folder = self._configurationFolder
        plans = self._readMetadata(folder / "plans.json")
        plansName = plans.get("plans_name")
        configurations = plans.get("configurations")
        if not isinstance(plansName, str) or not plansName:
            raise ValueError("plans.json is missing plans_name. Restore the original model metadata.")
        if not isinstance(configurations, dict) or not configurations:
            raise ValueError("plans.json is missing configurations. Restore the original model metadata.")
        if plans.get("dataset_name") != self._datasetName:
            raise ValueError("plans.json dataset_name does not match the enclosing Dataset folder. Preserve the original model layout.")
        candidates = []
        for configuration in configurations:
            suffix = f"__{plansName}__{configuration}"
            if folder.name.endswith(suffix):
                trainer = folder.name[:-len(suffix)]
                if trainer:
                    candidates.append([trainer, plansName, configuration])
        if len(candidates) != 1:
            raise ValueError("Model folder does not uniquely match plans.json plans_name and configurations. Select the unchanged exported run folder.")
        return candidates[0]

    @property
    def trainerName(self) -> str:
        return self._configurationNameParts[0]

    @property
    def plansName(self) -> str:
        return self._configurationNameParts[1]

    @property
    def configurationName(self) -> str:
        return self._configurationNameParts[2]

    def _isDatasetPathValid(self) -> bool:
        return self._datasetFilePath is not None


class _PathEncoder(json.JSONEncoder):
    """
    Helper encoder to save / restore modelPath from QSettings.
    """

    def default(self, obj):
        if isinstance(obj, Path):
            return {'_path': str(obj)}
        return super().default(obj)

    @staticmethod
    def decodePath(obj):
        if '_path' in obj:
            return Path(obj['_path'])
        return obj
