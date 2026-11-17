"""Saved UI selections pointing to ordinary, unchanged nnU-Net model folders."""

import json


class ModelRegistry:
    names = ("Manual", "CNN", "ViT")
    settingsKey = "SlicerNNUNet/RegisteredModels"

    def __init__(self, settings=None, parameterType=None):
        if settings is None:
            import qt
            settings = qt.QSettings()
        if parameterType is None:
            from .Parameter import Parameter
            parameterType = Parameter
        self.settings = settings
        self.parameterType = parameterType
        self.selected = "Manual"
        self.warnings = []
        value = settings.value(self.settingsKey, "")
        if value:
            try:
                index = json.loads(value)
                if not isinstance(index, dict) or index.get("selected") not in self.names:
                    raise ValueError("invalid selected model")
                self.selected = index["selected"]
            except (TypeError, ValueError):
                self.warnings.append("Saved model selection is invalid. Using Manual settings.")

    def _parameterKey(self, name):
        if name not in self.names:
            raise ValueError(f"Unknown model selection: {name}")
        return f"{self.settingsKey}/{name}"

    def load(self, name):
        key = self._parameterKey(name)
        value = self.settings.value(key, "")
        if not value:
            return None
        try:
            if not isinstance(json.loads(value), dict):
                raise ValueError("expected parameter object")
            return self.parameterType.fromSettings(self.settings, key)
        except (TypeError, ValueError, AttributeError, KeyError) as error:
            self.warnings.append(f"Saved {name} settings could not be read: {error}. Register the model again.")
            return None

    def save(self, name, parameter):
        """Only an explicit Save/Update action writes a registered model's settings."""
        parameter.toSettings(self.settings, self._parameterKey(name))

    def select(self, name):
        self._parameterKey(name)
        self.selected = name
        self.settings.setValue(self.settingsKey, json.dumps({"selected": name}))
        self.settings.sync()
