from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """
    Singleton helper that loads the unified YAML configuration once and provides
    centralized accessors for system/model/training settings.
    """

    _instance: "ConfigLoader | None" = None
    _lock = threading.Lock()

    def __init__(self, config_dir: Path | None = None) -> None:
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[2] / "configs"
        self._config_dir = config_dir
        self._config_path = self._config_dir / "config.yaml"
        self._raw_config: Dict[str, Any] = {}
        self._model_config: Dict[str, Any]
        self._system_config: Dict[str, Any]
        self._training_config: Dict[str, Any]
        self._load_configs()

    @classmethod
    def get_instance(cls) -> "ConfigLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reload(self) -> None:
        """Reloads the unified configuration file from disk."""
        self._load_configs()

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @property
    def system_config(self) -> Dict[str, Any]:
        return self._system_config

    @property
    def training_config(self) -> Dict[str, Any]:
        return self._training_config

    def _read_yaml(self, filename: str) -> Dict[str, Any]:
        config_path = self._config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}

    def _load_configs(self) -> None:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Unified config not found: {self._config_path}")
        self._raw_config = self._read_yaml("config.yaml")
        self._model_config = dict(self._raw_config.get("model", {}) or {})
        self._system_config = dict(self._raw_config.get("system", {}) or {})
        self._training_config = dict(self._raw_config.get("training", {}) or {})


__all__ = ["ConfigLoader"]
