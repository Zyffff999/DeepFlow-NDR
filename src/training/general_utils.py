from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_training_environment(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    logger = logging.getLogger(__name__)
    logger.info("Using device: %s", device)

    if device_obj.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device_obj


def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
) -> logging.Logger:
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return make_serializable(obj.__dict__)
    return obj


def save_config(
    config: Union[Dict, Any],
    output_dir: str,
    filename: str = "config.json",
) -> str:
    if hasattr(config, "__dict__"):
        config_dict = vars(config)
    else:
        config_dict = config

    os.makedirs(output_dir, exist_ok=True)
    config_dict = make_serializable(config_dict)

    config_path = os.path.join(output_dir, filename)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, sort_keys=True)

    yaml_path = config_path.replace(".json", ".yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(config_dict, handle, default_flow_style=False)

    return config_path


def load_config(config_path: str) -> Dict[str, Any]:
    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    raise ValueError(f"Unsupported config format: {config_path}")


def create_experiment_directory(
    base_dir: str, experiment_name: Optional[str] = None, timestamp: bool = True
) -> str:
    if experiment_name is None:
        experiment_name = "experiment"
    if timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{experiment_name}_{suffix}"

    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    for subdir in ("checkpoints", "logs", "plots", "results"):
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    return experiment_dir


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    hours = seconds / 3600
    return f"{hours:.1f}h"


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return float(param_group["lr"])
    return 0.0


def save_checkpoint_atomic(
    state: Dict[str, Any], filepath: str, is_best: bool = False
) -> None:
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    os.replace(temp_filepath, filepath)
    if is_best:
        best_filepath = filepath.replace(".pth", "_best.pth")
        if best_filepath != filepath:
            torch.save(state, best_filepath)


def load_checkpoint_safe(
    filepath: str,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    return torch.load(filepath, map_location=map_location)


class AverageMeter:
    def __init__(self, name: str = "Meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def __str__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class Timer:
    def __init__(self, name: str = "Timer", verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.verbose:
            print(f"{self.name} took {format_time(self.elapsed)}")
