from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.vae import CorrelatedGaussianVAE
from src.training.general_utils import create_experiment_directory, set_seed, setup_logger
from src.utils.config import ConfigLoader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


class TrainingNPZDataset(Dataset):
    """
    Dataset for a single NPZ file produced by train_data_export.py.
    Expected keys: byte_matrix, global_stats, global_stats_keys (optional).
    """

    def __init__(
        self,
        path: str | Path,
        normalize_bytes: bool = False,
        return_global_stats: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Training data not found: {resolved}")

        data = np.load(resolved, allow_pickle=True)
        if "byte_matrix" not in data:
            raise KeyError("NPZ is missing required key 'byte_matrix'.")

        self.byte_matrix = data["byte_matrix"]
        if return_global_stats and "global_stats" in data:
            self.global_stats = data["global_stats"]
        else:
            self.global_stats = None
        if "global_stats_keys" in data:
            keys = data["global_stats_keys"]
        else:
            keys = np.asarray([], dtype=object)
        self.global_stats_keys = keys.tolist()
        self.normalize_bytes = normalize_bytes
        self.return_global_stats = return_global_stats

        if limit is not None and limit > 0:
            self.byte_matrix = self.byte_matrix[:limit]
            if self.global_stats is not None:
                self.global_stats = self.global_stats[:limit]

    def __len__(self) -> int:
        return int(self.byte_matrix.shape[0])

    def __getitem__(self, idx: int):
        bytes_sample = self.byte_matrix[idx]
        if self.normalize_bytes:
            bytes_tensor = torch.from_numpy(bytes_sample.astype(np.float32)) / 255.0
        else:
            bytes_tensor = torch.from_numpy(bytes_sample.astype(np.uint8))

        if self.return_global_stats:
            if self.global_stats is None:
                stats_tensor = torch.zeros(0, dtype=torch.float32)
            else:
                stats_tensor = torch.from_numpy(self.global_stats[idx].astype(np.float32))
            return bytes_tensor, stats_tensor

        return bytes_tensor


def _default_dataset_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_dataset_config(path: Path, dataset_name: str) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    datasets = payload.get("datasets", {})
    if isinstance(datasets, dict) and datasets:
        if dataset_name not in datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in {path}")
        return datasets[dataset_name]

    if isinstance(payload.get("dataset"), dict):
        return payload["dataset"]

    if "training" in payload or "model" in payload or "system" in payload:
        train_cfg = payload.get("training", {}) if isinstance(payload.get("training", {}), dict) else {}
        model_cfg = payload.get("model", {}) if isinstance(payload.get("model", {}), dict) else {}
        name = (
            payload.get("dataset_name")
            or payload.get("name")
            or model_cfg.get("dataset_name")
            or dataset_name
        )
        return {
            "name": name,
            "train_data_path": train_cfg.get("train_data_path") or payload.get("train_data_path"),
            "model": model_cfg,
            "train": train_cfg,
        }

    return payload

def _kld_weight_for_epoch(epoch: int, train_cfg: Dict[str, object]) -> float:
    if not bool(train_cfg.get("use_kl_annealing", False)):
        return float(train_cfg.get("kld_weight", train_cfg.get("kld_weight_max", 1.0)))
    warmup = int(train_cfg.get("kld_warmup_epochs", 0))
    kld_min = float(train_cfg.get("kld_weight_min", 0.1))
    kld_max = float(train_cfg.get("kld_weight_max", 1.0))
    period = int(train_cfg.get("kl_anneal_period", 10))
    if epoch < warmup:
        return kld_min
    progress = (epoch - warmup) / max(1, period)
    progress = max(0.0, min(1.0, progress))
    return kld_min + (kld_max - kld_min) * progress

def _infer_pin_memory(device: str) -> bool:
    return torch.cuda.is_available() and not device.lower().startswith("cpu")


def _make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: str,
    *,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    pin = _infer_pin_memory(device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
    )


def create_train_val_loaders(
    data_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
    normalize_bytes: bool,
    use_global_stats: bool,
    limit: Optional[int],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    dataset = TrainingNPZDataset(
        data_path,
        normalize_bytes=normalize_bytes,
        return_global_stats=use_global_stats,
        limit=limit,
    )
    total = len(dataset)
    if total < 2:
        raise ValueError(f"Dataset too small (N={total}). Need >= 2 samples.")

    val_size = int(round(total * float(val_split)))
    val_size = max(1, min(val_size, total - 1))
    train_size = total - val_size

    g = torch.Generator()
    g.manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = _make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=True,
        drop_last=True,
    )
    val_loader = _make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


class VAETrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, log_interval: int = 50) -> None:
        self.model = model
        self.device = device
        self.log_interval = max(1, int(log_interval))

    def train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        kld_weight: float = 1.0,
        free_bits: float = 0.0,
        correlation_penalty_weight: float = 0.05,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        recon_loss = 0.0
        kld_loss = 0.0
        count = 0

        for step, batch in enumerate(self._wrap_loader(loader, desc="train")):
            x, global_stats = self._prepare_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            total, recon, kld = self._step_loss(
                x,
                global_stats,
                kld_weight=kld_weight,
                free_bits=free_bits,
                correlation_penalty_weight=correlation_penalty_weight,
            )
            total.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += float(total.item()) * batch_size
            recon_loss += float(recon.item()) * batch_size
            kld_loss += float(kld.item()) * batch_size
            count += batch_size

        return self._finalize_metrics(total_loss, recon_loss, kld_loss, count)

    def evaluate(
        self,
        loader,
        kld_weight: float = 1.0,
        free_bits: float = 0.0,
        correlation_penalty_weight: float = 0.05,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        recon_loss = 0.0
        kld_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self._wrap_loader(loader, desc="val"):
                x, global_stats = self._prepare_batch(batch)
                total, recon, kld = self._step_loss(
                    x,
                    global_stats,
                    kld_weight=kld_weight,
                    free_bits=free_bits,
                    correlation_penalty_weight=correlation_penalty_weight,
                )
                batch_size = x.size(0)
                total_loss += float(total.item()) * batch_size
                recon_loss += float(recon.item()) * batch_size
                kld_loss += float(kld.item()) * batch_size
                count += batch_size

        return self._finalize_metrics(total_loss, recon_loss, kld_loss, count)

    def _prepare_batch(
        self, batch: torch.Tensor | tuple | list
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        global_stats = None
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            x = batch[0]
            if len(batch) > 1:
                global_stats = batch[1]
        else:
            x = batch
        x = x.to(self.device)
        if x.dtype.is_floating_point:
            x = torch.clamp(x * 255.0, 0, 255).round()
        x = x.long()
        if global_stats is not None:
            global_stats = global_stats.to(self.device).float()
            if global_stats.numel() == 0:
                global_stats = None
        return x, global_stats

    def _step_loss(
        self,
        x: torch.Tensor,
        global_stats: Optional[torch.Tensor],
        kld_weight: float,
        free_bits: float,
        correlation_penalty_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, latent_params, recon = self.model(x, global_stats)
        total, recon_loss, kld_loss, _ = self.model.vae_loss_full(
            recon,
            x,
            mu,
            latent_params,
            kld_weight=kld_weight,
            free_bits=free_bits,
            correlation_penalty_weight=correlation_penalty_weight,
        )
        return total, recon_loss, kld_loss

    def _finalize_metrics(
        self, total_loss: float, recon_loss: float, kld_loss: float, count: int
    ) -> Dict[str, float]:
        denom = max(1, count)
        return {
            "total_loss": total_loss / denom,
            "recon_loss": recon_loss / denom,
            "kld_loss": kld_loss / denom,
        }

    def _wrap_loader(self, loader, desc: str):
        if tqdm is None:
            return loader
        return tqdm(loader, desc=desc, leave=False)

    def _update_progress(self, **metrics: float) -> None:
        if tqdm is None:
            return
        try:
            tqdm.write(
                " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            )
        except Exception:
            return

    def compute_reference_statistics(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Robust statistic estimation (OAS Shrinkage) for Mahalanobis distance.
        """
        self.model.eval()
        all_mu = []

        with torch.no_grad():
            for batch in dataloader:
                x, global_stats = self._prepare_batch(batch)
                _, mu = self.model.get_features(x, global_stats)
                all_mu.append(mu.detach().cpu())

        if not all_mu:
            raise ValueError("No samples provided for reference statistics.")

        all_mu = torch.cat(all_mu, dim=0).double()
        n_samples, dims = all_mu.shape

        mean = all_mu.mean(dim=0)
        centered = all_mu - mean

        denom = max(1, n_samples - 1)
        sample_cov = (centered.T @ centered) / denom
        trace = torch.trace(sample_cov)
        trace2 = torch.trace(sample_cov @ sample_cov)

        mu_trace = trace / dims
        alpha = mu_trace
        num = (1.0 - 2.0 / dims) * trace2 + trace**2
        den = (n_samples + 1.0 - 2.0 / dims) * (trace2 - trace**2 / dims)
        den_value = float(den)
        if den_value == 0.0:
            shrinkage = torch.tensor(1.0, dtype=sample_cov.dtype, device=sample_cov.device)
        else:
            shrinkage = torch.clamp(num / den, 0.0, 1.0)

        eye = torch.eye(dims, dtype=sample_cov.dtype, device=sample_cov.device)
        shrunk_cov = (1.0 - shrinkage) * sample_cov + shrinkage * alpha * eye

        jitter = 1e-6 * eye
        try:
            chol = torch.linalg.cholesky(shrunk_cov + jitter)
            chol_inv = torch.linalg.solve_triangular(chol, eye, upper=False)
            cov_inv = chol_inv.T @ chol_inv

            u, s, vt = torch.linalg.svd(shrunk_cov)
            s = torch.clamp(s, min=1e-12)
            cov_invsqrt = u @ torch.diag(s.rsqrt()) @ vt
        except Exception:
            cov_inv = torch.linalg.pinv(shrunk_cov)
            cov_invsqrt = torch.linalg.pinv(shrunk_cov).sqrt()

        return {
            "reference_mu": mean.float(),
            "reference_cov": shrunk_cov.float(),
            "reference_cov_inv": cov_inv.float(),
            "reference_cov_invsqrt": cov_invsqrt.float(),
        }


def _default_output_dir() -> str:
    config = ConfigLoader.get_instance()
    return str(config.system_config.get("checkpoints_dir", "data/checkpoints"))


def _default_reference_stats_path(dataset_name: str | None) -> Path:
    config = ConfigLoader.get_instance()
    base = Path(config.system_config.get("stats_dir", "data/stats"))
    if dataset_name:
        filename = f"reference_stats_{dataset_name}.pt"
    else:
        filename = "reference_stats.pt"
    return base / filename


def build_model_and_optim(
    device: torch.device,
    model_cfg: Dict[str, object],
    train_cfg: Dict[str, object],
    num_packets: int,
    packet_len: int,
    global_dim: int = 0,
    global_fuse: str = "concat",
) -> Tuple[torch.nn.Module, optim.Optimizer]:
    latent_dim = int(model_cfg.get("latent_dim", 64))
    training_mode = str(model_cfg.get("training_mode", "correlated"))
    n_components = int(model_cfg.get("n_components", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.2))

    encoder_cfg = {
        "vocab_size": int(model_cfg.get("vocab_size", 257)),
        "emb_dim": int(model_cfg.get("emb_dim", 128)),
        "packet_latent_dim": int(model_cfg.get("packet_latent_dim", 128)),
        "d_model": int(model_cfg.get("session_d_model", 128)),
        "latent_dim": latent_dim,
        "spatial_reduction_size": int(model_cfg.get("spatial_reduction_size", 8)),
        "header_len": model_cfg.get("header_len"),
        "header_tcn_channels": model_cfg.get("header_tcn_channels"),
        "payload_tcn_channels": model_cfg.get("payload_tcn_channels"),
        "session_tcn_channels": model_cfg.get("session_tcn_channels"),
        "header_only": bool(model_cfg.get("header_only", False)),
        "use_dilation": bool(model_cfg.get("use_dilation", True)),
        "depthwise_only": bool(model_cfg.get("depthwise_only", False)),
    }
    decoder_cfg = {
        "latent_dim": latent_dim,
        "output_size": num_packets * packet_len,
        "output_shape": (num_packets, packet_len),
        "rank": int(model_cfg.get("decoder_rank", hidden_dim)),
        "num_deep_layers": int(model_cfg.get("decoder_num_layers", 1)),
    }

    model = CorrelatedGaussianVAE(
        latent_dim=latent_dim,
        training_mode=training_mode,
        n_components=n_components,
        hidden_dim=hidden_dim,
        dropout=dropout_rate,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        global_dim=global_dim,
        global_mlp_hidden=model_cfg.get("global_mlp_hidden"),
        global_mlp_out=model_cfg.get("global_mlp_out"),
        global_fuse=global_fuse,
    ).to(device)

    lr = float(train_cfg.get("lr", train_cfg.get("learning_rate", 1e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return model, optimizer


def _infer_data_shape(loader) -> Optional[Tuple[int, int]]:
    dataset = loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "byte_matrix"):
        shape = getattr(dataset, "byte_matrix").shape
        if len(shape) >= 3:
            return int(shape[1]), int(shape[2])
    return None


def _infer_global_dim(loader) -> int:
    dataset = loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "global_stats") and getattr(dataset, "global_stats") is not None:
        global_stats = getattr(dataset, "global_stats")
        if hasattr(global_stats, "shape") and len(global_stats.shape) >= 2:
            return int(global_stats.shape[1])
    return 0


def train(args: argparse.Namespace) -> Tuple[VAETrainer, str]:
    dataset_cfg = _load_dataset_config(Path(args.dataset_config), args.dataset)
    model_cfg = dataset_cfg.get("model", {}) if isinstance(dataset_cfg.get("model", {}), dict) else {}
    train_cfg = dataset_cfg.get("train", {}) if isinstance(dataset_cfg.get("train", {}), dict) else {}
    dataset_name = str(
        dataset_cfg.get("name")
        or dataset_cfg.get("dataset_name")
        or model_cfg.get("dataset_name")
        or args.dataset
    )

    train_data = dataset_cfg.get("train_data_path")
    if not train_data:
        raise ValueError("train_data_path is required in the dataset config.")

    device_setting = str(train_cfg.get("device") or "").lower()
    if device_setting in ("", "auto", "none"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = device_setting
    device = torch.device(device_name)
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(train_cfg.get("output_dir") or _default_output_dir())
    out_dir = create_experiment_directory(output_dir, exp_name, timestamp=False)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    setup_logger(log_file=os.path.join(out_dir, "logs", "training.log"), log_level="INFO")
    logging.info("Experiment dir: %s", out_dir)
    logging.info("Dataset: %s", dataset_name)

    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 0))
    val_split = float(train_cfg.get("val_split", 0.1))
    normalize_bytes = bool(train_cfg.get("normalize_bytes", False))
    use_global_stats = bool(train_cfg.get("use_global_stats", False))
    limit = train_cfg.get("limit", None)
    log_interval = int(train_cfg.get("log_interval", 50))
    save_interval = int(train_cfg.get("save_interval", 5))
    global_fuse = str(model_cfg.get("global_fuse", "concat")).lower()

    train_loader, val_loader = create_train_val_loaders(
        data_path=str(train_data),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
        normalize_bytes=normalize_bytes,
        use_global_stats=use_global_stats,
        limit=limit,
        val_split=val_split,
        seed=seed,
    )
    shape = _infer_data_shape(train_loader)
    if shape is None:
        raise ValueError("Failed to infer training data shape.")
    num_packets = int(model_cfg.get("num_packets", shape[0]))
    packet_len = int(model_cfg.get("packet_len", shape[1]))

    global_dim = 0
    if use_global_stats:
        global_dim = _infer_global_dim(train_loader)
        if global_dim <= 0:
            raise ValueError("use_global_stats is true but global_stats is missing in training data.")
        logging.info("use_global_stats enabled (global_dim=%s, fuse=%s).", global_dim, global_fuse)

    model, optimizer = build_model_and_optim(
        device=device,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        num_packets=num_packets,
        packet_len=packet_len,
        global_dim=global_dim,
        global_fuse=global_fuse,
    )
    trainer = VAETrainer(model, device, log_interval=log_interval)
    scheduler = None
    if bool(train_cfg.get("use_scheduler", False)):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(train_cfg.get("scheduler_patience", 2)),
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )

    best_val = float("inf")
    early_stop_wait = 0
    num_epochs = int(train_cfg.get("num_epochs", 60))
    free_bits = float(train_cfg.get("free_bits", 0.0))
    corr_penalty = float(train_cfg.get("correlation_penalty_weight", 0.05))
    epoch_iter = range(num_epochs)
    epoch_bar = tqdm(epoch_iter, desc="epoch", leave=True) if tqdm is not None else None
    if epoch_bar is not None:
        epoch_iter = epoch_bar
    for epoch in epoch_iter:
        kld_weight = _kld_weight_for_epoch(epoch, train_cfg)
        train_m = trainer.train_epoch(
            train_loader,
            optimizer,
            kld_weight=kld_weight,
            free_bits=free_bits,
            correlation_penalty_weight=corr_penalty,
        )
        val_m = trainer.evaluate(
            val_loader,
            kld_weight=kld_weight,
            free_bits=free_bits,
            correlation_penalty_weight=corr_penalty,
        )

        val_loss = float(val_m["total_loss"])
        if scheduler is not None:
            scheduler.step(val_loss)

        metrics = {
            "epoch": epoch,
            "train_total": train_m["total_loss"],
            "train_recon": train_m["recon_loss"],
            "train_kld": train_m["kld_loss"],
            "val_total": val_m["total_loss"],
            "val_recon": val_m["recon_loss"],
            "val_kld": val_m["kld_loss"],
            "kld_weight": kld_weight,
        }
        logging.info(
            "epoch=%s train_total=%.6f train_recon=%.6f train_kl=%.6f "
            "val_total=%.6f val_recon=%.6f val_kl=%.6f",
            epoch,
            metrics["train_total"],
            metrics["train_recon"],
            metrics["train_kld"],
            metrics["val_total"],
            metrics["val_recon"],
            metrics["val_kld"],
        )

        min_delta = (
            float(train_cfg.get("early_stopping_min_delta", 1e-6))
            if bool(train_cfg.get("use_early_stopping", False))
            else 0.0
        )
        if metrics["val_total"] < best_val - min_delta:
            best_val = metrics["val_total"]
            early_stop_wait = 0
            ckpt_path = os.path.join(out_dir, "checkpoints", "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )
        else:
            early_stop_wait += 1

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )

        if bool(train_cfg.get("use_early_stopping", False)) and early_stop_wait >= int(
            train_cfg.get("early_stopping_patience", 5)
        ):
            logging.info("Early stopping triggered at epoch %s.", epoch)
            break

    if bool(train_cfg.get("compute_reference_stats", True)):
        best_ckpt = os.path.join(out_dir, "checkpoints", "best.pth")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        full_dataset = TrainingNPZDataset(
            str(train_data),
            normalize_bytes=normalize_bytes,
            return_global_stats=use_global_stats,
            limit=limit,
        )
        full_loader = _make_loader(
            full_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device_name,
            shuffle=False,
            drop_last=False,
        )
        stats = trainer.compute_reference_statistics(full_loader)
        stats_path = Path(
            train_cfg.get("reference_stats_path")
            or str(_default_reference_stats_path(dataset_name))
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, stats_path)
        logging.info("Reference stats saved to %s", stats_path)

    logging.info("Training done. Best val loss=%.6f", best_val)
    return trainer, out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CorrelatedGaussianVAE from NPZ training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset-config", type=str, default=str(_default_dataset_config_path()))
    p.add_argument("--dataset", type=str, default="cicids2017")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
