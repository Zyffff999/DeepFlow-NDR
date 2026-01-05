from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.extractor import ExtractedFlow, iter_pcap
from src.pipeline.tensorizer import TensorizedFlow, tensorize_flow
from src.services.db_manager import (
    DBManager,
    FlowRecordCreate,
    TrainingFlowCreate,
)
from src.utils.config import ConfigLoader

try:
    from src.core.vae import CorrelatedGaussianVAE
except ImportError:  # pragma: no cover - waiting for user-provided model
    CorrelatedGaussianVAE = None  # type: ignore


# === Data holders ===
@dataclass
class InferenceResult:
    flow: ExtractedFlow
    recon_loss: float
    z_score: float
    vae_score: Optional[float]
    score_type: str
    is_anomaly: bool
    explanation: Dict[str, List[int]]


@dataclass
class BenignStats:
    mean: float
    std: float

    @property
    def safe_std(self) -> float:
        return self.std if self.std > 0 else 1.0


@dataclass
class ReferenceStats:
    mean: torch.Tensor
    cov_inv: torch.Tensor
    cov: Optional[torch.Tensor] = None
    cov_invsqrt: Optional[torch.Tensor] = None


# === Service ===
class InferenceService:
    """
    Modes:
      - training: ingest PCAP -> NFStream -> store raw features in DB (no model)
      - inference: PCAP -> NFStream -> tensorize -> model -> score; optionally store raw features for RCA
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        db_url: str | None = None,
        benign_stats_path: str | Path | None = None,
        reference_stats_path: str | Path | None = None,
        device: str | None = None,
        mode: str = "inference",
        store_results: bool | None = None,
        store_anomalies_only: bool = True,
        score_mode: str | None = None,
        latent_input_mode: str | None = None,
        use_db: bool | None = None,
    ) -> None:
        config = ConfigLoader.get_instance()
        self.model_config = config.model_config
        system_config = config.system_config

        if mode not in ("inference", "training"):
            raise ValueError("mode must be 'inference' or 'training'.")
        self.mode = mode

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bytes_pkts = int(
            self.model_config.get("bytes_pkts", self.model_config.get("num_packets", 16))
        )
        self.bytes_len = int(
            self.model_config.get("bytes_len", self.model_config.get("packet_len", 256))
        )
        self.seq_len = int(self.model_config.get("seq_len", 50))
        self.use_global_stats = bool(self.model_config.get("use_global_stats", False))
        self.global_stats_keys = list(self.model_config.get("global_stats_keys", []) or [])
        self.global_dim = len(self.global_stats_keys) if self.use_global_stats else 0

        if use_db is None:
            use_db = bool(system_config.get("inference_use_db", False)) if mode == "inference" else True
        self.use_db = bool(use_db)
        self.db_manager: DBManager | None = None
        if self.use_db:
            self.db_manager = DBManager(db_url or system_config["db_url"])
        self.db_batch_size = int(system_config.get("db_batch_size", 1000))
        self.training_batch_size = int(system_config.get("training_batch_size", 1000))
        self.ingest_progress_every = int(system_config.get("ingest_progress_every", 1000))
        self.infer_progress_every = int(system_config.get("inference_progress_every", 1000))
        self.nfstream_idle_timeout = system_config.get("nfstream_idle_timeout")
        self.nfstream_active_timeout = system_config.get("nfstream_active_timeout")
        self.infer_filter_enabled = bool(system_config.get("inference_filter_enabled", True))
        self.infer_filter_exclude_dns = bool(system_config.get("inference_filter_exclude_dns", True))
        self.infer_filter_min_total_bytes = int(system_config.get("inference_filter_min_total_bytes", 100))
        self.infer_filter_min_duration_ms = float(system_config.get("inference_filter_min_duration_ms", 0.0))
        self.infer_filter_require_bidirectional = bool(
            system_config.get("inference_filter_require_bidirectional", True)
        )
        default_block_ports = [137, 138, 139, 1900, 5353, 67, 68]
        self.infer_filter_block_ports = {
            int(p) for p in system_config.get("inference_filter_block_ports", default_block_ports)
        }

        if store_results is None:
            self.store_results = bool(system_config.get("inference_store_results", False))
        else:
            self.store_results = store_results
        self.store_anomalies_only = store_anomalies_only
        if not self.use_db:
            self.store_results = False

        self.recon_threshold = float(self.model_config.get("threshold_zscore", 3.0))
        self.latent_threshold = float(
            system_config.get(
                "threshold_mahalanobis",
                self.model_config.get("threshold_mahalanobis", 6.0),
            )
        )
        configured_score_mode = (
            score_mode
            or system_config.get("anomaly_score_mode")
            or self.model_config.get("anomaly_score_mode")
        )
        self.anomaly_score_mode = (
            str(configured_score_mode).lower() if configured_score_mode is not None else None
        )
        self.latent_input_mode = str(
            latent_input_mode
            or system_config.get("latent_input_mode")
            or self.model_config.get("latent_input_mode", "raw")
        ).lower()
        if self.anomaly_score_mode is not None and self.anomaly_score_mode not in {
            "recon",
            "reconstruction",
            "vae",
            "latent",
            "mahalanobis",
            "both",
            "and",
            "either",
            "or",
        }:
            raise ValueError(f"Unsupported anomaly_score_mode: {self.anomaly_score_mode}")
        if self.latent_input_mode not in {"raw", "normalized"}:
            raise ValueError(f"Unsupported latent_input_mode: {self.latent_input_mode}")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.stats = BenignStats(mean=0.0, std=1.0)
        self.reference_stats: Optional[ReferenceStats] = None
        self.model = None

        if self.mode == "inference":
            if self.checkpoint_path is None:
                raise ValueError("checkpoint_path is required for inference mode.")
            if not self.use_global_stats:
                raise ValueError("Fixed inference mode requires use_global_stats = true.")
            if self.use_global_stats and self.global_dim == 0:
                raise ValueError(
                    "use_global_stats is enabled but global_stats_keys is empty in config."
                )
            self.model = self._load_model()
            self.stats = self._load_benign_stats(benign_stats_path)
            self.reference_stats = self._load_reference_stats(reference_stats_path)
            if self.anomaly_score_mode is None:
                self.anomaly_score_mode = "vae" if self.reference_stats else "recon"

    # === Public APIs ===
    def analyze_pcap(self, pcap_path: str | Path) -> Dict[str, int]:
        self._ensure_mode("inference")
        return self._process_pcap(pcap_path, collect_results=False)

    def analyze_pcap_verbose(self, pcap_path: str | Path) -> List[InferenceResult]:
        self._ensure_mode("inference")
        return self._process_pcap(pcap_path, collect_results=True)

    def iter_pcap_results(self, pcap_path: str | Path) -> Iterable[InferenceResult]:
        self._ensure_mode("inference")
        flows = iter_pcap(
            pcap_path,
            bytes_pkts=self.bytes_pkts,
            bytes_len=self.bytes_len,
            seq_len=self.seq_len,
            idle_timeout=self.nfstream_idle_timeout,
            active_timeout=self.nfstream_active_timeout,
        )
        batch_records: List[FlowRecordCreate] = []
        flow_count = 0
        anomalies = 0
        build_record = self.store_results

        for flow in flows:
            if self.use_global_stats and not self.global_stats_keys:
                self.global_stats_keys = sorted(flow.global_stats.keys())

            if not self._passes_inference_filters(flow):
                continue

            result, record = self._handle_flow(flow, build_record=build_record)
            flow_count += 1
            if result.is_anomaly:
                anomalies += 1

            should_store = self.store_results and (result.is_anomaly or not self.store_anomalies_only)
            if should_store and record is not None:
                batch_records.append(record)
                if len(batch_records) >= self.db_batch_size:
                    self._flush_inference_batch(batch_records)

            if self.infer_progress_every > 0 and flow_count % self.infer_progress_every == 0:
                print(
                    f"[infer] processed={flow_count} anomalies={anomalies}",
                    flush=True,
                )
            yield result

        if batch_records:
            self._flush_inference_batch(batch_records)

    def ingest_training_pcap(self, pcap_path: str | Path) -> Dict[str, int]:
        self._ensure_mode("training")
        return self._store_training_flows(pcap_path)

    def iter_training_tensors(
        self, batch_size: int | None = None, limit: int | None = None
    ) -> Iterable[List[TensorizedFlow]]:
        """
        Streams tensorized training samples from DB for model training.
        """
        self._ensure_mode("training")
        if self.db_manager is None:
            raise RuntimeError("DB is disabled; cannot stream training tensors.")
        target_batch = batch_size or self.training_batch_size
        batch: List[TensorizedFlow] = []
        count = 0
        for row in self.db_manager.iter_training_flows(batch_size=target_batch):
            flow = ExtractedFlow(
                byte_matrix=np.asarray(row.raw_bytes, dtype=np.uint8),
                iat_series=list(row.iat_series or []),
                size_series=list(row.size_series or []),
                global_stats=dict(row.global_stats or {}),
                src_ip=row.src_ip,
                dst_ip=row.dst_ip,
                dst_port=row.dst_port,
                protocol=row.protocol,
                timestamp=row.timestamp,
            )
            if self.use_global_stats and not self.global_stats_keys:
                self.global_stats_keys = sorted(flow.global_stats.keys())
            batch.append(
                tensorize_flow(
                    flow,
                    seq_len=self.seq_len,
                    global_stats_keys=self.global_stats_keys,
                    normalize_bytes=True,
                )
            )
            count += 1
            if limit is not None and count >= limit:
                if batch:
                    yield batch
                return
            if len(batch) >= target_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    # === Inference path ===
    def _process_pcap(
        self, pcap_path: str | Path, collect_results: bool
    ) -> List[InferenceResult] | Dict[str, int]:
        flows = iter_pcap(
            pcap_path,
            bytes_pkts=self.bytes_pkts,
            bytes_len=self.bytes_len,
            seq_len=self.seq_len,
            idle_timeout=self.nfstream_idle_timeout,
            active_timeout=self.nfstream_active_timeout,
        )
        results: List[InferenceResult] = []
        batch_records: List[FlowRecordCreate] = []
        flow_count = 0
        anomalies = 0

        for flow in flows:
            if self.use_global_stats and not self.global_stats_keys:
                self.global_stats_keys = sorted(flow.global_stats.keys())

            if not self._passes_inference_filters(flow):
                continue

            result, record = self._handle_flow(flow, build_record=self.store_results)
            flow_count += 1
            if result.is_anomaly:
                anomalies += 1
            if collect_results:
                results.append(result)

            should_store = self.store_results and (result.is_anomaly or not self.store_anomalies_only)
            if should_store and record is not None:
                batch_records.append(record)
                if len(batch_records) >= self.db_batch_size:
                    self._flush_inference_batch(batch_records)
            if self.infer_progress_every > 0 and flow_count % self.infer_progress_every == 0:
                print(
                    f"[infer] processed={flow_count} anomalies={anomalies}",
                    flush=True,
                )

        if batch_records:
            self._flush_inference_batch(batch_records)

        if collect_results:
            return results
        return {"flow_count": flow_count, "anomalies": anomalies}

    def _handle_flow(
        self, flow: ExtractedFlow, build_record: bool = True
    ) -> tuple[InferenceResult, FlowRecordCreate | None]:
        recon_loss, explanation = self._predict(flow)
        z_score = (recon_loss - self.stats.mean) / self.stats.safe_std
        vae_score = self._compute_latent_score(flow)
        is_anomaly, score_type = self._decide_anomaly(z_score, vae_score)

        record = None
        if build_record:
            record = FlowRecordCreate(
                timestamp=flow.timestamp.replace(tzinfo=None),
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                recon_loss=recon_loss,
                z_score=z_score,
                is_anomaly=is_anomaly,
                raw_bytes_preview=flow.byte_matrix.astype(int).tolist(),
                iat_series=list(flow.iat_series),
                size_series=list(flow.size_series),
                global_stats=dict(flow.global_stats),
            )

        result = InferenceResult(
            flow=flow,
            recon_loss=recon_loss,
            z_score=z_score,
            vae_score=vae_score,
            score_type=score_type,
            is_anomaly=is_anomaly,
            explanation=explanation,
        )
        return result, record

    def _normalize_protocol(self, value: object) -> str:
        text = str(value or "").strip().lower()
        if text in ("6", "tcp"):
            return "tcp"
        if text in ("17", "udp"):
            return "udp"
        return text

    def _passes_inference_filters(self, flow: ExtractedFlow) -> bool:
        if not self.infer_filter_enabled:
            return True

        protocol = self._normalize_protocol(flow.protocol)
        if protocol not in {"tcp", "udp"}:
            return False

        dst_port = int(flow.dst_port or 0)
        if dst_port in self.infer_filter_block_ports:
            return False
        if self.infer_filter_exclude_dns and protocol == "udp" and dst_port == 53:
            return False

        stats = flow.global_stats or {}
        total_bytes = int(stats.get("total_bytes", 0) or 0)
        if total_bytes < self.infer_filter_min_total_bytes:
            return False

        duration_ms = float(stats.get("duration_ms", 0.0) or 0.0)
        if duration_ms <= self.infer_filter_min_duration_ms:
            return False

        if self.infer_filter_require_bidirectional:
            src_bytes = int(stats.get("src2dst_bytes", 0) or 0)
            dst_bytes = int(stats.get("dst2src_bytes", 0) or 0)
            if src_bytes <= 0 or dst_bytes <= 0:
                return False

        return True

    def _predict(self, flow: ExtractedFlow) -> tuple[float, Dict[str, List[int]]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized for inference.")
        tensorized = tensorize_flow(
            flow,
            seq_len=self.seq_len,
            global_stats_keys=self.global_stats_keys,
            normalize_bytes=False,
        )
        byte_tensor = torch.tensor(
            tensorized.byte_matrix, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        global_tensor = None
        if self.use_global_stats:
            global_tensor = torch.tensor(
                tensorized.global_stats, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        with torch.no_grad():
            output = self._forward_model(byte_tensor, global_tensor)
            reconstruction = self._extract_reconstruction(output)
            target = byte_tensor / 255.0
            loss = float(F.mse_loss(reconstruction, target, reduction="mean").item())
        explanation = self._build_explanation(target, reconstruction)
        return loss, explanation

    def _compute_latent_score(self, flow: ExtractedFlow) -> Optional[float]:
        if self.model is None or self.reference_stats is None:
            return None
        byte_tensor, global_tensor = self._prepare_latent_inputs(flow)
        with torch.no_grad():
            if hasattr(self.model, "get_features"):
                _, mu = self.model.get_features(byte_tensor, global_tensor)
            else:
                output = self._forward_model(byte_tensor, global_tensor)
                if isinstance(output, (tuple, list)) and len(output) > 1:
                    mu = output[1]
                elif isinstance(output, dict) and "mu" in output:
                    mu = output["mu"]
                else:
                    return None
        return float(self._mahalanobis(mu, self.reference_stats))

    # === Training ingest path ===
    def _store_training_flows(self, pcap_path: str | Path) -> Dict[str, int]:
        flows = iter_pcap(
            pcap_path,
            bytes_pkts=self.bytes_pkts,
            bytes_len=self.bytes_len,
            seq_len=self.seq_len,
            idle_timeout=self.nfstream_idle_timeout,
            active_timeout=self.nfstream_active_timeout,
        )
        if self.db_manager is None:
            raise RuntimeError("DB is disabled; cannot store training flows.")
        batch: List[TrainingFlowCreate] = []
        flow_count = 0
        inserted = 0
        progress_every = max(0, int(self.ingest_progress_every or 0))

        for flow in flows:
            record = TrainingFlowCreate(
                timestamp=flow.timestamp.replace(tzinfo=None),
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
                dst_port=flow.dst_port,
                protocol=flow.protocol,
                raw_bytes=flow.byte_matrix.astype(int).tolist(),
                iat_series=list(flow.iat_series),
                size_series=list(flow.size_series),
                global_stats=dict(flow.global_stats),
            )
            batch.append(record)
            flow_count += 1
            if len(batch) >= self.db_batch_size:
                self.db_manager.bulk_insert_training(batch)
                inserted += len(batch)
                batch.clear()
                if progress_every and inserted % progress_every == 0:
                    print(f"[ingest] inserted={inserted}", flush=True)
            elif progress_every and flow_count % progress_every == 0:
                print(f"[ingest] processed={flow_count} buffered={len(batch)}", flush=True)

        if batch:
            self.db_manager.bulk_insert_training(batch)
            inserted += len(batch)

        print(f"[ingest] completed total={flow_count} inserted={inserted}", flush=True)
        return {"flow_count": flow_count}

    # === Helpers ===
    def _forward_model(
        self, byte_tensor: torch.Tensor, global_tensor: torch.Tensor | None
    ) -> torch.Tensor | Sequence[torch.Tensor] | Dict[str, torch.Tensor]:
        if self.use_global_stats and global_tensor is not None:
            try:
                return self.model(byte_tensor, global_tensor)
            except TypeError:
                return self.model(byte_tensor)
        return self.model(byte_tensor)

    def _extract_reconstruction(self, output):
        if isinstance(output, dict):
            for key in ("reconstruction", "recon", "output"):
                if key in output:
                    return output[key]
            return next(iter(output.values()))
        if isinstance(output, (tuple, list)):
            if len(output) >= 4:
                return output[-1]
            return output[0]
        return output

    def _build_explanation(
        self, original: torch.Tensor, reconstructed: torch.Tensor, top_k: int = 5
    ) -> Dict[str, List[int]]:
        diff = torch.abs(original - reconstructed).view(-1)
        top_k = min(top_k, diff.numel())
        if top_k <= 0:
            return {"top_indices": []}
        indices = torch.topk(diff, top_k).indices.tolist()
        return {"top_indices": indices}

    def _load_model(self):
        if CorrelatedGaussianVAE is None:
            raise RuntimeError(
                "CorrelatedGaussianVAE is not available. Please add src/core/vae.py."
            )
        model_cfg = self.model_config
        latent_dim = int(model_cfg.get("latent_dim", 32))
        training_mode = str(model_cfg.get("training_mode", "correlated"))
        n_components = int(model_cfg.get("n_components", 5))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout_rate = float(model_cfg.get("dropout_rate", 0.2))
        global_fuse = str(model_cfg.get("global_fuse", "concat")).lower()

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
            "output_size": self.bytes_pkts * self.bytes_len,
            "output_shape": (self.bytes_pkts, self.bytes_len),
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
            global_dim=self.global_dim,
            global_mlp_hidden=model_cfg.get("global_mlp_hidden"),
            global_mlp_out=model_cfg.get("global_mlp_out"),
            global_fuse=global_fuse,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict or checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def _load_reference_stats(self, stats_path: str | Path | None) -> Optional[ReferenceStats]:
        config = ConfigLoader.get_instance()
        system_config = config.system_config
        model_config = config.model_config

        candidate_files: List[Path] = []
        if stats_path:
            candidate_files.append(Path(stats_path))
        if system_config.get("reference_stats_path"):
            candidate_files.append(Path(system_config["reference_stats_path"]))

        stats_dir = Path(system_config.get("stats_dir", "data/stats"))
        dataset_name = (
            system_config.get("dataset")
            or model_config.get("dataset")
            or model_config.get("dataset_name")
        )
        if dataset_name:
            candidate_files.append(stats_dir / f"reference_stats_{dataset_name}.pt")
        candidate_files.append(stats_dir / "reference_stats.pt")

        if stats_dir.exists():
            candidates = sorted(stats_dir.glob("reference_stats_*.pt"))
            if candidates:
                candidate_files.append(candidates[-1])

        for path in candidate_files:
            if path and path.exists():
                payload = torch.load(path, map_location=self.device)
                if not isinstance(payload, dict):
                    continue
                mean = payload.get("reference_mu")
                cov_inv = payload.get("reference_cov_inv")
                if mean is None or cov_inv is None:
                    continue
                cov = payload.get("reference_cov")
                if cov is not None:
                    cov = cov.to(self.device)
                cov_invsqrt = payload.get("reference_cov_invsqrt")
                if cov_invsqrt is not None:
                    cov_invsqrt = cov_invsqrt.to(self.device)
                return ReferenceStats(
                    mean=mean.to(self.device),
                    cov_inv=cov_inv.to(self.device),
                    cov=cov,
                    cov_invsqrt=cov_invsqrt,
                )
        return None

    def _load_benign_stats(self, stats_path: str | Path | None) -> BenignStats:
        candidate_files: Sequence[Path] = []
        if stats_path:
            candidate_files.append(Path(stats_path))
        if self.checkpoint_path is not None:
            candidate_files.append(self.checkpoint_path.with_name("benign_stats.json"))

        for path in candidate_files:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    return BenignStats(
                        mean=float(payload.get("mean", 0.0)),
                        std=float(payload.get("std", 1.0)),
                    )
        return BenignStats(mean=0.0, std=1.0)

    def _model_expects_token_bytes(self) -> bool:
        if self.model is None:
            return False
        encoder = getattr(self.model, "encoder", None)
        packet_encoder = getattr(encoder, "packet_encoder", None)
        return packet_encoder is not None and hasattr(packet_encoder, "embedding")

    def _prepare_latent_inputs(
        self, flow: ExtractedFlow
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        byte_tensor = torch.tensor(
            flow.byte_matrix, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.latent_input_mode == "normalized" and not self._model_expects_token_bytes():
            byte_tensor = byte_tensor / 255.0

        global_tensor = None
        if self.use_global_stats:
            if not self.global_stats_keys:
                self.global_stats_keys = sorted(flow.global_stats.keys())
            global_tensor = torch.tensor(
                [float(flow.global_stats.get(key, 0.0)) for key in self.global_stats_keys],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        return byte_tensor, global_tensor

    def _mahalanobis(self, mu: torch.Tensor, stats: ReferenceStats) -> torch.Tensor:
        diff = mu - stats.mean.unsqueeze(0)
        score = torch.einsum("bi,ij,bj->b", diff, stats.cov_inv, diff)
        score = torch.clamp(score, min=0.0)
        return torch.sqrt(score).squeeze(0)

    def _decide_anomaly(
        self, recon_z: float, vae_score: Optional[float]
    ) -> tuple[bool, str]:
        mode = self.anomaly_score_mode or "recon"
        if mode in ("vae", "latent", "mahalanobis"):
            if vae_score is None:
                raise RuntimeError("VAE score requested but reference stats are missing.")
            return vae_score >= self.latent_threshold, "vae"
        if mode in ("both", "and"):
            if vae_score is None:
                return recon_z >= self.recon_threshold, "recon"
            return (
                recon_z >= self.recon_threshold and vae_score >= self.latent_threshold,
                "both",
            )
        if mode in ("either", "or"):
            if vae_score is None:
                return recon_z >= self.recon_threshold, "recon"
            return (
                recon_z >= self.recon_threshold or vae_score >= self.latent_threshold,
                "either",
            )
        return recon_z >= self.recon_threshold, "recon"

    def _flush_inference_batch(self, batch: List[FlowRecordCreate]) -> None:
        if not batch:
            return
        if self.db_manager is None:
            batch.clear()
            return
        self.db_manager.bulk_insert(batch)
        batch.clear()

    def _ensure_mode(self, expected: str) -> None:
        if self.mode != expected:
            raise RuntimeError(f"Service is in '{self.mode}' mode, expected '{expected}'.")


__all__ = ["InferenceService", "InferenceResult"]
