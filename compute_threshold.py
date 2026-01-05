from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml

from src.pipeline.extractor import ExtractedFlow
from src.services.db_manager import TrainingFlow
from src.services.inference import InferenceService
from src.utils.config import ConfigLoader


def _row_to_flow(row: TrainingFlow) -> ExtractedFlow:
    return ExtractedFlow(
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


def _compute_scores(
    svc: InferenceService,
    limit: int | None,
    apply_filters: bool,
    progress_every: int,
) -> List[float]:
    if svc.db_manager is None:
        raise RuntimeError("DB is disabled; cannot read training flows.")

    scores: List[float] = []
    count = 0
    kept = 0
    mode = (svc.anomaly_score_mode or "recon").lower()
    if mode in ("both", "and", "either", "or"):
        raise ValueError("Combined score modes are not supported for quantile threshold.")

    for row in svc.db_manager.iter_training_flows(batch_size=1000):
        flow = _row_to_flow(row)
        if apply_filters and not svc._passes_inference_filters(flow):
            continue

        if mode in ("vae", "latent", "mahalanobis"):
            score = svc._compute_latent_score(flow)
        else:
            recon_loss, _ = svc._predict(flow)
            score = (recon_loss - svc.stats.mean) / svc.stats.safe_std

        if score is None:
            continue
        scores.append(float(score))
        kept += 1
        count += 1
        if limit is not None and kept >= limit:
            break
        if progress_every > 0 and count % progress_every == 0:
            print(f"[threshold] processed={count} kept={kept}")

    return scores


def _update_config(config_path: Path, threshold: float) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload.setdefault("system", {})
    payload["system"]["threshold_mahalanobis"] = float(threshold)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute empirical quantile threshold from training flows in DB."
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="False positive rate.")
    parser.add_argument("--limit", type=int, default=0, help="Max samples to score (0=all).")
    parser.add_argument(
        "--apply-filters", action="store_true", help="Apply inference filters."
    )
    parser.add_argument("--write-config", action="store_true", help="Write to config.yaml.")
    parser.add_argument(
        "--progress-every", type=int, default=5000, help="Progress print interval."
    )
    args = parser.parse_args()

    cfg = ConfigLoader.get_instance().system_config
    checkpoint = cfg.get("offline_checkpoint_path") or cfg.get("api_checkpoint_path")
    reference_stats = cfg.get("offline_reference_stats_path")
    if not checkpoint:
        raise ValueError("offline_checkpoint_path or api_checkpoint_path is required.")
    if not reference_stats:
        raise ValueError("offline_reference_stats_path is required for Mahalanobis.")

    svc = InferenceService(
        checkpoint_path=checkpoint,
        reference_stats_path=reference_stats,
        benign_stats_path=cfg.get("offline_benign_stats_path"),
        use_db=True,
        store_results=False,
    )

    limit = None if args.limit <= 0 else args.limit
    scores = _compute_scores(
        svc, limit=limit, apply_filters=args.apply_filters, progress_every=args.progress_every
    )
    if not scores:
        raise RuntimeError("No scores were computed. Check DB content and filters.")

    quantile = 1.0 - float(args.alpha)
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))
    print(f"mode={svc.anomaly_score_mode or 'recon'} samples={len(scores)}")
    print(f"alpha={args.alpha} quantile={quantile} threshold={threshold}")

    if args.write_config:
        config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
        _update_config(config_path, threshold)
        print(f"updated: {config_path} -> system.threshold_mahalanobis")


if __name__ == "__main__":
    main()
