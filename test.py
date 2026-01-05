from __future__ import annotations

import csv
import heapq
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.services.inference import InferenceResult, InferenceService  # noqa: E402
from src.utils.config import ConfigLoader  # noqa: E402


BASE_FIELDS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "recon_loss",
    "z_score",
    "vae_score",
    "score_type",
    "anomaly_score",
    "is_anomaly",
]


def format_result(
    result: InferenceResult, global_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    flow = result.flow
    payload: Dict[str, Any] = {
        "timestamp": flow.timestamp.isoformat(),
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "dst_port": flow.dst_port,
        "protocol": flow.protocol,
        "recon_loss": result.recon_loss,
        "z_score": result.z_score,
        "vae_score": result.vae_score,
        "score_type": result.score_type,
        "anomaly_score": _score_value(result),
        "is_anomaly": result.is_anomaly,
    }
    if global_keys is None:
        payload["global_stats"] = dict(flow.global_stats)
    else:
        for key in global_keys:
            payload[f"gs_{key}"] = float(flow.global_stats.get(key, 0.0))
    return payload


def _score_value(result: InferenceResult) -> float:
    score_type = str(result.score_type or "").lower()
    if score_type in ("vae", "latent", "mahalanobis"):
        return float(result.vae_score or 0.0)
    if score_type in ("both", "either"):
        if result.vae_score is None:
            return float(result.z_score)
        return float(max(result.z_score, result.vae_score))
    return float(result.z_score)


def _cfg_path(config: Dict[str, Any], key: str) -> Optional[Path]:
    value = config.get(key)
    if not value:
        return None
    return Path(value)


def main() -> None:
    config = ConfigLoader.get_instance()
    system_cfg = config.system_config

    pcap_path = _cfg_path(system_cfg, "offline_pcap_path")
    checkpoint = _cfg_path(system_cfg, "offline_checkpoint_path")
    reference_stats = _cfg_path(system_cfg, "offline_reference_stats_path")
    benign_stats = _cfg_path(system_cfg, "offline_benign_stats_path")
    output_dir = Path(system_cfg.get("offline_output_dir", "data/outputs"))
    output_format = str(system_cfg.get("offline_output_format", "csv")).lower()
    output_only_anomalies = bool(system_cfg.get("offline_output_only_anomalies", True))
    if output_format not in {"csv", "jsonl"}:
        output_format = "csv"

    if pcap_path is None:
        raise ValueError("offline_pcap_path is required in the config.")
    if checkpoint is None:
        raise ValueError("offline_checkpoint_path is required in the config.")
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found at {pcap_path}.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}.")

    service = InferenceService(
        checkpoint_path=checkpoint,
        db_url=system_cfg.get("db_url"),
        benign_stats_path=benign_stats,
        reference_stats_path=reference_stats,
        use_db=bool(system_cfg.get("inference_use_db", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "jsonl" if output_format == "jsonl" else "csv"
    output_path = output_dir / f"offline_{timestamp}.{extension}"

    top_k = int(system_cfg.get("offline_top_k", 10) or 0)
    heap: List[tuple[float, Dict[str, Any]]] = []
    flow_count = 0
    anomaly_total = 0
    stored = 0
    global_keys = service.global_stats_keys or []

    if output_format == "jsonl":
        handle = output_path.open("w", encoding="utf-8")
        writer = None
    else:
        handle = output_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            handle,
            fieldnames=BASE_FIELDS + [f"gs_{key}" for key in global_keys],
        )
        writer.writeheader()

    with handle:
        for result in service.iter_pcap_results(pcap_path):
            flow_count += 1
            if result.is_anomaly:
                anomaly_total += 1

            if top_k > 0:
                score = _score_value(result)
                snapshot = format_result(result)
                if len(heap) < top_k:
                    heapq.heappush(heap, (score, snapshot))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, snapshot))

            if output_only_anomalies and not result.is_anomaly:
                continue

            if output_format == "jsonl":
                handle.write(json.dumps(format_result(result), ensure_ascii=True) + "\n")
            else:
                row = format_result(result, global_keys=global_keys)
                if writer is not None:
                    writer.writerow(row)
            stored += 1

    top_payload = [item[1] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
    payload: Dict[str, Any] = {
        "summary": {
            "flow_count": flow_count,
            "anomaly_total": anomaly_total,
            "stored": stored,
            "output_file": str(output_path),
        },
        "top_k": top_payload,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
