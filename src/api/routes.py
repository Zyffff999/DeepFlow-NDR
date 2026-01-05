from __future__ import annotations

import csv
import heapq
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from src.services.inference import InferenceResult, InferenceService
from src.utils.config import ConfigLoader


router = APIRouter()

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


def _score_value(result: InferenceResult) -> float:
    score_type = str(result.score_type or "").lower()
    if score_type in ("vae", "latent", "mahalanobis"):
        return float(result.vae_score or 0.0)
    if score_type in ("both", "either"):
        if result.vae_score is None:
            return float(result.z_score)
        return float(max(result.z_score, result.vae_score))
    return float(result.z_score)


def _format_result(result: InferenceResult) -> Dict[str, Any]:
    flow = result.flow
    return {
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
        "explanation": result.explanation,
        "global_stats": dict(flow.global_stats),
    }


def _format_result_flat(
    result: InferenceResult, global_keys: List[str]
) -> Dict[str, Any]:
    payload = _format_result(result)
    payload.pop("explanation", None)
    stats = payload.pop("global_stats", {})
    for key in global_keys:
        payload[f"gs_{key}"] = float(stats.get(key, 0.0))
    return payload


def _upload_dir() -> Path:
    config = ConfigLoader.get_instance().system_config
    return Path(config.get("upload_dir", "data/uploads"))


def _output_dir() -> Path:
    config = ConfigLoader.get_instance().system_config
    return Path(config.get("api_output_dir", "data/outputs/api"))


@lru_cache()
def get_inference_service() -> InferenceService:
    config = ConfigLoader.get_instance()
    system_cfg = config.system_config
    checkpoint_path = system_cfg.get("api_checkpoint_path") or system_cfg.get(
        "offline_checkpoint_path"
    )
    if not checkpoint_path:
        raise RuntimeError("api_checkpoint_path/offline_checkpoint_path is not set.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found at {checkpoint}.")
    return InferenceService(
        checkpoint_path=checkpoint,
        benign_stats_path=system_cfg.get("offline_benign_stats_path"),
        reference_stats_path=system_cfg.get("offline_reference_stats_path"),
        use_db=bool(system_cfg.get("inference_use_db", False)),
        store_results=bool(system_cfg.get("inference_store_results", False)),
    )


@router.post("/analyze")
async def analyze_pcap(
    pcap_file: UploadFile = File(...),
    top_k: Optional[int] = Query(None, ge=0, le=500),
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict:
    if not pcap_file.filename:
        raise HTTPException(status_code=400, detail="PCAP filename is missing.")

    upload_dir = _upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4()}_{pcap_file.filename}"

    config = ConfigLoader.get_instance().system_config
    if top_k is None:
        top_k = int(config.get("api_top_k", config.get("offline_top_k", 10)) or 10)
    save_output = bool(config.get("api_save_output", True))
    output_only_anomalies = bool(config.get("api_output_only_anomalies", False))
    output_dir = _output_dir()
    output_path = None
    writer = None

    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = await pcap_file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        if save_output:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"api_{timestamp}.csv"
            handle = output_path.open("w", encoding="utf-8", newline="")
            global_keys = inference_service.global_stats_keys or []
            writer = csv.DictWriter(
                handle,
                fieldnames=BASE_FIELDS + [f"gs_{key}" for key in global_keys],
            )
            writer.writeheader()
        else:
            handle = None

        heap: List[tuple[float, Dict[str, Any]]] = []
        flow_count = 0
        anomalies = 0
        stored = 0
        try:
            for result in inference_service.iter_pcap_results(destination):
                flow_count += 1
                if result.is_anomaly:
                    anomalies += 1
                if top_k and top_k > 0:
                    score = _score_value(result)
                    snapshot = _format_result(result)
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, snapshot))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, snapshot))
                if writer is not None:
                    if output_only_anomalies and not result.is_anomaly:
                        continue
                    row = _format_result_flat(result, global_keys)
                    writer.writerow(row)
                    stored += 1
        finally:
            if handle is not None:
                handle.close()

        top_payload = [item[1] for item in sorted(heap, key=lambda x: x[0], reverse=True)]
        return {
            "flow_count": flow_count,
            "anomalies": anomalies,
            "top_k": top_payload,
            "stored": stored,
            "output_file": str(output_path) if output_path else None,
        }
    finally:
        await pcap_file.close()
        if destination.exists():
            try:
                destination.unlink()
            except PermissionError:
                pass


@router.get("/flows")
def get_flows(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict:
    if inference_service.db_manager is None:
        raise HTTPException(status_code=400, detail="DB is disabled in config.")
    records = inference_service.db_manager.get_flows(limit=limit, offset=offset)
    return {"results": records, "count": len(records), "limit": limit, "offset": offset}


@router.get("/metrics")
def get_metrics(
    window_minutes: int = Query(60, ge=1, le=1440),
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict:
    if inference_service.db_manager is None:
        raise HTTPException(status_code=400, detail="DB is disabled in config.")
    metrics = inference_service.db_manager.get_metrics(window_minutes=window_minutes)
    return metrics
