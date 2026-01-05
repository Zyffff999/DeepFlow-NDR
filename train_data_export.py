from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.services.db_manager import DBManager, TrainingFlow
from src.services.inference import InferenceService
from src.utils.config import ConfigLoader


# DEFAULT_FILTERS: Dict[str, Any] = {
#     "protocol_allowlist": ["tcp", "udp", "6", "17"],
#     "min_total_packets": 4,
#     "require_bidirectional_bytes": True,
#     "exclude_ports": [137, 138, 1900, 5353],
#     "exclude_dns": True,
# }

# 推荐的训练配置 (Training Config)
DEFAULT_FILTERS = {
    # 允许所有协议
    "protocol_allowlist": ["tcp", "udp"],
    
    # 降低包数门槛，让模型学习 NTP 和 DNS 这种短流
    "min_total_packets": 2,  
    
    # 关闭双向强制，允许学习失败的连接或单向通知
    "require_bidirectional_bytes": False, 
    
    # 允许 DNS (非常重要!)
    "exclude_dns": False,
    
    # 建议只过滤极其特殊的非IP流量，或者保留为空
    # 137/138/1900 在内网非常常见，训练时最好留着
    "exclude_ports": [137, 1900, 5353], 
}


def _normalize_protocol(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("6", "tcp"):
        return "tcp"
    if text in ("17", "udp"):
        return "udp"
    return text or None


def filter_flow(row: TrainingFlow, filters: Dict[str, Any]) -> bool:
    """
    SOC-focused benign filtering with protocol/traffic rules.
    Override defaults via --filters JSON.
    """
    cfg = dict(DEFAULT_FILTERS)
    for key, value in filters.items():
        if key in cfg and value is not None:
            cfg[key] = value

    gs = row.global_stats or {}
    protocol = _normalize_protocol(row.protocol)
    allowlist = { _normalize_protocol(p) for p in cfg.get("protocol_allowlist", []) }
    allowlist.discard(None)
    if allowlist and protocol not in allowlist:
        return False

    total_packets = gs.get("total_packets")
    if total_packets is None:
        src_packets = gs.get("src2dst_packets", 0) or 0
        dst_packets = gs.get("dst2src_packets", 0) or 0
        total_packets = src_packets + dst_packets
    total_packets = int(total_packets or 0)
    if total_packets < int(cfg.get("min_total_packets", 0) or 0):
        return False

    src_bytes = int(gs.get("src2dst_bytes", 0) or 0)
    dst_bytes = int(gs.get("dst2src_bytes", 0) or 0)
    if bool(cfg.get("require_bidirectional_bytes", True)):
        if src_bytes <= 0 or dst_bytes <= 0:
            return False

    dst_port = int(row.dst_port or 0)
    if dst_port in set(cfg.get("exclude_ports", []) or []):
        return False

    if bool(cfg.get("exclude_dns", True)) and protocol == "udp" and dst_port == 53:
        return False

    for key, value in filters.items():
        if key in cfg or value is None:
            continue
        if not hasattr(row, key):
            continue
        if getattr(row, key) != value:
            return False
    return True


def export_training_file(
    svc: InferenceService,
    output_file: Path,
    limit: Optional[int],
    filters: Dict[str, Any],
) -> Dict[str, int | str]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    batch_bytes: List[np.ndarray] = []
    batch_globals: List[np.ndarray] = []
    exported = 0

    db: DBManager = svc.db_manager
    for row in db.iter_training_flows(batch_size=svc.training_batch_size):
        if not filter_flow(row, filters):
            continue
        if not svc.global_stats_keys:
            svc.global_stats_keys = sorted((row.global_stats or {}).keys())
        raw_bytes = np.asarray(row.raw_bytes, dtype=np.uint8)
        global_stats = np.asarray(
            [float((row.global_stats or {}).get(key, 0.0)) for key in svc.global_stats_keys],
            dtype=np.float32,
        )
        batch_bytes.append(raw_bytes)
        batch_globals.append(global_stats)
        exported += 1
        if limit is not None and exported >= limit:
            break

    if not batch_bytes:
        return {"exported": 0, "file_written": 0, "output_file": str(output_file)}

    np.savez_compressed(
        output_file,
        byte_matrix=np.stack(batch_bytes),
        global_stats=np.stack(batch_globals),
        global_stats_keys=np.asarray(svc.global_stats_keys, dtype=object),
    )
    return {"exported": exported, "file_written": 1, "output_file": str(output_file)}


def parse_filters(filter_json: Optional[str]) -> Dict[str, Any]:
    if not filter_json:
        return {}
    return json.loads(filter_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export training samples from DB to a single NPZ file (raw bytes)."
    )
    config = ConfigLoader.get_instance()
    default_dir = Path(
        config.system_config.get(
            "training_data_dir",
            config.system_config.get("training_shards_dir", "data/training/shards"),
        )
    )
    default_file = default_dir / "training_data.npz"
    parser.add_argument("--output-file", type=Path, default=default_file)
    parser.add_argument(
        "--limit",
        type=int,
        default=250000,
        help="Max samples to export (default: 100000). Use 0 to export all.",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help='JSON string for filters, e.g. {"exclude_dns": false, "min_total_packets": 6}',
    )
    args = parser.parse_args()

    svc = InferenceService(mode="training")
    filters = parse_filters(args.filters)
    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    result = export_training_file(svc, args.output_file, limit, filters)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
