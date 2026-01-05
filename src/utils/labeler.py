from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml

from src.pipeline.extractor import ExtractedFlow


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _match_ip(target: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    try:
        ip = ip_address(target)
    except ValueError:
        return False
    for raw in patterns:
        text = str(raw).strip()
        if not text:
            continue
        try:
            if "/" in text:
                if ip in ip_network(text, strict=False):
                    return True
            elif ip == ip_address(text):
                return True
        except ValueError:
            continue
    return False


@dataclass
class LabelRule:
    label: int
    name: str = ""
    src_ips: Sequence[str] = ()
    dst_ips: Sequence[str] = ()
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    def matches(self, flow: ExtractedFlow) -> bool:
        if self.time_start and flow.timestamp < self.time_start:
            return False
        if self.time_end and flow.timestamp > self.time_end:
            return False
        if not _match_ip(flow.src_ip, self.src_ips):
            return False
        if not _match_ip(flow.dst_ip, self.dst_ips):
            return False
        return True


class FlowLabeler:
    def __init__(self, rules: Iterable[LabelRule], default_label: int = 0) -> None:
        self.rules = list(rules)
        self.default_label = int(default_label)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlowLabeler":
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Label rules not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        default_label = int(payload.get("default_label", 0))
        rules = []
        for item in payload.get("rules", []) or []:
            rules.append(
                LabelRule(
                    label=int(item.get("label", 0)),
                    name=str(item.get("name", "")),
                    src_ips=list(item.get("src_ips", []) or []),
                    dst_ips=list(item.get("dst_ips", []) or []),
                    time_start=_parse_time(item.get("time_start")),
                    time_end=_parse_time(item.get("time_end")),
                )
            )
        return cls(rules, default_label=default_label)

    def label(self, flow: ExtractedFlow) -> int:
        for rule in self.rules:
            if rule.matches(flow):
                return rule.label
        return self.default_label


__all__ = ["FlowLabeler", "LabelRule"]
