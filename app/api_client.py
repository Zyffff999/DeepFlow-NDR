from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class APIClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.getenv("DEEPFLOW_API_URL", "http://localhost:8000")
        self.timeout = float(os.getenv("DEEPFLOW_API_TIMEOUT", "600"))

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        response = requests.request(method, url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def analyze_pcap(self, file_path: Path, top_k: Optional[int] = None) -> Dict[str, Any]:
        with file_path.open("rb") as handle:
            files = {"pcap_file": (file_path.name, handle, "application/octet-stream")}
            params = {"top_k": top_k} if top_k is not None else None
            return self._request("POST", "/analyze", files=files, params=params)

    def get_flows(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        params = {"limit": limit, "offset": offset}
        return self._request("GET", "/flows", params=params)

    def get_metrics(self, window_minutes: int = 60) -> Dict[str, Any]:
        params = {"window_minutes": window_minutes}
        return self._request("GET", "/metrics", params=params)


__all__ = ["APIClient"]
