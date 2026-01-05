from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    from app.api_client import APIClient
except ModuleNotFoundError:  # pragma: no cover - streamlit path fallback
    from api_client import APIClient


st.set_page_config(page_title="DeepFlow NDR Dashboard", layout="wide")
st.title("DeepFlow NDR")
st.caption("Upload PCAPs, monitor flows, and track anomalies in real time.")

api_default = os.getenv("DEEPFLOW_API_URL", "http://localhost:8000")
api_url = st.text_input("API Base URL", value=api_default)
client = APIClient(api_url)

if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)
if "job_future" not in st.session_state:
    st.session_state.job_future = None
if "job_started_at" not in st.session_state:
    st.session_state.job_started_at = None
if "job_temp_path" not in st.session_state:
    st.session_state.job_temp_path = None


def _run_analysis(base_url: str, temp_path: str, top_k: int) -> dict:
    client = APIClient(base_url)
    return client.analyze_pcap(Path(temp_path), top_k=top_k)

upload_tab, monitor_tab = st.tabs(["Upload", "Monitor"])

with upload_tab:
    st.subheader("Analyze PCAP")
    uploaded = st.file_uploader("Drop a PCAP/PCAPNG file", type=["pcap", "pcapng"])
    top_k = st.number_input("Top-K anomalies to return", min_value=0, max_value=500, value=10)
    analyze_btn = st.button("Analyze Capture", width="stretch", disabled=uploaded is None)
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    summary_placeholder = st.empty()
    topk_placeholder = st.empty()

    if analyze_btn and uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as temp_file:
            temp_file.write(uploaded.getbuffer())
            temp_path = Path(temp_file.name)
        st.session_state.job_started_at = time.time()
        st.session_state.job_temp_path = str(temp_path)
        st.session_state.job_future = st.session_state.executor.submit(
            _run_analysis, api_url, str(temp_path), int(top_k)
        )
        progress_bar.progress(10)
        status_placeholder.info("Upload complete. Processing capture...")

    job_future: Future | None = st.session_state.job_future
    if job_future is not None:
        if job_future.done():
            result = None
            try:
                try:
                    result = job_future.result()
                except requests.RequestException as exc:  # pragma: no cover - UI only
                    progress_bar.progress(0)
                    status_placeholder.error(f"Failed to analyze PCAP: {exc}")
                except Exception as exc:  # pragma: no cover - UI only
                    progress_bar.progress(0)
                    status_placeholder.error(f"Failed to analyze PCAP: {exc}")
                else:
                    progress_bar.progress(100)
                    status_placeholder.success(
                        f"Processed {result.get('flow_count', 0)} flows "
                        f"({result.get('anomalies', 0)} anomalies)."
                    )

                if result is not None:
                    summary_placeholder.json(result.get("summary", result))
                    output_file = result.get("output_file")
                    if output_file:
                        st.caption(f"CSV saved to: {output_file}")
                    topk = result.get("top_k", [])
                    if topk:
                        topk_df = pd.DataFrame(topk)
                        if "global_stats" in topk_df.columns:
                            stats_df = pd.json_normalize(topk_df["global_stats"]).add_prefix("gs_")
                            topk_df = pd.concat(
                                [topk_df.drop(columns=["global_stats"]), stats_df], axis=1
                            )
                        topk_placeholder.dataframe(topk_df, width="stretch")
                        numeric_cols = [
                            col
                            for col in topk_df.columns
                            if pd.api.types.is_numeric_dtype(topk_df[col])
                        ]
                        default_metrics = [
                            col
                            for col in (
                                "anomaly_score",
                                "gs_max_ps",
                                "gs_mean_ps",
                                "gs_bytes_per_sec",
                                "gs_packets_per_sec",
                                "gs_duration_ms",
                            )
                            if col in numeric_cols
                        ]
                        selected = st.multiselect(
                            "Visualize metrics (Top-K)",
                            options=numeric_cols,
                            default=default_metrics,
                        )
                        if selected:
                            topk_df["flow_id"] = (
                                topk_df["src_ip"].astype(str)
                                + ":"
                                + topk_df["dst_port"].astype(str)
                                + " -> "
                                + topk_df["dst_ip"].astype(str)
                            )
                            chart_df = topk_df.set_index("flow_id")[selected]
                            st.bar_chart(chart_df)
                    else:
                        topk_placeholder.info("No anomalies returned.")
            finally:
                temp_path_str = st.session_state.job_temp_path
                if temp_path_str:
                    Path(temp_path_str).unlink(missing_ok=True)
                st.session_state.job_future = None
                st.session_state.job_started_at = None
                st.session_state.job_temp_path = None
        else:
            started_at = st.session_state.job_started_at or time.time()
            elapsed = time.time() - started_at
            progress = min(95, 10 + int(elapsed * 2))
            progress_bar.progress(progress)
            status_placeholder.info(f"Processing... {elapsed:.1f}s elapsed")
            st_autorefresh(interval=1000, key="upload_refresh")

with monitor_tab:
    st.subheader("Network Monitor")
    enable_monitor = st.checkbox("Enable live monitor (requires DB)", value=False)
    if not enable_monitor:
        st.info("Monitor polling is disabled. Enable it once DB is turned on.")
    else:
        refresh_seconds = st.slider(
            "Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=5
        )
        st_autorefresh(interval=refresh_seconds * 1000, key="monitor_refresh")
        try:
            flows_payload = client.get_flows(limit=200)
            metrics_payload = client.get_metrics(window_minutes=60)
        except requests.HTTPError as exc:  # pragma: no cover - UI only
            st.info("DB is disabled; enable inference_use_db to view flows/metrics.")
            st.caption(str(exc))
        except requests.RequestException as exc:  # pragma: no cover - UI only
            st.error(f"Backend unreachable: {exc}")
        else:
            flow_df = pd.DataFrame(flows_payload.get("results", []))
            if flow_df.empty:
                st.info("No flow records yet. Upload a PCAP to begin.")
            else:
                flow_df["timestamp"] = pd.to_datetime(flow_df["timestamp"])
                flow_df.rename(columns={"z_score": "anomaly_score"}, inplace=True)

                def highlight_anomalies(row):
                    color = "background-color: #ffcccc" if row["anomaly_score"] >= 3.0 else ""
                    return [color] * len(row)

                st.dataframe(
                    flow_df.style.apply(highlight_anomalies, axis=1),
                    width="stretch",
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Flows (1h)", int(metrics_payload.get("total_flows", 0)))
            col2.metric("Anomalies (1h)", int(metrics_payload.get("anomaly_count", 0)))
            col3.metric(
                "Anomaly Rate",
                f"{metrics_payload.get('anomaly_rate', 0.0) * 100:.1f}%",
            )
