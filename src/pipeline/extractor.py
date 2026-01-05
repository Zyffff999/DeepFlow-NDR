from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from nfstream import NFPlugin, NFStreamer


def _zero_address_bytes(payload: bytes) -> bytes:
    """
    鲁棒的地址脱敏函数：
    1. 优先作为 L2 (Ethernet) 处理，强制抹除 MAC。
    2. 支持识别 VLAN (802.1Q)，正确处理偏移量。
    3. 如果不是 Ethernet，则尝试作为 Raw L3 (IP) 处理。
    4. 抹除 IPv4/IPv6 的源和目的 IP。
    """
    if not payload:
        return payload
    
    # 转换为可变的 bytearray
    data = bytearray(payload)
    length = len(data)
    
    # 状态标记
    is_ethernet = False
    ip_offset = 0
    ethertype = 0
    
    # === 1. 尝试作为 L2 Ethernet 处理 ===
    # NFStream 的 packet.raw 通常包含 L2 头
    # 最小以太网帧头为 14 字节 (6 Dst + 6 Src + 2 Type)
    if length >= 14:
        raw_et = (data[12] << 8) | data[13]
        
        # 情况 A: 带 VLAN 标签 (802.1Q, EtherType 0x8100)
        if raw_et == 0x8100 and length >= 18:
            # 抹除 MAC (0-12字节)
            # 保留 VLAN 标签 (12-16字节)
            data[0:6] = b"\x00" * 6
            data[6:12] = b"\x00" * 6
            
            # 真正的 EtherType 在偏移量 16
            ethertype = (data[16] << 8) | data[17]
            ip_offset = 18
            is_ethernet = True
            
        # 情况 B: 标准以太网 (IPv4, IPv6, ARP 等)
        # 只要长度够且没命中 VLAN，我们默认它是 Ethernet 并抹除 MAC
        # (注：严格来说应判断 raw_et >= 1536，但为了安全这里对所有长得像 L2 的都抹 MAC)
        elif length >= 14:
            # 抹除 MAC
            data[0:6] = b"\x00" * 6
            data[6:12] = b"\x00" * 6
            
            ethertype = raw_et
            ip_offset = 14
            is_ethernet = True

    # === 2. 尝试作为 L3 IP 处理 ===
    # 确定目标是 IPv4 还是 IPv6
    target_is_ipv4 = False
    target_is_ipv6 = False
    
    if is_ethernet:
        if ethertype == 0x0800:
            target_is_ipv4 = True
        elif ethertype == 0x86DD:
            target_is_ipv6 = True
    else:
        # Fallback: 如果不是以太网（或无法识别），尝试作为纯 L3 包解析
        # 检查第一个字节的版本号
        if length > 0:
            version = data[0] >> 4
            if version == 4:
                target_is_ipv4 = True
            elif version == 6:
                target_is_ipv6 = True
                
    # === 3. 执行 IP 抹除 ===
    if target_is_ipv4:
        # IPv4: Src 偏移 12, Dst 偏移 16 (相对于 ip_offset)
        # 总共抹除 8 字节
        if length >= ip_offset + 20:
            start = ip_offset + 12
            data[start : start + 8] = b"\x00" * 8
            
    elif target_is_ipv6:
        # IPv6: Src 偏移 8, Dst 偏移 24 (相对于 ip_offset)
        # 总共抹除 32 字节
        if length >= ip_offset + 40:
            start = ip_offset + 8
            data[start : start + 32] = b"\x00" * 32
            
    return bytes(data)


@dataclass
class ExtractedFlow:
    byte_matrix: np.ndarray

    iat_series: List[float]
    size_series: List[int]
    global_stats: Dict[str, float]

    # === 4. 基础元数据 ===
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    timestamp: datetime


class ETALogicExtractor(NFPlugin):
    """
    涵盖: 头部字节矩阵、时序序列、交互序列。
    """

    def __init__(
        self, 
        bytes_pkts: int = 8,      # 头部指纹只看前 8 个包
        bytes_len: int = 64,      # 每个包只看前 64 字节 (Header)
        seq_len: int = 50         # 序列特征提取前 50 个交互
    ) -> None:
        self.bytes_pkts = bytes_pkts
        self.bytes_len = bytes_len
        self.seq_len = seq_len

    def on_init(self, packet, flow) -> None:
        # 1. Init byte matrix
        flow.udps.byte_matrix = np.zeros(
            (self.bytes_pkts, self.bytes_len), dtype=np.uint8
        )
        flow.udps.bytes_count = 0  # packets captured so far

        # 2. Init sequence stats
        flow.udps.iat_list = []
        flow.udps.size_list = []

        # Helper: last_seen_ms for IAT
        flow.udps.last_seen_ms = packet.time

        # 3. Record first seen timestamp
        flow.udps.first_seen_ms = getattr(
            flow, "bidirectional_first_seen_ms", packet.time
        )

        # Process first packet
        self._process_packet(packet, flow)
        
    def on_update(self, packet, flow) -> None:
        self._process_packet(packet, flow)

    def _process_packet(self, packet, flow) -> None:
        # === A. 提取 IAT 序列 ===
        if len(flow.udps.iat_list) < self.seq_len:
            # 为了稳妥，这里假设 packet.time 是时间戳
            current_ms = packet.time
            delta = float(current_ms - flow.udps.last_seen_ms)
            
            # 只有当这不是第一个包时，IAT 才有意义
            # 但为了序列对齐，第一个包的 IAT 可以记为 0.0 或者跳过
            # 这里策略：第一个包之后才记录 delta
            if delta >= 0 and len(flow.udps.iat_list) > 0 or (len(flow.udps.iat_list) == 0 and delta > 0): 
                 # 简单的逻辑：直接追加 delta，如果是第一个包 delta=0
                 pass

            flow.udps.iat_list.append(max(0.0, delta))
            flow.udps.last_seen_ms = current_ms

        # === B. 提取 Size 序列 (带方向) ===
        if len(flow.udps.size_list) < self.seq_len:
            # packet.direction: 0 = src->dst, 1 = dst->src
            # 我们定义: src->dst 为正, dst->src 为负
            direction_sign = 1 if packet.direction == 0 else -1
            payload_size = packet.payload_size
            # 如果 payload_size 为0 (纯ACK)，也可以记录，体现交互节奏
            flow.udps.size_list.append(direction_sign * payload_size)

        # === C. 提取 Header Bytes ===
        if flow.udps.bytes_count < self.bytes_pkts:
            # 优先尝试获取 Raw L2 数据 (Ethernet)
            raw_bytes = getattr(packet, "raw", None)
            
            # 如果没有 Raw，尝试获取 L3 数据 (IP Packet)
            # 这通常发生在 Tunnel 解码或特殊接口捕获时
            if raw_bytes is None:
                raw_bytes = getattr(packet, "ip_packet", None)
            
            # 如果还是没有，尝试 payload (虽然 payload 通常不含头，但作为兜底)
            if raw_bytes is None:
                raw_bytes = getattr(packet, "payload", None)
            
            if raw_bytes:
                limit_data = bytes(raw_bytes)[: self.bytes_len]
                # 调用增强后的脱敏函数
                limit_data = _zero_address_bytes(limit_data)
                arr = np.frombuffer(limit_data, dtype=np.uint8)
                
                # 确保 arr 长度匹配 byte_matrix 的列宽，防止溢出或不足
                copy_len = min(arr.size, self.bytes_len)
                flow.udps.byte_matrix[flow.udps.bytes_count, : copy_len] = arr[:copy_len]
                
                flow.udps.bytes_count += 1


def iter_pcap(
    pcap_path: str | Path,
    bytes_pkts: int = 8,
    bytes_len: int = 64,
    seq_len: int = 50,
    idle_timeout: float | int | None = None,
    active_timeout: float | int | None = None,
) -> Iterable[ExtractedFlow]:
    """
    处理 PCAP 文件，返回包含 ETA 多模态特征的流对象列表。
    """
    resolved = Path(pcap_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"PCAP source not found: {resolved}")

    # 加载自定义插件
    plugin = ETALogicExtractor(
        bytes_pkts=bytes_pkts, 
        bytes_len=bytes_len, 
        seq_len=seq_len
    )
    
    # 启动流处理
    # decode_tunnels=True 推荐开启，能看到隧道内的真实 IP
    
    streamer_kwargs = {
        "source": str(resolved),
        "udps": plugin,
        "decode_tunnels": True,
    }
    if idle_timeout is not None:
        streamer_kwargs["idle_timeout"] = idle_timeout
    if active_timeout is not None:
        streamer_kwargs["active_timeout"] = active_timeout
    streamer = NFStreamer(**streamer_kwargs)
    for flow in streamer:
        # 必须确保插件正常初始化了数据
        if not hasattr(flow.udps, "byte_matrix"):
            continue

        # 1. 获取基础时间
        timestamp_ms = getattr(flow.udps, "first_seen_ms", 0)
        timestamp = (
            datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            if timestamp_ms > 0
            else datetime.now(tz=timezone.utc)
        )

        # 2. 计算/获取全局统计特征 (给 SOC 看的 RCA 指标)
        src_bytes = int(getattr(flow, "src2dst_bytes", 0) or 0)
        dst_bytes = int(getattr(flow, "dst2src_bytes", 0) or 0)
        src_packets = int(getattr(flow, "src2dst_packets", 0) or 0)
        dst_packets = int(getattr(flow, "dst2src_packets", 0) or 0)
        total_packets = int(getattr(flow, "bidirectional_packets", 0) or (src_packets + dst_packets))
        total_bytes = int(getattr(flow, "bidirectional_bytes", 0) or (src_bytes + dst_bytes))

        byte_ratio = float(src_bytes) / (float(dst_bytes) + 1.0)
        packet_ratio = float(src_packets) / (float(dst_packets) + 1.0)

        protocol_value = str(getattr(flow, "protocol", "") or "").strip().lower()
        is_tcp = 1 if protocol_value in ("6", "tcp") else 0
        is_udp = 1 if protocol_value in ("17", "udp") else 0

        iat_mean = getattr(flow, "bidirectional_mean_piat_ms", None)
        if iat_mean is None:
            iat_mean = getattr(flow, "bidirectional_avg_piat_ms", None)
        iat_std = getattr(flow, "bidirectional_stddev_piat_ms", None)
        if iat_std is None:
            iat_std = getattr(flow, "bidirectional_var_piat_ms", None)

        iat_values = [float(v) for v in getattr(flow.udps, "iat_list", []) if v is not None]
        if (iat_mean is None or float(iat_mean) == 0.0) and iat_values:
            iat_mean = sum(iat_values) / float(len(iat_values))
        if (iat_std is None or float(iat_std) == 0.0) and len(iat_values) > 1:
            mean_val = sum(iat_values) / float(len(iat_values))
            iat_std = (sum((v - mean_val) ** 2 for v in iat_values) / float(len(iat_values))) ** 0.5

        size_values = [abs(int(v)) for v in getattr(flow.udps, "size_list", []) if v is not None]

        mean_ps = getattr(flow, "bidirectional_mean_ps", None)
        if mean_ps is None or float(mean_ps) == 0.0:
            mean_ps = float(total_bytes) / float(total_packets) if total_packets > 0 else 0.0

        std_ps = getattr(flow, "bidirectional_stddev_ps", None)
        if std_ps is None and len(size_values) > 1:
            mean_size = sum(size_values) / float(len(size_values))
            std_ps = (sum((v - mean_size) ** 2 for v in size_values) / float(len(size_values))) ** 0.5
        if std_ps is None:
            std_ps = 0.0

        min_ps = getattr(flow, "bidirectional_min_ps", None)
        if min_ps is None and size_values:
            min_ps = min(size_values)
        if min_ps is None:
            min_ps = 0.0

        max_ps = getattr(flow, "bidirectional_max_ps", None)
        if max_ps is None and size_values:
            max_ps = max(size_values)
        if max_ps is None:
            max_ps = 0.0

        psh_count = getattr(flow, "bidirectional_psh_packets", None)
        if psh_count is None:
            psh_count = 0

        fin_count = getattr(flow, "bidirectional_fin_packets", None)
        if fin_count is None:
            fin_count = 0

        window_mean = getattr(flow, "bidirectional_mean_tcp_win", None)
        if window_mean is None:
            window_mean = getattr(flow, "bidirectional_mean_tcp_window", None)
        if window_mean is None:
            src_win = getattr(flow, "src2dst_mean_tcp_win", None)
            dst_win = getattr(flow, "dst2src_mean_tcp_win", None)
            values = [v for v in (src_win, dst_win) if v is not None]
            window_mean = sum(values) / float(len(values)) if values else 0.0

        duration_ms = float(getattr(flow, "bidirectional_duration_ms", 0.0) or 0.0)
        duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 0.0
        packets_per_sec = float(total_packets) / duration_sec if duration_sec > 0 else 0.0
        bytes_per_sec = float(total_bytes) / duration_sec if duration_sec > 0 else 0.0

        global_stats = {
            "duration_ms": float(duration_ms),
            "total_packets": int(total_packets),
            "total_bytes": int(total_bytes),
            "src2dst_bytes": int(src_bytes),
            "dst2src_bytes": int(dst_bytes),
            "src2dst_packets": int(src_packets),
            "dst2src_packets": int(dst_packets),
            "packet_ratio": float(packet_ratio),
            "byte_ratio": float(byte_ratio),
            "mean_ps": float(mean_ps),
            "std_ps": float(std_ps),
            "min_ps": float(min_ps),
            "max_ps": float(max_ps),
            "iat_mean": float(iat_mean or 0.0),
            "iat_std": float(iat_std or 0.0),
            "syn_count": int(getattr(flow, "bidirectional_syn_packets", 0) or 0),
            "rst_count": int(getattr(flow, "bidirectional_rst_packets", 0) or 0),
            "psh_count": int(psh_count or 0),
            "fin_count": int(fin_count or 0),
            "window_mean": float(window_mean or 0.0),
            "packets_per_sec": float(packets_per_sec),
            "bytes_per_sec": float(bytes_per_sec),
            "is_tcp": int(is_tcp),
            "is_udp": int(is_udp),
        }

        # 3. 组装最终对象
        yield ExtractedFlow(
            byte_matrix=flow.udps.byte_matrix.copy(),
            iat_series=list(flow.udps.iat_list),   # 转成纯 list
            size_series=list(flow.udps.size_list), # 转成纯 list
            global_stats=global_stats,
            src_ip=getattr(flow, "src_ip", "0.0.0.0"),
            dst_ip=getattr(flow, "dst_ip", "0.0.0.0"),
            dst_port=int(getattr(flow, "dst_port", 0) or 0),
            protocol=str(getattr(flow, "protocol", "")),
            timestamp=timestamp,
        )


def extract_pcap(
    pcap_path: str | Path,
    bytes_pkts: int = 8,
    bytes_len: int = 64,
    seq_len: int = 50,
    idle_timeout: float | int | None = None,
    active_timeout: float | int | None = None,
) -> List[ExtractedFlow]:
    return list(
        iter_pcap(
            pcap_path,
            bytes_pkts=bytes_pkts,
            bytes_len=bytes_len,
            seq_len=seq_len,
            idle_timeout=idle_timeout,
            active_timeout=active_timeout,
        )
    )


__all__ = ["ExtractedFlow", "ETALogicExtractor", "iter_pcap", "extract_pcap"]