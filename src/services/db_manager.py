from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class FlowRecord(Base):
    __tablename__ = "flow_records"

    # === 1. 基础信息 ===
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    src_ip = Column(String(64), nullable=False)
    dst_ip = Column(String(64), nullable=False)
    dst_port = Column(Integer, nullable=False)
    protocol = Column(String(32), nullable=False)

    # === 2. 模型判定结果 (AI 输出) ===
    # 这些是模型算出来的，如果只是单纯采集数据，可以设为默认值
    recon_loss = Column(Float, nullable=True) # 允许为空，方便采集原始数据

    
    z_score = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, nullable=False, default=False, index=True)

    # === 3. ETA 特征数据 (用于 RCA 分析和重训练) ===
    
    # [A] 视觉特征: 8x64 的头部字节矩阵
    # 存: [[0, 255, ...], [1, ...]]
    raw_bytes_preview = Column(JSONB, nullable=False)

    # [B] 时序特征: IAT 序列 (List[float])
    # 存: [0.1, 120.5, 0.0, ...]
    iat_series = Column(JSONB, nullable=True)

    # [C] 交互特征: 包大小序列 (List[int])
    # 存: [+1500, -60, +120, ...]
    size_series = Column(JSONB, nullable=True)

    # [D] 全局统计: 宏观行为 (Dict)
    # 存: {"duration": 500, "byte_ratio": 1.2, "iat_std": 0.0}
    global_stats = Column(JSONB, nullable=True)


class TrainingFlow(Base):
    __tablename__ = "training_flows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    src_ip = Column(String(64), nullable=False)
    dst_ip = Column(String(64), nullable=False)
    dst_port = Column(Integer, nullable=False)
    protocol = Column(String(32), nullable=False)

    raw_bytes = Column(JSONB, nullable=False)
    iat_series = Column(JSONB, nullable=True)
    size_series = Column(JSONB, nullable=True)
    global_stats = Column(JSONB, nullable=True)


@dataclass
class FlowRecordCreate:
    """数据传输对象 (DTO)"""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    
    # 特征数据 (必须是 Python 原生类型: List/Dict, 不能是 numpy)
    raw_bytes_preview: List[List[int]]  # 对应 8x64 矩阵
    iat_series: List[float]             # 对应 IAT 序列
    size_series: List[int]              # 对应 Size 序列
    global_stats: Dict[str, Any]        # 对应 全局统计
    
    # 模型结果 (可选，默认为 0/False)
    recon_loss: float = 0.0
    z_score: float = 0.0
    is_anomaly: bool = False


@dataclass
class TrainingFlowCreate:
    timestamp: datetime
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str

    raw_bytes: List[List[int]]
    iat_series: List[float]
    size_series: List[int]
    global_stats: Dict[str, Any]


class DBManager:
    def __init__(self, db_url: str | None = None) -> None:
        if db_url is None:
            # 默认连接 Docker 里的 Postgres
            db_url = "postgresql://admin:123456@localhost:5432/network_flow_db"
            
        self.engine = create_engine(db_url, pool_size=20, max_overflow=0) # 增加连接池配置
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Iterable[Session]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def insert_flow(self, record: FlowRecordCreate) -> FlowRecord:
        """单条插入 (测试用)"""
        payload = FlowRecord(
            timestamp=record.timestamp,
            src_ip=record.src_ip,
            dst_ip=record.dst_ip,
            dst_port=record.dst_port,
            protocol=record.protocol,
            recon_loss=record.recon_loss,
            z_score=record.z_score,
            is_anomaly=record.is_anomaly,
            # 特征字段
            raw_bytes_preview=record.raw_bytes_preview,
            iat_series=record.iat_series,
            size_series=record.size_series,
            global_stats=record.global_stats,
        )
        with self.session_scope() as session:
            session.add(payload)
            session.flush()
            session.refresh(payload)
            session.expunge(payload)
            return payload

    def bulk_insert(self, records: List[FlowRecordCreate]) -> None:
        """
        [新增] 批量插入 (生产环境/线上模式用)
        比一条条 insert 快 10-50 倍
        """
        if not records:
            return

        # 将 DTO 列表转换为 ORM 对象列表
        orm_objects = [
            FlowRecord(
                timestamp=r.timestamp,
                src_ip=r.src_ip,
                dst_ip=r.dst_ip,
                dst_port=r.dst_port,
                protocol=r.protocol,
                recon_loss=r.recon_loss,
                z_score=r.z_score,
                is_anomaly=r.is_anomaly,
                # 特征字段
                raw_bytes_preview=r.raw_bytes_preview,
                iat_series=r.iat_series,
                size_series=r.size_series,
                global_stats=r.global_stats,
            )
            for r in records
        ]

        with self.session_scope() as session:
            # 使用 SQLAlchemy 的批量保存方法
            session.bulk_save_objects(orm_objects)

    def bulk_insert_training(self, records: List[TrainingFlowCreate]) -> None:
        if not records:
            return

        orm_objects = [
            TrainingFlow(
                timestamp=r.timestamp,
                src_ip=r.src_ip,
                dst_ip=r.dst_ip,
                dst_port=r.dst_port,
                protocol=r.protocol,
                raw_bytes=r.raw_bytes,
                iat_series=r.iat_series,
                size_series=r.size_series,
                global_stats=r.global_stats,
            )
            for r in records
        ]

        with self.session_scope() as session:
            session.bulk_save_objects(orm_objects)

    def iter_training_flows(self, batch_size: int = 1000) -> Iterable[TrainingFlow]:
        session = self.SessionLocal()
        try:
            query = session.query(TrainingFlow).order_by(TrainingFlow.id).yield_per(batch_size)
            for row in query:
                yield row
        finally:
            session.close()

    def get_flows(self, limit: int = 50, offset: int = 0) -> List[Dict[str, object]]:
        with self.session_scope() as session:
            query = (
                session.query(FlowRecord)
                .order_by(FlowRecord.timestamp.desc())
                .offset(offset)
                .limit(limit)
            )
            results = []
            for flow in query:
                results.append(
                    {
                        "id": flow.id,
                        "timestamp": flow.timestamp.isoformat(),
                        "src_ip": flow.src_ip,
                        "dst_ip": flow.dst_ip,
                        "dst_port": flow.dst_port,
                        "protocol": flow.protocol,
                        "is_anomaly": flow.is_anomaly,
                        # 特征展示
                        "global_stats": flow.global_stats,
                        # 序列数据太长，通常 dashboard 只要取一部分预览，或者前端再通过 ID 单独查详情
                        # 这里为了演示先全返回
                        "iat_series_len": len(flow.iat_series) if flow.iat_series else 0,
                    }
                )
            return results

    def get_metrics(self, window_minutes: int = 60) -> Dict[str, float]:
        window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
        with self.session_scope() as session:
            total = (
                session.query(func.count(FlowRecord.id))
                .filter(FlowRecord.timestamp >= window_start)
                .scalar()
                or 0
            )
            anomalies = (
                session.query(func.count(FlowRecord.id))
                .filter(
                    FlowRecord.timestamp >= window_start,
                    FlowRecord.is_anomaly.is_(True),
                )
                .scalar()
                or 0
            )
        anomaly_rate = float(anomalies) / total if total else 0.0
        return {
            "window_minutes": float(window_minutes),
            "total_flows": float(total),
            "anomaly_count": float(anomalies),
            "anomaly_rate": anomaly_rate,
        }


__all__ = [
    "DBManager",
    "FlowRecordCreate",
    "FlowRecord",
    "TrainingFlow",
    "TrainingFlowCreate",
]
