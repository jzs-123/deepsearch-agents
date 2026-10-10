"""
进程内指标与健康检查模块

用线程安全的计数器/计时器统计任务执行情况,提供 Prometheus 文本格式输出。
适合单实例/教学场景;多副本部署时建议替换为 OpenTelemetry + 指标网关。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator


class Metrics:
    """线程安全的进程内指标收集器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        # 耗时直方图:sum / count / min / max
        self._hist: dict[str, dict[str, float]] = {}

    # ─── 计数器 ────────────────────────────────────────────

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            h = self._hist.setdefault(name, {"sum": 0.0, "count": 0, "min": float("inf"), "max": 0.0})
            h["sum"] += seconds
            h["count"] += 1
            h["min"] = min(h["min"], seconds)
            h["max"] = max(h["max"], seconds)

    # ─── 渲染 ──────────────────────────────────────────────

    def render_prometheus(self) -> str:
        """输出 Prometheus 文本格式。"""
        with self._lock:
            lines: list[str] = []
            for name, value in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, value in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            for name, h in self._hist.items():
                lines.append(f"# TYPE {name}_seconds summary")
                lines.append(f"{name}_seconds_count {h['count']}")
                lines.append(f"{name}_seconds_sum {h['sum']}")
                if h["count"]:
                    lines.append(f"{name}_seconds_min {h['min']}")
                    lines.append(f"{name}_seconds_max {h['max']}")
            return "\n".join(lines) + ("\n" if lines else "")

    def snapshot(self) -> dict:
        """以字典形式返回当前指标,供 /api/health 使用。"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: dict(v) for k, v in self._hist.items()},
            }


metrics = Metrics()


@contextmanager
def task_timer(metric_name: str = "task_duration") -> Iterator[None]:
    """计时上下文管理器:进入时记活跃数,退出时记录耗时。"""
    metrics.incr("tasks_started_total")
    metrics.gauge("tasks_active", _active_value() + 1)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics.observe(metric_name, elapsed)
        metrics.incr("tasks_completed_total")
        metrics.gauge("tasks_active", max(_active_value() - 1, 0))


def mark_task_failed() -> None:
    metrics.incr("tasks_failed_total")


def mark_task_cancelled() -> None:
    metrics.incr("tasks_cancelled_total")


def _active_value() -> int:
    """读取当前活跃任务数(近似值,供 gauge 计算用)。"""
    snap = metrics._gauges.get("tasks_active")
    return int(snap) if isinstance(snap, (int, float)) else 0
