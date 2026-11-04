"""Latency profiler computing statistically robust percentile distributions."""

import numpy as np
from typing import List, Dict, Any


class LatencyProfiler:
    """Profiles and records execution times across system pipeline stages."""

    def __init__(self):
        self.records: Dict[str, List[float]] = {
            "e2e": [],
            "retrieval": [],
            "reranking": [],
            "synthesis": [],
            "audit": [],
        }

    def record(self, stage: str, latency_ms: float) -> None:
        if stage not in self.records:
            self.records[stage] = []
        self.records[stage].append(latency_ms)

    def summary(self, stage: str = "e2e") -> Dict[str, float]:
        values = self.records.get(stage, [])
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}

        arr = np.array(values, dtype=np.float64)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "count": len(arr),
        }
