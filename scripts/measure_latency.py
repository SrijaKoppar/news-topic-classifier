"""Measure real single-request latency through the /predict endpoint (in-process).

Uses validation-set TEXT only (no labels), so nothing is evaluated or tuned here.
Usage:  python -m scripts.measure_latency
"""
import json
import statistics
import time

from fastapi.testclient import TestClient

from src.api import app
from src.config import RESULTS_DIR
from src.data import load_raw, make_split

N_WARMUP, N_TIMED = 20, 300

sd = make_split(load_raw(), "split_a")
rows = sd.val.head(N_WARMUP + N_TIMED)
with TestClient(app) as client:
    times = []
    for i, (h, d) in enumerate(zip(rows["headline"], rows["short_description"])):
        payload = {"headline": h, "short_description": d if isinstance(d, str) else None}
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        dt = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, r.text
        if i >= N_WARMUP:
            times.append(dt)

times.sort()
out = {
    "requests_timed": len(times),
    "mean_ms": round(statistics.mean(times), 1),
    "p50_ms": round(times[len(times) // 2], 1),
    "p95_ms": round(times[int(len(times) * 0.95)], 1),
    "cpu_seconds_per_1000_predictions": round(statistics.mean(times), 1),
    "cost_usd_per_1000_predictions": 0.0,
    "note": "local CPU, in-process (no network); cost is $0 apart from local compute",
}
(RESULTS_DIR / "serving_latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))