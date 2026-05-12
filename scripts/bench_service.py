"""Benchmark the LSD service under production-like load.

Spawns N "worker" processes (forking, like PreCache does) and has each call
``service.compute`` C times back-to-back. Measures per-call wall time and
aggregate throughput.

Defaults match the run23 setup: 20 workers, 8 calls each (= bs=8), production
shape, sigma=30, voxel_size=8.

Run from anywhere that has corditea + lsd-jax installed:
    pixi run python scripts/bench_service.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import List, Tuple

import numpy as np

from corditea._lsd_backends import _next_power_of_two
from corditea._lsd_service import _shm_pool_cleanup, ensure_service_started, get_service


# --- config ------------------------------------------------------------
N_WORKERS = int(os.environ.get("BENCH_WORKERS", "20"))
CALLS_PER_BATCH = int(os.environ.get("BENCH_CALLS_PER_BATCH", "8"))   # = batch_size
N_BATCHES = int(os.environ.get("BENCH_BATCHES", "5"))
SHAPE = (237, 237, 237)            # match production seg input size
SIGMA = 30.0
VOXEL_SIZE = 8
CHANNELS = 10
# Lognormal label-count distribution: median ≈ 5, P(X > 1000) ≈ 1%.
N_LABELS_MU = float(np.log(5.0))
N_LABELS_SIGMA = 2.28
N_LABELS_MAX = 4096                 # cap so a rogue tail draw can't request a 32k-bucket compile
# -----------------------------------------------------------------------


def _sample_n_labels(rng: np.random.Generator) -> int:
    n = int(round(rng.lognormal(N_LABELS_MU, N_LABELS_SIGMA)))
    return max(1, min(n, N_LABELS_MAX))


def _make_seg(shape, n_labels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, n_labels + 1, size=shape, dtype=np.int32)


def _gen_worker_specs(worker_id: int) -> List[Tuple[np.ndarray, int]]:
    """Deterministic per-worker seg + max_labels list. Used by both the
    pre-warmup phase (in main) and the worker's actual run."""
    sampler_rng = np.random.default_rng(worker_id)
    seg_specs: List[Tuple[np.ndarray, int]] = []
    for b in range(N_BATCHES):
        n_labels = _sample_n_labels(sampler_rng)
        seg = _make_seg(SHAPE, n_labels, seed=worker_id * 1000 + b)
        n_distinct = int(np.unique(seg).size)
        max_labels = _next_power_of_two(n_distinct + 1)
        seg_specs.append((seg, max_labels))
    return seg_specs


def _worker(
    worker_id: int,
    n_calls: int,
    result_queue: "mp.Queue",
) -> None:
    """Forked worker: pre-generates segs, then hammers service.compute."""
    service = get_service()
    if service is None:
        result_queue.put((worker_id, [], [], "service not running in worker"))
        return
    seg_specs = _gen_worker_specs(worker_id)

    times: List[float] = []
    err: str | None = None
    try:
        for call_idx in range(n_calls):
            seg, max_labels = seg_specs[call_idx // CALLS_PER_BATCH]
            t0 = time.perf_counter()
            out = service.compute(seg, SIGMA, VOXEL_SIZE, max_labels, CHANNELS, "gaussian")
            t1 = time.perf_counter()
            # Touch the result so we can't be optimized away.
            _ = out.shape
            times.append(t1 - t0)
    except BaseException as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        # Belt-and-suspenders: explicitly unlink pooled SHMs before returning so
        # the resource_tracker doesn't warn about leaked blocks. atexit usually
        # handles this for forked processes but not always.
        _shm_pool_cleanup()
    result_queue.put((worker_id, times, [m for _, m in seg_specs], err))


def main() -> None:
    mp.set_start_method("fork", force=True)

    print(f"bench config: {N_WORKERS} workers x {CALLS_PER_BATCH * N_BATCHES} calls (= {N_BATCHES} batches of {CALLS_PER_BATCH})")
    print(f"shape={SHAPE} sigma={SIGMA} voxel_size={VOXEL_SIZE} max_labels=lognormal(mu=ln5, sigma={N_LABELS_SIGMA}) -> next_pow2")
    print(f"LSD_JAX_GPU_INDEX={os.environ.get('LSD_JAX_GPU_INDEX', '<unset>')}")
    print()

    # Start the service in main; forks below will inherit it.
    ensure_service_started()

    # --- pre-warm JIT cache ---------------------------------------------
    # Generate every worker's seg specs, find the union of buckets, and pre-warm
    # one compile per bucket from main. Workers in Phase 1/2 will then hit the
    # cached JIT and we measure steady-state throughput rather than compile time.
    print("Pre-warm: gathering buckets across all worker specs")
    all_specs: List[Tuple[np.ndarray, int]] = []
    for wid in range(N_WORKERS):
        all_specs.extend(_gen_worker_specs(wid))
    bucket_to_seg: dict[int, np.ndarray] = {}
    for seg, ml in all_specs:
        bucket_to_seg.setdefault(ml, seg)  # one representative seg per bucket
    buckets_sorted = sorted(bucket_to_seg)
    print(f"  unique buckets to warm: {buckets_sorted}")
    service = get_service()
    if service is None:
        print("  ERROR: service not running in main")
        return
    t_warm_start = time.perf_counter()
    for ml in buckets_sorted:
        seg = bucket_to_seg[ml]
        t0 = time.perf_counter()
        _ = service.compute(seg, SIGMA, VOXEL_SIZE, ml, CHANNELS, "gaussian")
        print(f"  bucket {ml}: {(time.perf_counter() - t0)*1000:.0f}ms")
    print(f"  total warmup: {time.perf_counter() - t_warm_start:.2f}s")
    print()

    # --- single-worker baseline (no contention) -------------------------
    print("Phase 1: single worker, no contention")
    rq = mp.Queue()
    p = mp.Process(target=_worker, args=(0, CALLS_PER_BATCH * N_BATCHES, rq))
    t_start = time.perf_counter()
    p.start()
    wid, single_times, single_buckets, err = rq.get()
    p.join()
    t_solo = time.perf_counter() - t_start
    if err:
        print(f"  worker error: {err}")
        return
    single_arr = np.asarray(single_times)
    # Skip the first call (likely a JIT compile) for the steady-state percentiles.
    steady = single_arr[1:] if len(single_arr) > 1 else single_arr
    print(f"  total wall: {t_solo:.2f}s  ({len(single_arr)} calls)")
    print(f"  first call: {single_arr[0]*1000:.1f}ms (likely includes compile)")
    print(f"  steady mean: {steady.mean()*1000:.1f}ms  median: {np.median(steady)*1000:.1f}ms  p95: {np.percentile(steady, 95)*1000:.1f}ms")
    print(f"  buckets sampled: {sorted(set(single_buckets))}")
    print()

    # --- N workers contending ------------------------------------------
    print(f"Phase 2: {N_WORKERS} workers contending")
    rq = mp.Queue()
    procs = []
    n_calls_per_worker = CALLS_PER_BATCH * N_BATCHES
    t_start = time.perf_counter()
    for wid in range(N_WORKERS):
        p = mp.Process(target=_worker, args=(wid, n_calls_per_worker, rq))
        p.start()
        procs.append(p)
    all_times: List[float] = []
    all_buckets: List[int] = []
    errors: List[Tuple[int, str]] = []
    for _ in range(N_WORKERS):
        wid, times, buckets, err = rq.get()
        if err:
            errors.append((wid, err))
        all_times.extend(times)
        all_buckets.extend(buckets)
    for p in procs:
        p.join()
    t_total = time.perf_counter() - t_start

    if errors:
        for wid, err in errors:
            print(f"  worker {wid} error: {err}")

    arr = np.asarray(all_times)
    n = len(arr)
    print(f"  total wall: {t_total:.2f}s")
    print(f"  total calls: {n}")
    print(f"  aggregate throughput: {n / t_total:.1f} calls/sec")
    print(f"  per-call mean (incl warmup): {arr.mean()*1000:.0f}ms  median: {np.median(arr)*1000:.0f}ms  p95: {np.percentile(arr, 95)*1000:.0f}ms  max: {arr.max()*1000:.0f}ms")
    bucket_arr = np.asarray(all_buckets)
    uniq, counts = np.unique(bucket_arr, return_counts=True)
    hist = "  ".join(f"{int(b)}:{int(c)}" for b, c in zip(uniq, counts))
    print(f"  bucket histogram (max_labels:count across {len(bucket_arr)} segs): {hist}")
    print()

    # Production rate target: ~160 calls per training iter at 14s/iter = ~11.4 calls/sec
    target_rate = (N_WORKERS * CALLS_PER_BATCH) / 14.0
    print(f"production target: ~{target_rate:.1f} calls/sec (160 calls per ~14s training iter)")
    print(f"this run achieved: {n / t_total:.1f} calls/sec ({(n / t_total) / target_rate:.2f}x of target)")


if __name__ == "__main__":
    main()
