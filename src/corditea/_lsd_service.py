"""LSD-as-service: a long-lived process holding the JAX/GPU context.

Phase 2 (Plan B) for the lsd-jax backend. Many gunpowder PreCache workers
submit segmentations to a single dedicated service process that does the
GPU compute. The service is started in the main process before workers fork
so all workers share one JAX/CUDA context — the only way to use GPU JAX
under multi-worker prefetch without N CUDA contexts on one card.

Inter-process protocol (shared-memory variant):
- Worker allocates two ``SharedMemory`` blocks per call (input segmentation
  + output descriptor) plus a one-shot ``mp.Pipe`` for status return.
- Worker stages the segmentation into the input block, then puts a small
  request descriptor (block names, shapes, dtypes, params, status pipe) on
  the service's request queue.
- Service attaches both blocks (zero-copy numpy views), runs LSD on GPU,
  writes the result into the output block, and sends ``"ok"`` (or the
  exception) through the status pipe.
- Worker reads the status, copies the result out of shared memory, frees
  both blocks. Concurrent workers don't interfere — each owns its own
  SharedMemory + status pipe.

This avoids pickling ~80 MB float32 descriptors back through ``mp.Pipe``
on every batch.

Service lifecycle:
- ``ensure_service_started()`` is idempotent and only starts the service when
  called from the main process. Call sites: AddLSD.__init__ for backend
  ``lsd-jax``, so pipeline construction in main starts it before fork.
- atexit hook tears it down on main exit.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import threading
from multiprocessing import shared_memory

import numpy as np

logger = logging.getLogger(__name__)


_service: "LsdService | None" = None
_service_lock = threading.Lock()


# Per-worker SHM pool. Each call to LsdService.compute() needs two SharedMemory
# blocks (input + output). Allocating them via shm_open + ftruncate every call
# is ~5-10 ms each on /dev/shm; pooling by size lets repeated calls of the
# same shape skip that work entirely. Lifecycle: blocks are returned to the
# pool after the call completes; an atexit handler unlinks them on exit.
_shm_pool: dict[int, list] = {}
_shm_pool_lock = threading.Lock()
_shm_pool_atexit_registered = False
_shm_pool_owner_pid: int | None = None


def _acquire_shm(size: int):
    global _shm_pool_owner_pid
    pid = os.getpid()
    with _shm_pool_lock:
        if _shm_pool_owner_pid != pid:
            # Forked from a process that had pool entries — drop those
            # references; the SHMs they point to may have been unlinked by the
            # parent or another sibling. Each PID maintains its own pool.
            _shm_pool.clear()
            _shm_pool_owner_pid = pid
        bucket = _shm_pool.get(size)
        if bucket:
            return bucket.pop()
    return shared_memory.SharedMemory(create=True, size=size)


def _release_shm(shm) -> None:
    with _shm_pool_lock:
        _shm_pool.setdefault(shm.size, []).append(shm)


def _shm_pool_cleanup() -> None:
    with _shm_pool_lock:
        for bucket in _shm_pool.values():
            for shm in bucket:
                try:
                    shm.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass
        _shm_pool.clear()


def ensure_service_started() -> None:
    """Start the LSD service singleton if not running. No-op when called from
    a non-main process (workers inherit the singleton via fork)."""
    if mp.current_process().name != "MainProcess":
        return
    global _service
    with _service_lock:
        if _service is not None:
            return
        gpu_index = int(os.environ.get("LSD_JAX_GPU_INDEX", "1"))
        _service = LsdService(gpu_index=gpu_index)
        atexit.register(_service.shutdown)


def get_service() -> "LsdService | None":
    """Return the running service singleton, or None if not started."""
    return _service


class LsdService:
    def __init__(self, gpu_index: int):
        # Spawn (not fork) so the service starts a fresh Python interpreter
        # without inheriting any CUDA state main may have touched. That's the
        # whole point — we want the service to be the sole JAX-on-GPU process.
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self.request_queue: mp.Queue = ctx.Queue()
        # Handshake pipe so we can detect a startup crash (JAX init failure,
        # missing CUDA, etc.) instead of getting silent EOFErrors later.
        ready_recv, ready_send = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=_service_loop,
            args=(gpu_index, self.request_queue, ready_send),
            daemon=True,
            name="lsd-jax-service",
        )
        self.process.start()
        # Parent doesn't write to ready_send.
        ready_send.close()
        # Resolve LSD_JAX_GPU_INDEX into the actual GPU id (numeric index or
        # UUID) the service will pin to, so the log line is accurate even
        # under SLURM/LSF allocations that hand out GPUs in arbitrary order.
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if parent_visible:
            items = [item.strip() for item in parent_visible.split(",") if item.strip()]
            resolved = items[gpu_index] if 0 <= gpu_index < len(items) else f"<index {gpu_index} out of range>"
        else:
            resolved = str(gpu_index)
        logger.info(
            "lsd-service: started; pinning to GPU %s (LSD_JAX_GPU_INDEX=%d, pid %d), waiting for ready signal",
            resolved,
            gpu_index,
            self.process.pid,
        )
        # Block up to 60s for the service to confirm JAX init succeeded.
        if ready_recv.poll(timeout=60.0):
            status = ready_recv.recv()
            ready_recv.close()
            if isinstance(status, BaseException):
                raise RuntimeError("lsd-service failed to start") from status
            logger.info("lsd-service: ready (JAX initialized)")
        else:
            ready_recv.close()
            if not self.process.is_alive():
                msg = (
                    "lsd-service died during startup before signaling ready; "
                    "check stderr for the JAX/CUDA initialization error."
                )
            else:
                msg = "lsd-service did not signal ready within 60s"
            raise RuntimeError(msg)

    def compute(
        self,
        segmentation: np.ndarray,
        sigma: float,
        voxel_size: int,
        max_labels: int,
        channels: int,
        mode: str = "gaussian",
    ) -> np.ndarray:
        """Stage segmentation into shared memory, dispatch, copy result out."""
        global _shm_pool_atexit_registered
        if not _shm_pool_atexit_registered:
            atexit.register(_shm_pool_cleanup)
            _shm_pool_atexit_registered = True

        seg = np.ascontiguousarray(segmentation)
        out_shape = (channels,) + seg.shape
        out_nbytes = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize

        in_shm = _acquire_shm(int(seg.nbytes))
        out_shm = _acquire_shm(out_nbytes)
        recv_end, send_end = self._ctx.Pipe(duplex=False)
        try:
            in_view = np.ndarray(seg.shape, dtype=seg.dtype, buffer=in_shm.buf)
            in_view[:] = seg

            self.request_queue.put(
                (
                    in_shm.name,
                    seg.shape,
                    str(seg.dtype),
                    out_shm.name,
                    out_shape,
                    float(sigma),
                    int(voxel_size),
                    int(max_labels),
                    int(channels),
                    mode,
                    send_end,
                )
            )
            # Don't close send_end before recv: Queue.put() is async (a
            # feeder thread serializes it later) and closing the FD before
            # serialization happens raises EBADF in the feeder.
            status = recv_end.recv()
            if isinstance(status, BaseException):
                raise RuntimeError("lsd-service request failed") from status
            out_view = np.ndarray(out_shape, dtype=np.float32, buffer=out_shm.buf)
            return np.array(out_view, copy=True)
        finally:
            try:
                send_end.close()
            except Exception:  # noqa: BLE001
                pass
            recv_end.close()
            # Return SHMs to the pool instead of unlinking. Service-side already
            # closed its handles; the SHMs are now free for the next call.
            _release_shm(in_shm)
            _release_shm(out_shm)

    def shutdown(self) -> None:
        try:
            self.request_queue.put(None)
        except Exception:  # noqa: BLE001
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()


def _service_loop(gpu_index: int, request_queue, ready_send) -> None:  # runs in the service process
    # Configure logging early so any startup error is visible. ``force=True``
    # is required because something we import later (JAX/absl) installs root
    # handlers, which would otherwise make basicConfig a no-op and silently
    # drop INFO-level messages via Python's WARNING-only lastResort handler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    log = logging.getLogger("corditea._lsd_service.worker")

    try:
        # Restrict CUDA visibility before importing JAX so we initialize
        # exactly one context, on the intended GPU. ``LSD_JAX_GPU_INDEX`` is
        # an index into the *parent's* CUDA_VISIBLE_DEVICES list, so we work
        # correctly under SLURM/LSF allocations that hand out GPUs in
        # arbitrary order (e.g. "1,0") or as UUIDs.
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if parent_visible:
            items = [item.strip() for item in parent_visible.split(",") if item.strip()]
            if gpu_index >= len(items):
                msg = (
                    f"LSD_JAX_GPU_INDEX={gpu_index} is out of range; "
                    f"parent CUDA_VISIBLE_DEVICES has {len(items)} entries: {items!r}"
                )
                raise RuntimeError(msg)
            os.environ["CUDA_VISIBLE_DEVICES"] = items[gpu_index]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        # Module-level import inside the spawned process — fresh interpreter.
        import jax
        import jax.numpy as jnp

        try:
            from lsd_jax import compute_LSD
        except ImportError as exc:
            msg = (
                "The 'lsd-jax' LSD backend requires the lsd-jax package, which "
                "is not installed. It ships as an optional extra (fork of a "
                "private repo). Install with `pip install 'corditea[jax]'`, or "
                "use backend='lsd-lite'."
            )
            raise ImportError(msg) from exc

        # Wrap compute_LSD in jit ourselves: upstream lsd_jax has the @jit on
        # compute_LSD commented out (only vectorized_LSD is jit'd). Bench showed
        # jit'd compute_LSD is 5-7x faster than vectorized_LSD with bs=1, because
        # vectorized_LSD's vmap rewrites lax.cond -> select, defeating the
        # padding-skip optimization in iterate_over_label_statistics.
        compute_LSD_jit = jax.jit(compute_LSD, static_argnums=(1, 2, 3, 4, 5))

        gpus = jax.devices("gpu") if "cpu" not in jax.default_backend() else []
        log.info("lsd-service: JAX initialized; default backend=%s, gpus=%s", jax.default_backend(), gpus)
    except BaseException as e:
        log.exception("lsd-service: startup failed")
        try:
            ready_send.send(e)
        finally:
            ready_send.close()
        return

    try:
        ready_send.send("ready")
    finally:
        ready_send.close()

    seen_buckets: set[int] = set()
    while True:
        msg = request_queue.get()
        if msg is None:
            log.info("lsd-service: shutdown sentinel received")
            return
        (
            in_name,
            in_shape,
            in_dtype,
            out_name,
            out_shape,
            sigma,
            voxel_size,
            max_labels,
            channels,
            mode,
            send_end,
        ) = msg
        if max_labels not in seen_buckets:
            log.info(
                "lsd-service: JIT-compiling for new bucket max_labels=%d (this call will stall for several seconds)",
                max_labels,
            )
            seen_buckets.add(max_labels)
        in_shm = None
        out_shm = None
        try:
            in_shm = shared_memory.SharedMemory(name=in_name)
            out_shm = shared_memory.SharedMemory(name=out_name)
            seg = np.ndarray(in_shape, dtype=np.dtype(in_dtype), buffer=in_shm.buf)
            seg_jax = jnp.asarray(seg)
            out = compute_LSD_jit(seg_jax, sigma, max_labels, voxel_size, mode, channels)
            out_view = np.ndarray(out_shape, dtype=np.float32, buffer=out_shm.buf)
            np.copyto(out_view, np.asarray(out, dtype=np.float32))
            send_end.send("ok")
        except BaseException as e:  # noqa: BLE001
            log.exception("lsd-service: error processing request")
            try:
                send_end.send(e)
            except Exception:  # noqa: BLE001
                pass
        finally:
            if in_shm is not None:
                in_shm.close()
            if out_shm is not None:
                out_shm.close()
            send_end.close()
