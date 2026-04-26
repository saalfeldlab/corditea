"""LSD-as-service: a long-lived process holding the JAX/GPU context.

Phase 2 (Plan B) for the lsd-jax backend. Many gunpowder PreCache workers
submit segmentations to a single dedicated service process that does the
GPU compute. The service is started in the main process before workers fork
so all workers share one JAX/CUDA context — the only way to use GPU JAX
under multi-worker prefetch without N CUDA contexts on one card.

Inter-process protocol:
- Service holds a `mp.Queue` for incoming requests.
- Each request is `(segmentation, sigma, voxel_size, max_labels, channels,
  mode, send_end)` where ``send_end`` is the writable end of a one-shot
  ``mp.Pipe`` the worker created. Service writes the descriptor back through
  it and closes.
- Workers wait on ``recv_end`` of their pipe — concurrent requests from
  different workers don't interfere because each waits on its own pipe.

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

import numpy as np

logger = logging.getLogger(__name__)


_service: "LsdService | None" = None
_service_lock = threading.Lock()


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
        logger.info("lsd-service: started on physical GPU %d (pid %d), waiting for ready signal", gpu_index, self.process.pid)
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
        """Submit a request and block on the dedicated pipe for the result."""
        recv_end, send_end = self._ctx.Pipe(duplex=False)
        try:
            self.request_queue.put(
                (segmentation, float(sigma), int(voxel_size), int(max_labels), int(channels), mode, send_end)
            )
            # Don't close send_end before recv: Queue.put() is async (a
            # feeder thread serializes it later) and closing the FD before
            # serialization happens raises EBADF in the feeder.
            result = recv_end.recv()
        finally:
            try:
                send_end.close()
            except Exception:  # noqa: BLE001
                pass
            recv_end.close()
        if isinstance(result, BaseException):
            raise RuntimeError("lsd-service request failed") from result
        return result

    def shutdown(self) -> None:
        try:
            self.request_queue.put(None)
        except Exception:  # noqa: BLE001
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()


def _service_loop(gpu_index: int, request_queue, ready_send) -> None:  # runs in the service process
    # Configure logging early so any startup error is visible.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log = logging.getLogger("corditea._lsd_service.worker")

    try:
        # Restrict CUDA visibility before importing JAX so we initialize
        # exactly one context, on the intended physical GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        # Module-level import inside the spawned process — fresh interpreter.
        import jax
        import jax.numpy as jnp
        from lsd_jax import vectorized_LSD

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

    while True:
        msg = request_queue.get()
        if msg is None:
            log.info("lsd-service: shutdown sentinel received")
            return
        segmentation, sigma, voxel_size, max_labels, channels, mode, send_end = msg
        try:
            seg_jax = jnp.asarray(segmentation)[None, ...]  # (1, *spatial)
            out = vectorized_LSD(seg_jax, sigma, max_labels, voxel_size, mode, channels)
            out_np = np.asarray(out[0], dtype=np.float32)
            send_end.send(out_np)
        except BaseException as e:  # noqa: BLE001
            log.exception("lsd-service: error processing request")
            try:
                send_end.send(e)
            except Exception:  # noqa: BLE001
                pass
        finally:
            send_end.close()
