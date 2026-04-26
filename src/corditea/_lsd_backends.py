"""Dispatch layer between the lsd-lite (scipy) and lsd-jax (JAX) LSD backends.

Keeps AddLSD backend-agnostic. The public entry point is `compute_lsds`, which
matches the signature of the lsd-lite call site in AddLSD.process.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)


_jax_initialized = False


def _ensure_jax_initialized() -> None:
    """Lazily init JAX on first use. Restrict JAX to exactly the GPU indexed
    by ``$LSD_JAX_GPU_INDEX`` (default 1) via ``CUDA_VISIBLE_DEVICES`` so JAX
    does not create stray CUDA contexts on every visible GPU. Called from
    ``_compute_lsds_jax`` rather than at module import so corditea can be
    imported in main without touching CUDA; the ``CUDA_VISIBLE_DEVICES`` set
    here only affects this process (the PreCache worker), not main's PyTorch.
    """
    global _jax_initialized
    if _jax_initialized:
        return
    # Don't let JAX grab 75% of the GPU; leaves room for the LSD_JAX working
    # set (~2-3 GB for 256^3 crops) on top of its own JIT scratch.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    requested_index = int(os.environ.get("LSD_JAX_GPU_INDEX", "1"))
    # Limit CUDA's device enumeration before jax imports, so JAX initializes
    # a context only on the target GPU. Force (not setdefault) because the
    # user's shell env will typically have this unset (= all GPUs visible).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_index)
    import jax

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if len(gpus) == 1:
        # After CUDA_VISIBLE_DEVICES restriction, the one visible GPU is
        # logical index 0 in JAX's view; physically it's requested_index.
        jax.config.update("jax_default_device", gpus[0])
        logger.info("lsd-jax pinned to physical GPU %d (CUDA_VISIBLE_DEVICES)", requested_index)
    elif len(gpus) > 1:
        # CUDA_VISIBLE_DEVICES didn't take effect (e.g., CUDA was pre-init'd).
        # Fall back to jax_default_device; still leaks stray contexts.
        target = gpus[requested_index] if requested_index < len(gpus) else gpus[0]
        jax.config.update("jax_default_device", target)
        logger.warning(
            "lsd-jax saw %d GPUs despite CUDA_VISIBLE_DEVICES restriction; default device set "
            "but stray contexts may exist on other GPUs",
            len(gpus),
        )
    else:
        logger.info("lsd-jax using CPU (no GPU visible)")
    _jax_initialized = True


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 2
    return 1 << (n - 1).bit_length()


def _require_isotropic(voxel_size: Sequence[int] | None) -> int:
    if voxel_size is None:
        return 1
    vs = tuple(int(v) for v in voxel_size)
    if len(set(vs)) != 1:
        msg = (
            f"lsd-jax backend requires isotropic voxel_size; got {vs}. "
            "Use lsd-lite or extend LSD_JAX upstream for anisotropic support."
        )
        raise NotImplementedError(msg)
    return vs[0]


def _compute_lsds_lite(
    segmentation: np.ndarray,
    sigma,
    voxel_size,
    labels,
    downsample: int,
) -> np.ndarray:
    from lsd_lite import get_lsds

    return get_lsds(
        segmentation=segmentation,
        sigma=sigma,
        voxel_size=voxel_size,
        downsample=downsample,
        labels=labels,
    )


def _compute_lsds_jax(
    segmentation: np.ndarray,
    sigma,
    voxel_size,
    labels,
    downsample: int,
) -> np.ndarray:
    if downsample != 1:
        msg = "lsd-jax backend does not support downsample != 1"
        raise NotImplementedError(msg)
    iso_voxel = _require_isotropic(voxel_size)

    # Match lsd-lite's labels= semantics: zero out voxels whose values are not
    # in the target label list. LSD_JAX skips label 0 internally (via labels[1:]
    # on sorted unique), so zeroed voxels get no descriptor.
    seg = np.ascontiguousarray(segmentation)
    if labels is not None:
        keep = np.isin(seg, np.asarray(list(labels), dtype=seg.dtype))
        if not keep.all():
            seg = np.where(keep, seg, 0)

    n_distinct = int(np.unique(seg).size)
    # +1 headroom so a padding slot is always available
    max_labels = _next_power_of_two(n_distinct + 1)

    dims = len(seg.shape)
    channels = 10 if dims == 3 else 6

    # Sigma must be a scalar for LSD_JAX (isotropic only). AddLSD passes either
    # a float or a uniform tuple; collapse to scalar.
    sigma_scalar = float(sigma[0]) if isinstance(sigma, (tuple, list)) else float(sigma)

    # Plan B: if a service is running, route the call through it (one JAX
    # context for many gunpowder workers). Otherwise fall back to in-process
    # JAX — used by tests and by anyone who didn't bootstrap the service.
    from corditea._lsd_service import get_service

    service = get_service()
    if service is not None:
        return service.compute(seg, sigma_scalar, iso_voxel, max_labels, channels, "gaussian")

    _ensure_jax_initialized()
    import jax.numpy as jnp

    from lsd_jax import vectorized_LSD

    seg_jax = jnp.asarray(seg)[None, ...]  # (1, *spatial)
    out = vectorized_LSD(seg_jax, sigma_scalar, max_labels, iso_voxel, "gaussian", channels)
    return np.asarray(out[0], dtype=np.float32)


def compute_lsds(
    segmentation: np.ndarray,
    sigma,
    voxel_size,
    labels,
    downsample: int,
    *,
    backend: Literal["lsd-lite", "lsd-jax"] = "lsd-lite",
) -> np.ndarray:
    """Compute local shape descriptors, dispatching to the selected backend.

    Arguments match the existing lsd_lite.get_lsds call site in AddLSD.process.
    Output is always (channels, *spatial), float32, clipped to [0, 1].
    """
    if backend == "lsd-lite":
        return _compute_lsds_lite(segmentation, sigma, voxel_size, labels, downsample)
    if backend == "lsd-jax":
        return _compute_lsds_jax(segmentation, sigma, voxel_size, labels, downsample)
    msg = f"Unknown LSD backend: {backend!r}"
    raise ValueError(msg)
