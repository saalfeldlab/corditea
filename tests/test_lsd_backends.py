"""Parity tests between the lsd-lite and lsd-jax LSD backends.

Exercises the `compute_lsds` dispatch function directly, bypassing gunpowder
machinery.
"""

import numpy as np
import pytest

from corditea._lsd_backends import _next_power_of_two, compute_lsds


# Tolerance: the JAX implementation uses slightly different variance clamps
# (1e-2 in 3D, 1e-3 in 2D) vs lsd-lite, so exact equality is not expected.
ATOL = 1e-2


@pytest.fixture
def seg_2d():
    """Two disjoint rectangles plus some background (255)."""
    data = np.full((16, 16), 255, dtype=np.uint32)
    data[2:6, 2:7] = 1
    data[9:13, 9:14] = 2
    return data


@pytest.fixture
def seg_3d():
    """One cube and one ellipsoid-ish blob, background 0 and 255 mixed."""
    data = np.zeros((12, 12, 12), dtype=np.uint32)
    data[2:6, 2:6, 2:6] = 1
    data[7:11, 7:11, 7:11] = 2
    data[0, :, :] = 255  # background slab on one face
    return data


def _labels_excluding_background(seg, bg_values=(0, 255)):
    labels = list(np.unique(seg))
    return [int(v) for v in labels if int(v) not in bg_values]


@pytest.mark.parametrize("backend", ["lsd-lite", "lsd-jax"])
def test_output_shape_and_dtype_3d(seg_3d, backend):
    labels = _labels_excluding_background(seg_3d)
    out = compute_lsds(
        segmentation=seg_3d,
        sigma=2.0,
        voxel_size=(1, 1, 1),
        labels=labels,
        downsample=1,
        backend=backend,
    )
    assert out.shape == (10, *seg_3d.shape)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("backend", ["lsd-lite", "lsd-jax"])
def test_output_shape_and_dtype_2d(seg_2d, backend):
    labels = _labels_excluding_background(seg_2d)
    out = compute_lsds(
        segmentation=seg_2d,
        sigma=1.0,
        voxel_size=(1, 1),
        labels=labels,
        downsample=1,
        backend=backend,
    )
    assert out.shape == (6, *seg_2d.shape)
    assert out.dtype == np.float32


def test_parity_3d(seg_3d):
    labels = _labels_excluding_background(seg_3d)
    common = dict(
        segmentation=seg_3d,
        sigma=2.0,
        voxel_size=(1, 1, 1),
        labels=labels,
        downsample=1,
    )
    lite = compute_lsds(backend="lsd-lite", **common)
    jax_out = compute_lsds(backend="lsd-jax", **common)

    # Zero out positions where either backend has no meaningful output
    # (outside any real label or inside background), to focus parity check
    # on the foreground descriptors.
    keep = np.isin(seg_3d, np.asarray(labels))
    diff = np.abs(lite - jax_out) * keep[None, ...]
    max_diff = float(diff.max())
    assert max_diff < ATOL, f"max |lite - jax| on foreground = {max_diff}"


def test_parity_2d(seg_2d):
    labels = _labels_excluding_background(seg_2d)
    common = dict(
        segmentation=seg_2d,
        sigma=1.0,
        voxel_size=(1, 1),
        labels=labels,
        downsample=1,
    )
    lite = compute_lsds(backend="lsd-lite", **common)
    jax_out = compute_lsds(backend="lsd-jax", **common)

    keep = np.isin(seg_2d, np.asarray(labels))
    diff = np.abs(lite - jax_out) * keep[None, ...]
    max_diff = float(diff.max())
    assert max_diff < ATOL, f"max |lite - jax| on foreground = {max_diff}"


def test_jax_rejects_anisotropic_voxels(seg_3d):
    labels = _labels_excluding_background(seg_3d)
    with pytest.raises(NotImplementedError, match="isotropic"):
        compute_lsds(
            segmentation=seg_3d,
            sigma=2.0,
            voxel_size=(2, 1, 1),
            labels=labels,
            downsample=1,
            backend="lsd-jax",
        )


def test_jax_rejects_downsample(seg_3d):
    labels = _labels_excluding_background(seg_3d)
    with pytest.raises(NotImplementedError, match="downsample"):
        compute_lsds(
            segmentation=seg_3d,
            sigma=2.0,
            voxel_size=(1, 1, 1),
            labels=labels,
            downsample=2,
            backend="lsd-jax",
        )


def test_jax_bucket_transition(seg_3d):
    """Calling with varying label counts should not crash (bucket promotion)."""
    # First call: 2 labels → bucket 4
    compute_lsds(
        segmentation=seg_3d,
        sigma=2.0,
        voxel_size=(1, 1, 1),
        labels=[1, 2],
        downsample=1,
        backend="lsd-jax",
    )
    # Second call on same shape but with many synthetic labels → bucket grows.
    seg_many = seg_3d.copy()
    seg_many[3:9, 3:9, 3:9] = np.arange(6 * 6 * 6, dtype=np.uint32).reshape(6, 6, 6) + 10
    labels_many = _labels_excluding_background(seg_many)
    out = compute_lsds(
        segmentation=seg_many,
        sigma=2.0,
        voxel_size=(1, 1, 1),
        labels=labels_many,
        downsample=1,
        backend="lsd-jax",
    )
    assert out.shape == (10, *seg_3d.shape)


def test_next_power_of_two():
    assert _next_power_of_two(1) == 2
    assert _next_power_of_two(2) == 2
    assert _next_power_of_two(3) == 4
    assert _next_power_of_two(4) == 4
    assert _next_power_of_two(5) == 8
    assert _next_power_of_two(1000) == 1024


# --- Node-level integration: AddLSD end-to-end with each backend ---


@pytest.mark.parametrize("backend", ["lsd-lite", "lsd-jax"])
def test_addlsd_node_3d(backend):
    """Exercise the full AddLSD gunpowder node path with each backend."""
    import gunpowder as gp
    from funlib.persistence import Array as PersistenceArray

    from corditea import AddLSD

    data = np.zeros((10, 10, 10), dtype=np.uint32)
    data[2:7, 2:7, 2:7] = 1

    seg_key = gp.ArrayKey("SEGMENTATION")
    desc_key = gp.ArrayKey("DESCRIPTOR")
    voxel_size = gp.Coordinate((1, 1, 1))
    input_size = gp.Coordinate(data.shape) * voxel_size

    source = gp.ArraySource(seg_key, PersistenceArray(data, voxel_size=voxel_size))
    lsd_node = AddLSD(segmentation=seg_key, descriptor=desc_key, sigma=1.5, backend=backend)

    context = gp.Coordinate((int(1.5 * 3) + 1,) * 3)
    pipeline = source + gp.Pad(seg_key, context) + lsd_node

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    descriptor = batch.arrays[desc_key].data
    assert descriptor.shape == (10, *data.shape)
    assert descriptor.dtype == np.float32
