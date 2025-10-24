import pytest
import numpy as np
import gunpowder as gp
from funlib.persistence import Array as PersistenceArray

from corditea import AddLSD


@pytest.fixture
def array_keys():
    return {
        'segmentation': gp.ArrayKey('SEGMENTATION'),
        'descriptor': gp.ArrayKey('DESCRIPTOR'),
        'lsds_mask': gp.ArrayKey('LSDS_MASK'),
        'labels_mask': gp.ArrayKey('LABELS_MASK'),
    }


@pytest.fixture
def simple_2d_segmentation():
    """Simple 2D segmentation with two objects."""
    data = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 3, 0, 0],
        [3, 3, 3, 0, 0],
    ], dtype=np.uint32)
    return data


@pytest.fixture
def simple_3d_segmentation():
    """Simple 3D segmentation with one object."""
    data = np.zeros((5, 5, 5), dtype=np.uint32)
    data[1:4, 1:4, 1:4] = 1  # 3x3x3 cube in center
    return data


def test_basic_2d_lsd_computation(array_keys, simple_2d_segmentation):
    """Test basic LSD computation on 2D data."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=0.5
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data

    # LSDs should have shape (num_descriptors, height, width)
    assert len(descriptor.shape) == 3
    assert descriptor.shape[1:] == simple_2d_segmentation.shape
    # For 2D, we expect multiple descriptor channels
    assert descriptor.shape[0] > 1


def test_3d_lsd_computation(array_keys, simple_3d_segmentation):
    """Test LSD computation on 3D data."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1, 1))
    input_size = gp.Coordinate(simple_3d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_3d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=1.5
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data

    # LSDs should have shape (num_descriptors, depth, height, width)
    assert len(descriptor.shape) == 4
    assert descriptor.shape[1:] == simple_3d_segmentation.shape
    # For 3D, we expect multiple descriptor channels
    assert descriptor.shape[0] > 1


def test_background_exclusion(array_keys):
    """Test background exclusion with custom background value."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']
    mask_key = array_keys['lsds_mask']

    # Create segmentation with background value 255
    data = np.array([
        [1, 1, 255, 2, 2],
        [1, 1, 255, 2, 2],
        [255, 255, 255, 255, 255],
    ], dtype=np.uint32)

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(data.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(data, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        lsds_mask=mask_key,
        background_mode="exclude",
        background_value=255,
        sigma=1.0
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))
    request[mask_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    assert mask_key in batch.arrays

    mask = batch.arrays[mask_key].data

    # Background pixels should be masked out (set to 0)
    # Check that background locations have mask value 0
    assert mask[0, 2, 2] == 0.0  # Background pixel
    assert mask[0, 2, 0] == 0.0  # Background pixel
    # Non-background pixels should have mask value 1
    assert mask[0, 0, 0] == 1.0  # Foreground pixel


def test_multiple_background_values(array_keys):
    """Test exclusion of multiple background values."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']
    mask_key = array_keys['lsds_mask']

    data = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 255, 2, 2],
        [0, 255, 0, 255, 0],
    ], dtype=np.uint32)

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(data.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(data, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        lsds_mask=mask_key,
        background_mode="exclude",
        background_value=(0, 255),  # Multiple background values
        sigma=1.0
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))
    request[mask_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    mask = batch.arrays[mask_key].data

    # Both 0 and 255 should be masked out
    assert mask[0, 0, 2] == 0.0  # 0 background
    assert mask[0, 1, 2] == 0.0  # 255 background
    assert mask[0, 2, 1] == 0.0  # 255 background
    # Non-background pixels should have mask value 1
    assert mask[0, 0, 0] == 1.0  # Label 1


def test_sigma_tuple(array_keys, simple_2d_segmentation):
    """Test using sigma as a tuple for anisotropic kernels."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=(1.0, 2.0)  # Different sigma for each dimension
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data
    assert len(descriptor.shape) == 3
    assert descriptor.shape[1:] == simple_2d_segmentation.shape


def test_with_labels_mask(array_keys, simple_2d_segmentation):
    """Test using a labels mask to filter which regions are processed."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']
    mask_key = array_keys['lsds_mask']
    labels_mask_key = array_keys['labels_mask']

    # Create a labels mask that excludes some regions
    labels_mask = np.ones_like(simple_2d_segmentation, dtype=np.float32)
    labels_mask[0:2, 3:5] = 0  # Mask out the area with label 2

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = (
        (
            gp.ArraySource(
                seg_key,
                PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
            ),
            gp.ArraySource(
                labels_mask_key,
                PersistenceArray(labels_mask, voxel_size=voxel_size)
            )
        ) +
        gp.MergeProvider()
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        lsds_mask=mask_key,
        labels_mask=labels_mask_key,
        sigma=1.0
    )

    # Add padding to handle context requirements (sigma * 3)
    context_size = int(1.0 * 3) + 1  # sigma=1.0, so context=3, pad by 4
    pipeline = (
        source +
        gp.Pad(seg_key, gp.Coordinate((context_size, context_size))) +
        gp.Pad(labels_mask_key, gp.Coordinate((context_size, context_size))) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))
    request[mask_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    assert mask_key in batch.arrays


@pytest.mark.parametrize("downsample_factor", [1, 2, 3, 4])
def test_downsample_parameter(array_keys, simple_2d_segmentation, downsample_factor):
    """Test the downsample parameter with different factors."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=1.0,
        downsample=downsample_factor
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data

    # Output shape should always match requested ROI, regardless of downsampling
    assert len(descriptor.shape) == 3
    assert descriptor.shape[1:] == simple_2d_segmentation.shape

    # LSDs should have the right number of channels (6 for 2D)
    assert descriptor.shape[0] == 6


def test_roi_cropping(array_keys, simple_2d_segmentation):
    """Test that LSD computation works with partial ROI requests."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=1.0
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    # Request only a subset of the full ROI
    subset_roi = gp.Roi((1, 1), (3, 3))
    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=subset_roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data
    assert len(descriptor.shape) == 3
    assert descriptor.shape[1:] == (3, 3)  # Should match requested ROI size


@pytest.mark.parametrize("voxel_size", [
    gp.Coordinate((1, 1)),    # Unit voxels
    gp.Coordinate((2, 2)),    # 2x2 voxels
    gp.Coordinate((1, 2)),    # Anisotropic voxels
    gp.Coordinate((3, 3)),    # Larger voxels
])
def test_different_voxel_sizes(array_keys, simple_2d_segmentation, voxel_size):
    """Test LSD computation with different voxel sizes."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=2.0,  # Use larger sigma to test coordinate conversion
        downsample=2  # Test downsample with different voxel sizes
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims) * voxel_size

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi(gp.Coordinate((0, 0)), input_size))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert desc_key in batch.arrays
    descriptor = batch.arrays[desc_key].data

    # Output shape should match the input shape in voxels
    expected_shape_voxels = simple_2d_segmentation.shape
    assert len(descriptor.shape) == 3
    assert descriptor.shape[1:] == expected_shape_voxels

    # Should have 6 channels for 2D
    assert descriptor.shape[0] == 6


def test_scale_invariance(array_keys):
    """Test that LSD computation is scale-invariant across different voxel sizes."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    # Create a simple test segmentation
    base_data = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 3, 0, 0],
        [3, 3, 3, 0, 0],
    ], dtype=np.uint32)

    # Test equivalent configurations where sigma/voxel_size ratio stays the same
    # This should give the same results since the relative scale is identical
    configs = [
        {"voxel_size": gp.Coordinate((1, 2)), "sigma": (2.0, 2.0)},
        {"voxel_size": gp.Coordinate((3, 6)), "sigma": (6.0, 6.0)},
    ]

    results = []

    for config in configs:
        voxel_size = config["voxel_size"]
        sigma = config["sigma"]

        input_size = gp.Coordinate(base_data.shape) * voxel_size

        source = gp.ArraySource(
            seg_key,
            PersistenceArray(base_data, voxel_size=voxel_size)
        )

        lsd_node = AddLSD(
            segmentation=seg_key,
            descriptor=desc_key,
            sigma=sigma,
            downsample=1,  # No downsampling for cleaner comparison
            background_mode="label"  # Test label background mode
        )

        # Add padding
        if isinstance(sigma, (float, int)):
            context_size = int(sigma * 3) + 1
        else:
            context_size = int(max(sigma) * 3) + 1

        dims = len(voxel_size)
        context_coord = gp.Coordinate((context_size,) * dims) * voxel_size

        pipeline = (
            source +
            gp.Pad(seg_key, context_coord) +
            lsd_node
        )

        request = gp.BatchRequest()
        request[desc_key] = gp.ArraySpec(roi=gp.Roi(gp.Coordinate((0, 0)), input_size))

        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        results.append(batch.arrays[desc_key].data)

    # Compare results - they should be very similar since the physical configuration is the same
    desc1, desc2 = results

    # Both should have same shape (in voxels)
    assert desc1.shape == desc2.shape

    # The descriptors should be very similar (allowing for small numerical differences)
    # We use a relatively loose tolerance since different voxel sizes might lead to slight
    # differences in the discrete computation
    np.testing.assert_allclose(desc1, desc2, rtol=1e-1, atol=1e-2,
                              err_msg="LSDs should be similar for equivalent physical configurations")


@pytest.mark.parametrize("downsample_factor", [1, 2])
def test_scale_invariance_with_downsample(array_keys, downsample_factor):
    """Test scale invariance with downsampling and label background mode."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    # Create a larger test segmentation for downsampling
    base_data = np.array([
        [1, 1, 1, 1, 0, 2, 2, 2, 2],
        [1, 1, 1, 1, 0, 2, 2, 2, 2],
        [1, 1, 1, 1, 0, 2, 2, 2, 2],
        [1, 1, 1, 1, 0, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 0, 4, 4, 4, 4],
        [3, 3, 3, 3, 0, 4, 4, 4, 4],
        [3, 3, 3, 3, 0, 4, 4, 4, 4],
        [3, 3, 3, 3, 0, 4, 4, 4, 4],
    ], dtype=np.uint32)

    # Test equivalent configurations where sigma/voxel_size ratio stays the same
    configs = [
        {"voxel_size": gp.Coordinate((1, 2)), "sigma": (4.0, 4.0)},
        {"voxel_size": gp.Coordinate((2, 4)), "sigma": (8.0, 8.0)},
    ]

    results = []

    for config in configs:
        voxel_size = config["voxel_size"]
        sigma = config["sigma"]

        input_size = gp.Coordinate(base_data.shape) * voxel_size

        source = gp.ArraySource(
            seg_key,
            PersistenceArray(base_data, voxel_size=voxel_size)
        )

        lsd_node = AddLSD(
            segmentation=seg_key,
            descriptor=desc_key,
            sigma=sigma,
            downsample=downsample_factor,
            background_mode="label",  # Test label background mode
            background_value=0
        )

        # Add padding
        context_size = int(max(sigma) * 3) + 1
        dims = len(voxel_size)
        context_coord = gp.Coordinate((context_size,) * dims) * voxel_size

        pipeline = (
            source +
            gp.Pad(seg_key, context_coord) +
            lsd_node
        )

        request = gp.BatchRequest()
        request[desc_key] = gp.ArraySpec(roi=gp.Roi(gp.Coordinate((0, 0)), input_size))

        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        results.append(batch.arrays[desc_key].data)

    # Compare results
    desc1, desc2 = results

    # Both should have same shape (in voxels)
    assert desc1.shape == desc2.shape

    # Should be very similar despite downsampling and label background mode
    np.testing.assert_allclose(desc1, desc2, rtol=1e-1, atol=1e-2,
                              err_msg=f"LSDs should be similar with downsample={downsample_factor} and label background mode")


def test_invalid_sigma_dimensions(array_keys, simple_2d_segmentation):
    """Test that invalid sigma dimensions raise an error."""
    seg_key = array_keys['segmentation']
    desc_key = array_keys['descriptor']

    voxel_size = gp.Coordinate((1, 1))
    input_size = gp.Coordinate(simple_2d_segmentation.shape) * voxel_size

    source = gp.ArraySource(
        seg_key,
        PersistenceArray(simple_2d_segmentation, voxel_size=voxel_size)
    )

    lsd_node = AddLSD(
        segmentation=seg_key,
        descriptor=desc_key,
        sigma=(1.0, 2.0, 3.0)  # 3D sigma for 2D data
    )

    # Add padding to handle context requirements (sigma * 3)
    if isinstance(lsd_node.sigma, (int, float)):
        context_size = int(lsd_node.sigma * 3) + 1
    else:
        context_size = int(max(lsd_node.sigma) * 3) + 1

    # Get dimensionality from voxel_size
    dims = len(voxel_size)
    context_coord = gp.Coordinate((context_size,) * dims)

    pipeline = (
        source +
        gp.Pad(seg_key, context_coord) +
        lsd_node
    )

    request = gp.BatchRequest()
    request[desc_key] = gp.ArraySpec(roi=gp.Roi((0, 0), input_size))

    with pytest.raises(gp.PipelineRequestError, match="Sigma tuple length .* must match spatial dimensions"):
        with gp.build(pipeline):
            pipeline.request_batch(request)