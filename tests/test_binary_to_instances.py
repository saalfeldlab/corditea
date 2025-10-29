import gunpowder as gp
import numpy as np
import pytest
from funlib.persistence import Array as PersistenceArray

from corditea import BinaryToInstances


@pytest.fixture
def array_keys():
    """Fixture providing test array keys."""
    return {
        "source": gp.ArrayKey("TEST_ARRAY"),
        "target": gp.ArrayKey("TARGET_ARRAY"),
        "mask": gp.ArrayKey("MASK_ARRAY"),
    }


def test_simple_2d_binary(array_keys):
    """Test basic 2D binary to instances conversion."""
    # Create a simple 2D binary array with two separate objects
    binary_data = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], dtype=np.uint8
    )

    # Create array spec
    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (5, 5))

    # Create test pipeline
    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])

    pipeline = source + binary_to_instances

    # Request data
    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # Check results
    result = batch.arrays[array_keys["target"]].data

    # Should have 2 distinct objects (plus background 0)
    unique_labels = np.unique(result)
    assert len(unique_labels) == 3  # 0, 1, 2
    assert unique_labels[0] == 0  # Background

    # Check that connected regions have same ladobels
    # Top-left 2x2 region should have same label
    top_left_label = result[1, 1]
    assert result[1, 2] == top_left_label
    assert result[2, 1] == top_left_label
    assert result[2, 2] == top_left_label

    # Bottom-right region should have different label
    bottom_right_label = result[3, 3]
    assert result[4, 3] == bottom_right_label
    assert top_left_label != bottom_right_label


def test_3d_binary(array_keys):
    """Test 3D binary to instances conversion."""
    # Create a 3D binary array with one connected component
    binary_data = np.zeros((3, 3, 3), dtype=np.uint8)
    binary_data[1, 1, 1] = 1  # Center voxel
    binary_data[1, 1, 0] = 1  # Connected in z
    binary_data[1, 1, 2] = 1  # Connected in z

    # Create array spec
    voxel_size = gp.Coordinate((1, 1, 1))
    roi = gp.Roi((0, 0, 0), (3, 3, 3))

    # Create test pipeline
    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])

    pipeline = source + binary_to_instances

    # Request data
    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # Check results
    result = batch.arrays[array_keys["target"]].data

    # Should have 1 object (plus background 0)
    unique_labels = np.unique(result)
    assert len(unique_labels) == 2  # 0, 1

    # All connected voxels should have the same label
    connected_label = result[1, 1, 1]
    assert connected_label != 0
    assert result[1, 1, 0] == connected_label
    assert result[1, 1, 2] == connected_label


def test_in_place_modification(array_keys):
    """Test in-place modification when target is not specified."""
    binary_data = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

    # Create array spec
    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    # Create test pipeline with no target specified (in-place)
    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"])  # No target

    pipeline = source + binary_to_instances

    # Request data
    request = gp.BatchRequest()
    request[array_keys["source"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    # Check results - should have 4 separate objects (corners)
    result = batch.arrays[array_keys["source"]].data
    unique_labels = np.unique(result)
    assert len(unique_labels) == 5  # 0, 1, 2, 3, 4


def test_connectivity_parameter(array_keys):
    """Test different connectivity parameters."""
    # Create diagonal connection
    binary_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    # Test with 4-connectivity (should be 3 separate objects)
    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], connectivity=1)
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result_4conn = batch.arrays[array_keys["target"]].data
    unique_4conn = np.unique(result_4conn)
    assert len(unique_4conn) == 4  # 0, 1, 2, 3

    # Test with 8-connectivity (should be 1 connected object)
    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], connectivity=2)
    pipeline = source + binary_to_instances

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result_8conn = batch.arrays[array_keys["target"]].data
    unique_8conn = np.unique(result_8conn)
    assert len(unique_8conn) == 2  # 0, 1


def test_custom_background(array_keys):
    """Test custom background value."""
    # Use 255 as background
    binary_data = np.array([[255, 1, 255], [1, 1, 1], [255, 1, 255]], dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], background=255)
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should have 1 connected component (the cross shape)
    unique_labels = np.unique(result)
    assert len(unique_labels) == 2  # 255, 1

    # Check that background is preserved
    assert result[0, 0] == 255  # Was 255, should stay 255
    assert result[0, 2] == 255  # Was 255, should stay 255


def test_empty_array(array_keys):
    """Test behavior with empty (all background) array."""
    binary_data = np.zeros((3, 3), dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should only have background (0)
    unique_labels = np.unique(result)
    assert len(unique_labels) == 1
    assert unique_labels[0] == 0


def test_dtype_preservation(array_keys):
    """Test that appropriate data types are used for output."""
    binary_data = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should preserve uint8 dtype since labels fit
    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    "connectivity,expected_components",
    [
        (1, 4),  # 4-connectivity: diagonal pixels not connected
        (2, 2),  # 8-connectivity: diagonal pixels connected
    ],
)
def test_connectivity_parametrized(array_keys, connectivity, expected_components):
    """Parametrized test for different connectivity values."""
    binary_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], connectivity=connectivity)
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data
    unique_labels = np.unique(result)
    assert len(unique_labels) == expected_components  # background + components


def test_singleton_channel_dimensions(array_keys):
    """Test that singleton channel dimensions are handled correctly."""
    # 2D spatial data with singleton channel dimension
    binary_data = np.array([[[1, 0], [0, 1]]], dtype=np.uint8)  # Shape: (1, 2, 2)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (2, 2))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should maintain original shape
    assert result.shape == (1, 2, 2)

    # Should have proper instance labels
    unique_labels = np.unique(result)
    assert len(unique_labels) == 3  # 0, 1, 2 (two separate diagonal components)


def test_singleton_spatial_dimension(array_keys):
    """Test handling of arrays where spatial dimensions can be 1."""
    # 2D array where one spatial dimension is 1: (1, 1, 5) - channel, height, width
    binary_data = np.array([[[1, 0, 1, 0, 1]]], dtype=np.uint8)  # Shape: (1, 1, 5)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (1, 5))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should maintain original shape including spatial dimension of 1
    assert result.shape == (1, 1, 5)

    # Should have 3 separate components (3 isolated pixels)
    unique_labels = np.unique(result)
    assert len(unique_labels) == 4  # 0, 1, 2, 3


def test_non_singleton_channel_error(array_keys):
    """Test that non-singleton channel dimensions raise an error."""
    # 2D spatial data with non-singleton channel dimension
    binary_data = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)  # Shape: (2, 2, 2)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (2, 2))

    source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"])
    pipeline = source + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        with pytest.raises(Exception) as exc_info:
            batch = pipeline.request_batch(request)

    # Should raise ValueError about non-singleton dimensions
    assert "Non-spatial dimensions must be singleton" in str(exc_info.value)


def test_mask_input(array_keys):
    """Test mask functionality to limit where connected components are found."""
    # Create binary data with two separate regions
    binary_data = np.array(
        [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.uint8
    )

    # Create mask that only allows left side
    mask_data = np.array(
        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=np.uint8
    )

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (5, 5))

    # Create pipeline with mask
    binary_source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    mask_source = gp.ArraySource(array_keys["mask"], PersistenceArray(mask_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], mask=array_keys["mask"])

    pipeline = (binary_source, mask_source) + gp.MergeProvider() + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should only have connected components where mask allows (left side)
    # Right side should be all zeros despite having foreground in binary_data
    assert np.all(result[:, 3:] == 0)  # Right side should be all background

    # Left side should have 2 connected components (top-left and bottom-left)
    left_side = result[:2, :2]
    bottom_left = result[3:, :2]

    # Both regions should have different labels since they're not connected
    left_labels = np.unique(left_side[left_side > 0])
    bottom_labels = np.unique(bottom_left[bottom_left > 0])

    assert len(left_labels) == 1
    assert len(bottom_labels) == 1
    assert left_labels[0] != bottom_labels[0]  # Should be different components


def test_mask_completely_blocks(array_keys):
    """Test that an empty mask blocks all connected component detection."""
    binary_data = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

    # Empty mask
    mask_data = np.zeros((3, 3), dtype=np.uint8)

    voxel_size = gp.Coordinate((1, 1))
    roi = gp.Roi((0, 0), (3, 3))

    binary_source = gp.ArraySource(array_keys["source"], PersistenceArray(binary_data, voxel_size=voxel_size))
    mask_source = gp.ArraySource(array_keys["mask"], PersistenceArray(mask_data, voxel_size=voxel_size))
    binary_to_instances = BinaryToInstances(array_keys["source"], array_keys["target"], mask=array_keys["mask"])

    pipeline = (binary_source, mask_source) + gp.MergeProvider() + binary_to_instances

    request = gp.BatchRequest()
    request[array_keys["target"]] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    result = batch.arrays[array_keys["target"]].data

    # Should be all background since mask blocks everything
    unique_labels = np.unique(result)
    assert len(unique_labels) == 1
    assert unique_labels[0] == 0
