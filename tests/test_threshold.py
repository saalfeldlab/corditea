import gunpowder as gp
import numpy as np
import pytest
from funlib.persistence import Array as PersistenceArray

from corditea import Threshold


@pytest.fixture
def array_keys():
    return {
        "source": gp.ArrayKey("SOURCE"),
        "target": gp.ArrayKey("TARGET"),
    }


def create_persistence_array(data):
    """Create a PersistenceArray that matches the data shape."""
    if len(data.shape) == 2:
        voxel_size = gp.Coordinate((1, 1))
    elif len(data.shape) == 3:
        voxel_size = gp.Coordinate((1, 1, 1))
    else:
        msg = f"Unsupported data shape: {data.shape}"
        raise ValueError(msg)

    return PersistenceArray(data, voxel_size=voxel_size)


def test_basic_threshold(array_keys):
    """Test basic thresholding functionality."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data
    data = np.array([[0.0, 0.3, 0.7, 1.0], [0.2, 0.5, 0.8, 0.9], [0.1, 0.4, 0.6, 1.2]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=0.5)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (3, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check thresholding (>= 0.5 becomes 1, < 0.5 becomes 0)
    expected = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_preserve_background_single_value(array_keys):
    """Test that background values are preserved."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data with background value 255
    data = np.array([[255, 0.3, 0.7, 1.0], [0.2, 255, 0.8, 0.9], [0.1, 0.4, 255, 1.2]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=0.5, background_values=255)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (3, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check that 255 values are preserved, others thresholded
    expected = np.array([[255, 0, 1, 1], [0, 255, 1, 1], [0, 0, 255, 1]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_preserve_background_multiple_values(array_keys):
    """Test that multiple background values are preserved."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data with background values 0 and 255
    data = np.array([[0, 0.3, 0.7, 255], [0.2, 255, 0.8, 0.9], [255, 0.4, 0, 1.2]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=0.5, background_values=(0, 255))

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (3, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check that 0 and 255 values are preserved, others thresholded
    expected = np.array([[0, 0, 1, 255], [0, 255, 1, 1], [255, 0, 0, 1]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_custom_threshold_values(array_keys):
    """Test custom above/below threshold values."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data
    data = np.array([[0.0, 0.3, 0.7, 1.0], [0.2, 0.5, 0.8, 0.9]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(
        source_key, target_key, threshold=0.5, above_threshold_value=100, below_threshold_value=50, background_values=0
    )

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (2, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check custom threshold values, preserving background 0
    expected = np.array([[0, 50, 100, 100], [50, 100, 100, 100]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_in_place_modification(array_keys):
    """Test in-place modification when target is None."""
    source_key = array_keys["source"]

    # Create test data
    data = np.array([[255, 0.3, 0.7, 1.0], [0.2, 0.5, 0.8, 255]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, threshold=0.5, background_values=255)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (2, 4))
    request = gp.BatchRequest()
    request[source_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[source_key].data

    # Check that modification happened in place
    expected = np.array([[255, 0, 1, 1], [0, 1, 1, 255]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_integer_data_types(array_keys):
    """Test thresholding with integer data types."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create integer test data
    data = np.array([[0, 10, 20, 30], [5, 15, 25, 255]], dtype=np.uint8)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=15, background_values=255)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (2, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check thresholding with integer data
    expected = np.array([[0, 0, 1, 1], [0, 1, 1, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)


def test_3d_data(array_keys):
    """Test thresholding with 3D data."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create 3D test data
    data = np.array([[[255, 0.3], [0.7, 1.0]], [[0.2, 0.5], [0.8, 255]]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=0.5, background_values=255)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0, 0), (2, 2, 2))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check 3D thresholding
    expected = np.array([[[255, 0], [1, 1]], [[0, 1], [1, 255]]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_different_thresholds(array_keys, threshold):
    """Test different threshold values."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32).reshape(1, 5)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=threshold)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (1, 5))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check that threshold is applied correctly
    expected = (data >= threshold).astype(np.float32)
    np.testing.assert_array_equal(result, expected)


def test_floating_point_background(array_keys):
    """Test with floating point background values."""
    source_key = array_keys["source"]
    target_key = array_keys["target"]

    # Create test data with floating point background
    data = np.array([[-1.0, 0.3, 0.7, 1.0], [0.2, -1.0, 0.8, 0.9]], dtype=np.float32)
    persistence_array = create_persistence_array(data)

    threshold_node = Threshold(source_key, target_key, threshold=0.5, background_values=-1.0)

    # Create pipeline
    pipeline = gp.ArraySource(source_key, persistence_array)
    pipeline += threshold_node

    # Request data
    roi = gp.Roi((0, 0), (2, 4))
    request = gp.BatchRequest()
    request[target_key] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    result = batch[target_key].data

    # Check that -1.0 values are preserved
    expected = np.array([[-1.0, 0, 1, 1], [0, -1.0, 1, 1]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)
