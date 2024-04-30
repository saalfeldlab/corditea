import logging
from functools import partial

import corditea
import gunpowder as gp
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def range_through_shape(shape):
    range_vals = np.arange(np.prod(shape)).reshape(shape)
    return np.full(shape, range_vals)


def test_average_downsample():
    test_arr = gp.ArrayKey("TEST_ARR")
    test_arr_down = gp.ArrayKey("TEST_ARR_DOWN")
    src = corditea.LambdaSource(
        range_through_shape,
        test_arr,
        {test_arr: gp.ArraySpec(roi=None, voxel_size=gp.Coordinate((2, 2)), interpolatable=True)},
    )
    pipeline = src + gp.AsType(test_arr, np.float32) + corditea.AverageDownSample(test_arr, (4, 4), test_arr_down)
    request = gp.BatchRequest()
    request.add(test_arr_down, (8, 8), voxel_size=(4, 4))

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
    assert np.array_equal(batch[test_arr_down].data, np.array([[2.5, 4.5], [10.5, 12.5]]))


def test_average_downsample_inplace():
    test_arr = gp.ArrayKey("TEST_ARR")
    src = corditea.LambdaSource(
        range_through_shape,
        test_arr,
        {test_arr: gp.ArraySpec(roi=None, voxel_size=gp.Coordinate((2, 2)), interpolatable=True)},
    )
    pipeline = src + gp.AsType(test_arr, np.float32) + corditea.AverageDownSample(test_arr, (4, 4))
    request = gp.BatchRequest()
    request.add(test_arr, (8, 8), voxel_size=(4, 4))

    with gp.build(pipeline) as p:
        batch = p.request_batch(request)
    assert np.array_equal(batch[test_arr].data, np.array([[2.5, 4.5], [10.5, 12.5]]))


if __name__ == "__main__":
    test_average_downsample_inplace()
