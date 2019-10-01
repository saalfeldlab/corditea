import logging
import numpy as np
import collections

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array, ArrayKey
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class LambdaSource(BatchProvider):
    '''A lambda data source.

    Provides arrays using a given function for each array key given. In order to
    know the dimensionality at least each array's ``voxel_size needs to be provided.

    Args:

        func (``function``):

            A function that returns a numpy array of a given, arbitrary shape,
            e.g. np.ones

        array_keys (``tuple``, ``list`` or :class:`ArrayKey`):

            List or tuple of array_keys that should be provided using the given
            function.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`,):

            A dictionary of array keys to array specs. At least the ``voxel_size``
            needs to be defined.
    '''
    def __init__(
            self,
            func,
            array_keys,
            array_specs,):

        self.func = func
        if not isinstance(array_keys, collections.Iterable):
            assert isinstance(array_keys, ArrayKey)
            array_keys = (array_keys,)
        self.array_keys = array_keys

        self.array_specs = array_specs

    def setup(self):
        for array_key in self.array_keys:
            spec = self.__read_spec(array_key)
            self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for (array_key, request_spec) in request.array_specs.items():
            logger.debug("Reading %s in %s...", array_key, request_spec.roi)

            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            # dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[array_key] = Array(self.func(dataset_roi.get_shape()), array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, array_key):

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()
        assert spec.voxel_size is not None, "Voxel size needs to be given"

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            roi = Roi(Coordinate((0,) * self.ndims), shape=Coordinate((1,)*self.ndims))
            roi.set_shape(None)
            spec.roi = roi

        arr = self.func((2,) * self.ndims)
        if spec.dtype is not None:
            assert spec.dtype == arr.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from function output %s dtype %s" %
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, self.func, arr.dtype))
        else:
            spec.dtype = arr.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(func %s) . Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, self.func, spec.dtype,
                           spec.interpolatable)

        return spec

    def __repr__(self):

        return self.func
