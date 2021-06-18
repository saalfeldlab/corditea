import logging
from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class Sum(BatchFilter):
    '''Combine

    Args:

        array_keys(tuple or list of :class:``ArrayKey``): Tuple or list containing the :class:``ArrayKey`` that
            should be summed up.

        sum_array_key(:class:``ArrayKey``): The :class:``ArrayKey`` for the sum.

        sum_array_spec(:class:``ArraySpec``, optional): The :class:``ArraySpec`` for the `sum_array_key`. If None or
            for missing values the :class:``ArraySpec`` of the first element in `array_keys` will be used
    '''

    def __init__(
            self,
            array_keys,
            sum_array_key,
            sum_array_spec=None,
            ):

        self.array_keys = array_keys
        self.sum_array_key = sum_array_key
        self.sum_array_spec = sum_array_spec

    def setup(self):
        vs = self.spec[self.array_keys[0]].voxel_size
        roi = self.spec[self.array_keys[0]].roi
        for ak in self.array_keys:
            assert ak in self.spec, ("Upstream does not provide %s needed by Sum" % ak)
            assert self.spec[ak].voxel_size == vs, ("Inconsistent voxel sizes in Sum {0:} {1:} and {2:} {3:}".format(
            self.array_keys[0], vs, ak, self.spec[ak].voxel_size))
            assert self.spec[ak].roi == roi, ("Inconsistent ROIs in Sum {0:} {1:} and {2:} {3:}".format(
            self.array_keys[0], roi, ak, self.spec[ak].roi))

        if self.sum_array_spec is not None:
            spec = self.sum_array_spec
            if spec.voxel_size is None:
                spec.voxel_size = vs
            if spec.roi is None:
                spec.roi = roi
            if spec.dtype is None:
                spec.dtype = self.spec[self.array_keys[0]].dtype
            if spec.interpolatable is None:
                spec.interpolatable = self.spec[self.array_keys[0]].interpolatable
        else:
            spec = self.spec[self.array_keys[0]].copy()
        self.provides(self.sum_array_key, spec)

    def process(self, batch, request):

        if (self.sum_array_key not in request):
            return
        sum_arr = batch.arrays[self.array_keys[0]].data
        for ak in self.array_keys[1:]:
            sum_arr += batch.arrays[ak].data
        spec = self.spec[self.sum_array_key].copy()
        spec.roi = request[self.sum_array_key].roi
        batch.arrays[self.sum_array_key] = Array(sum_arr, spec)

