import logging
import gunpowder as gp

logger = logging.getLogger(__name__)


class LambdaFilter(gp.BatchFilter):
    '''Apply an aribitrary function to the data in an array.

    Args:

        func(``function``): Function that will be applied to the array data. The only required input should be a
            numpy array.

        source_key(:class:``ArrayKey``): The :class:``ArrayKey`` that provides the input to the function.

        target_key(tuple of :class:``ArrayKey``, optional): The :class:``ArrayKey`` that the result will be written
            to. If None, `source_key` will be overwritten. (default: None)

        target_spec(:class: ``ArraySpec``, optional): The :class:``ArraySpec`` for the target key. If not given the
            spec of source_key will be used. All members that are None will also be overwritten by spec of source_key.
    '''

    def __init__(
            self,
            func,
            source_key,
            target_key=None,
            target_spec=None
            ):
        self.func = func
        self.source_key = source_key
        if target_key is not None:
            self.target_key = target_key
        else:
            self.target_key = source_key
        self.target_spec = target_spec

    def setup(self):

        assert self.source_key in self.spec, (
            "Upstream does not provide %s needed by "
            "LambdaFilter" % self.source_key)

        if self.target_key != self.source_key:
            if self.target_spec is not None:
                spec = self.target_spec
                if spec.voxel_size is None:
                    spec.voxel_size = self.spec[self.source_key].voxel_size
                if spec.roi is None:
                    spec.roi = self.spec[self.source_key].roi
                if spec.dtype is None:
                    spec.dtype = self.spec[self.source_key].dtype
                if spec.interpolatable is None:
                    spec.interpolatable = self.spec[self.source_key].interpolatable
            else:
                spec = self.spec[self.source_key].copy()
            self.provides(self.target_key, spec)

    def process(self, batch, request):

        if self.target_key not in request:
            return
        arr = batch.arrays[self.source_key].data
        func_output = self.func(arr)

        spec = self.spec[self.target_key].copy()
        spec.roi = request[self.target_key].roi
        batch.arrays[self.target_key] = gp.Array(func_output, spec)
