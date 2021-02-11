import logging
import gunpowder as gp


logger = logging.getLogger(__name__)


class Multiply(gp.BatchFilter):
    def __init__(self, keys, target_key, target_spec=None):
        self.keys = keys
        self.target_key = target_key
        self.target_spec = target_spec

    def setup(self):
        vs = self.spec[self.keys[0]].voxel_size
        roi = self.spec[self.keys[0]].roi
        dtype = self.spec[self.keys[0]].dtype
        for key in self.keys:
            assert key in self.spec, (
                "Upstream does not provide %s needed by Multiply" % key
            )
            assert (
                self.spec[key].voxel_size == vs
            ), "Inconsistent voxel sizes in Multiply {0:} {1:} and {2:} {3:}".format(
                self.keys[0], vs, key, self.spec[key].voxel_size
            )
            assert (
                self.spec[key].roi == roi
            ), "Inconsistent ROIs in Multiply {0:} {1:} and {2:} {3:}".format(
                self.keys[0], vs, key, self.spec[key].voxel_size
            )
            if self.target_spec.dtype is None:
                assert (
                    self.spec[key].dtype == dtype
                ), "Inconsistent dtypes in Multiply {0:} {1:} and {2:} {3:}, but target dtype is not specified".format(
                    self.keys[0], dtype, key, self.spec[key].dtype
                )

        if self.target_spec is not None:
            spec = self.target_spec
            if spec.voxel_size is None:
                spec.voxel_size = vs
            if spec.roi is None:
                spec.roi = roi
            if spec.dtype is None:
                spec.dtype = self.spec[self.keys[0]].dtype
            if spec.interpolatable is None:
                spec.interpolatable = self.spec[self.keys[0]].interpolatable
        else:
            spec = self.spec[self.keys[0]].copy()
        if self.target_key in self.keys:
            self.updates(self.target_key, spec)
        else:
            self.provides(self.target_key, spec)

    def process(self, batch, request):
        if self.target_key not in request:
            return
        prod_arr = batch.arrays[self.keys[0]].data
        for key in self.keys[1:]:
            prod_arr *= batch.arrays[key].data
        if self.target_spec.dtype is not None:
            prod_arr.astype(self.target_spec.dtype)
        spec = self.spec[self.target_key].copy()
        spec.roi = request[self.target_key].roi
        spec.dtype = prod_arr.dtype
        batch.arrays[self.target_key] = gp.Array(prod_arr, spec)
