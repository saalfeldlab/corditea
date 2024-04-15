import gunpowder as gp
import numpy as np

class Concatenate(gp.BatchFilter):
    def __init__(self, arrays, tgt_array, concatenation_axis=0):
        self.arrays = arrays
        self.tgt_array = tgt_array
        self.concatenation_axis = concatenation_axis
    def setup(self):
        spec = None
        for ak in self.arrays:
            this_spec = self.spec[ak].copy()
            if spec is not None and spec != this_spec:
                msg = f"Specs of concatenated arrays all need to be the same, found {spec} and {this_spec}"
                raise RuntimeError(msg)
            spec = this_spec
        self.provides(self.tgt_array, spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()
        for arr in self.arrays:
            deps[arr] = request[self.tgt_array].copy()
        return deps

    def process(self, batch, request):
        data = np.concatenate(tuple(batch.arrays[ak].data for ak in self.arrays), axis = self.concatenation_axis)
        spec = self.spec[self.tgt_array].copy()
        spec.roi = request[self.tgt_array].roi.copy()
        output = gp.Batch()
        output.arrays[self.tgt_array] = gp.Array(data, spec)
        return output