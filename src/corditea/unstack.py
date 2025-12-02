"""Node to extract the first sample from a stacked batch, removing the batch dimension."""
import logging
from typing import List

import numpy as np
from gunpowder.array import ArrayKey
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest
from gunpowder.nodes import BatchFilter

logger = logging.getLogger(__name__)


class Unstack(BatchFilter):
    """Extract the first sample from a stacked batch, removing the batch dimension.

    This is useful for saving snapshots when using gp.Stack - it extracts just the
    first sample and removes the batch dimension, converting 5D (B,C,D,H,W) to 4D (C,D,H,W).

    Args:
        arrays (List[ArrayKey]): ArrayKeys to unstack.
        index (int): Index of the sample to extract from the batch (default: 0).
    """

    def __init__(self, arrays: List[ArrayKey], index: int = 0):
        self.arrays = arrays
        self.index = index

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = BatchRequest()
        for array in self.arrays:
            if array in request:
                deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = Batch()
        for array in self.arrays:
            if array in batch:
                outputs[array] = batch[array]
                # Extract the first sample from the batch dimension
                outputs[array].data = batch[array].data[self.index]
                logger.debug(f"{array} shape after unstack: {outputs[array].data.shape}")

        return outputs
