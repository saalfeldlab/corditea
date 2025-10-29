import logging

import gunpowder as gp
import numpy as np

logger = logging.getLogger(__name__)


class IntensityCrop(gp.BatchFilter):
    """Crop the values of an array to a given range.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        lower (scalar or None):

            Minimum value. If None, clipping is not performed on lower interval edge. Not more than one of ``lower``
            and ``upper`` may be None.

        upper (scalar or None):

            Maximum value. If None, clipping is not performed on upper interval edge. Not more than one of ``lower``
            and ``upper`` may be None.

    """

    def __init__(self, array, lower, upper):
        self.array = array
        self.lower = lower
        self.upper = upper

    def process(self, batch, request):
        if self.array not in batch.arrays:
            return

        array = batch.arrays[self.array]

        array.data = np.clip(array.data, self.lower, self.upper)
