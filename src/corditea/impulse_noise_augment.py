import logging
from collections.abc import Iterable

import gunpowder as gp
import numpy as np

logger = logging.getLogger(__name__)


class ImpulseNoiseAugment(gp.BatchFilter):
    '''Add random valued impulse noise to an intensity array or a list of intensity arrays.

    Args:

        arrays (:class:`ArrayKey` or list of :class:`ArrayKey`s):

            The intensity arrays to modify, applying the same noise pattern to each..

        p (``float``):

            Per-pixel probablity to be corrupted by noise

        range (``tuple``, default: (0,1) ):

            Range for random values of noise, drawn from a uniform distribution. For fixed valued impulse noise set
            start and end of this range to the same value.
    '''

    def __init__(self, arrays, p, range=(0, 1)):
        if not isinstance(arrays, Iterable):
            arrays = [
                arrays,
            ]
        self.arrays = arrays
        self.p = p
        self.range = range

    def process(self, batch, request):

        assert all([batch.arrays[array].data.shape == batch.arrays[self.arrays[0]].data.shape for array in self.arrays])

        noise_locations = (np.random.binomial(1, self.p, batch.arrays[self.arrays[0]].data.shape)).astype(bool)
        noise_values = np.random.uniform(self.range[0], self.range[1], np.sum(noise_locations))

        for array in self.arrays:
            raw = batch.arrays[array]
            raw.data[noise_locations] = noise_values
