import numpy as np

from .batch_filter import BatchFilter
from collections import Iterable

class GammaAugment(BatchFilter):
    '''Randomly scale and shift the values of an intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        scale_min (``float``):
        scale_max (``float``):
        shift_min (``float``):
        shift_max (``float``):

            The min and max of the uniformly randomly drawn scaling and
            shifting values for the intensity augmentation. Intensities are
            changed as::

                a = a.mean() + (a-a.mean())*scale + shift

        z_section_wise (``bool``):

            Perform the augmentation z-section wise. Requires 3D arrays and
            assumes that z is the first dimension.
    '''

    def __init__(self, arrays, gamma_min, gamma_max):
        if not isinstance(arrays, Iterable):
            arrays = [arrays,]
        self.arrays = arrays
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        assert self.gamma_max >= self.gamma_min

    def process(self, batch, request):
        sample_gamma_min = (max(self.gamma_min, 1./self.gamma_min) - 1)*(-1)**(self.gamma_min < 1)
        sample_gamma_max = (max(self.gamma_max, 1./self.gamma_max) - 1)*(-1)**(self.gamma_max < 1)
        gamma = np.random.uniform(sample_gamma_min, sample_gamma_max)
        if gamma < 0:
            gamma = 1./(-gamma+1)
        else:
            gamma = gamma + 1
        for array in self.arrays:
            raw = batch.arrays[array]

            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Gamma augmentation requires float " \
                                                                                 "types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
            assert raw.data.min() >= 0 and raw.data.max() <= 1, "Gamma augmentation expects raw values in [0," \
                                                                "1]. Consider using Normalize before."

            raw.data = self.__augment(raw.data, gamma)

            # clip values, we might have pushed them out of [0,1]
            raw.data[raw.data > 1] = 1
            raw.data[raw.data < 0] = 0

    def __augment(self, a, gamma):

        return a**gamma
