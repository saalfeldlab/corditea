import logging
import random

import numpy as np
import skimage
from gunpowder import BatchFilter, BatchRequest

logger = logging.getLogger(__name__)


class GaussianNoiseAugment(BatchFilter):
    """Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation
    """

    def __init__(self, array, clip=True, noise_prob=1.0, var_range=(0, 0.02), **kwargs):
        self.array = array
        self.clip = clip
        self.noise_prob = noise_prob
        self.var_range = var_range
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = BatchRequest()
        r = random.random()
        if r < self.noise_prob:
            self.var = random.uniform(self.var_range[0], self.var_range[1])
        else:
            self.var = 0
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Noise augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert raw.data.min() >= -1 and raw.data.max() <= 1, (
                "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."
            )

        seed = request.random_seed

        if self.var > 0:
            try:
                raw.data = skimage.util.random_noise(
                    raw.data, mode="gaussian", rng=seed, clip=self.clip, var=self.var, **self.kwargs
                ).astype(raw.data.dtype)
            except ValueError:
                # legacy version of skimage random_noise
                raw.data = skimage.util.random_noise(
                    raw.data, mode="gaussian", seed=seed, clip=self.clip, var=self.var, **self.kwargs
                ).astype(raw.data.dtype)
