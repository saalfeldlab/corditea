import logging
from typing import Optional, Union

import gunpowder as gp
import numpy as np

logger = logging.getLogger(__name__)


class Threshold(gp.BatchFilter):
    """Apply thresholding to an array while preserving background values.

    This node applies a threshold operation to convert values to binary output,
    but preserves specified background values unchanged. This is useful when
    working with arrays that have special background values that should not
    be affected by thresholding.

    Args:

        source (:class:`ArrayKey`):
            The key of the array to threshold.

        target (:class:`ArrayKey`, optional):
            The key to store the thresholded result. If None, the source array
            will be modified in place.

        threshold (float):
            The threshold value. Values >= threshold become 1, values < threshold become 0.

        background_values (int, float, or tuple of int/float):
            Value(s) to consider as background that should be preserved unchanged.
            Default is 255.

        above_threshold_value (int or float):
            Value to assign to pixels that are >= threshold (and not background).
            Default is 1.

        below_threshold_value (int or float):
            Value to assign to pixels that are < threshold (and not background).
            Default is 0.
    """

    def __init__(
        self,
        source: gp.ArrayKey,
        target: Optional[gp.ArrayKey] = None,
        threshold: float = 0.5,
        background_values: Union[int, float, tuple[Union[int, float], ...]] = 255,
        above_threshold_value: Union[int, float] = 1,
        below_threshold_value: Union[int, float] = 0,
    ) -> None:
        assert isinstance(source, gp.ArrayKey)
        self.source = source
        self.threshold = threshold
        self.above_threshold_value = above_threshold_value
        self.below_threshold_value = below_threshold_value

        if target is None:
            self.target = source
        else:
            assert isinstance(target, gp.ArrayKey)
            self.target = target

        # Normalize background values to tuple
        if isinstance(background_values, (int, float)):
            background_values = (background_values,)
        self.background_values = background_values

    def setup(self) -> None:
        # Get the source spec
        source_spec = self.spec[self.source]

        # Create target spec with same properties as source
        target_spec = source_spec.copy()

        if self.target == self.source:
            # Update the existing array spec
            self.updates(self.target, target_spec)
        else:
            # Provide a new array
            self.provides(self.target, target_spec)

    def prepare(self, request: gp.BatchRequest) -> gp.BatchRequest:
        # Request the source array
        deps = gp.BatchRequest()
        deps[self.source] = request[self.target].copy()
        return deps

    def process(self, batch: gp.Batch, request: gp.BatchRequest) -> gp.Batch:
        # Get the source array
        source_array = batch.arrays[self.source]
        data = source_array.data.copy()

        logger.debug(
            f"Applying threshold {self.threshold} to array with shape {data.shape}, "
            f"preserving background values {self.background_values}"
        )

        # Create mask for background values
        background_mask = np.zeros_like(data, dtype=bool)
        for bg_val in self.background_values:
            background_mask |= data == bg_val

        # Apply threshold to non-background pixels
        non_bg_mask = ~background_mask

        # Apply thresholding
        thresholded = np.where(data >= self.threshold, self.above_threshold_value, self.below_threshold_value)

        # Preserve background values
        output_data = np.where(background_mask, data, thresholded)

        logger.debug(
            f"Threshold applied: {np.sum(non_bg_mask & (data >= self.threshold))} pixels "
            f"above threshold, {np.sum(non_bg_mask & (data < self.threshold))} below, "
            f"{np.sum(background_mask)} background pixels preserved"
        )

        # Create output array with same spec as source
        target_spec = source_array.spec.copy()
        target_spec.dtype = output_data.dtype

        # Create the output batch
        output_batch = gp.Batch()
        output_batch.arrays[self.target] = gp.Array(output_data, target_spec)

        return output_batch
