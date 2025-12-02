import logging

import gunpowder as gp
import numpy as np


class LogBatch(gp.BatchFilter):
    """Log batch information periodically during training.

    Args:
        mask_key: ArrayKey for mask to compute valid fraction
        log_every: Log every N iterations
        logger: Logger instance to use (defaults to module logger)
    """

    def __init__(self, mask_key: gp.ArrayKey = None, log_every: int = 100, logger=None):
        self.mask_key = mask_key
        self.log_every = log_every
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.iteration = 0

    def process(self, batch, request):
        if self.iteration % self.log_every == 0:
            log_parts = [f"Iteration {self.iteration}"]

            # Log loss if available (from Train node)
            if hasattr(batch, 'loss'):
                log_parts.append(f"Loss: {batch.loss:.4f}")

            # Log mask coverage (fraction of valid pixels per channel)
            if self.mask_key and self.mask_key in batch.arrays:
                mask = batch.arrays[self.mask_key].data
                # Average across spatial dimensions but keep channel dimension
                # Shape is typically (batch, channels, z, y, x)
                valid_per_channel = np.mean(mask > 0, axis=tuple(range(2, mask.ndim)))
                avg_valid = np.mean(valid_per_channel)
                log_parts.append(f"Valid fraction: {avg_valid:.3f}")

            self.logger.info(" | ".join(log_parts))

        self.iteration += 1
        return batch
