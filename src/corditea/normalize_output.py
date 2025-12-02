import logging
from typing import Optional, Sequence

import gunpowder as gp
import numpy as np
import torch

logger = logging.getLogger(__name__)


class NormalizeOutput(gp.BatchFilter):
    """Apply activations to unnormalized output channels for visualization.

    This node applies specified activations to channels on a per-channel basis,
    typically used to normalize outputs for snapshot visualization during training.

    Args:
        input_array: The array key containing raw model outputs
        channel_activations: List of activation names (e.g., "Sigmoid", "Tanh") per channel.
            None means leave the channel unchanged.
        output_array: Optional output array key. If None, modifies input_array in-place.
    """

    def __init__(
        self,
        input_array: gp.ArrayKey,
        channel_activations: Sequence[Optional[str]],
        output_array: Optional[gp.ArrayKey] = None,
    ):
        self.input_array = input_array
        self.output_array = output_array
        self.channel_activations = channel_activations
        self.in_place = output_array is None

        # Instantiate activation modules
        self.activations = []
        for activation_name in channel_activations:
            if activation_name is not None:
                activation_class = getattr(torch.nn, activation_name)
                activation = activation_class()
            else:
                activation = None
            self.activations.append(activation)

        mode = "in-place" if self.in_place else "with output"
        logger.info(f"NormalizeOutput initialized {mode} with {len(channel_activations)} channels")
        for i, activation in enumerate(self.activations):
            if activation is not None:
                logger.debug(f"  Channel {i}: {activation}")

    def setup(self):
        if not self.in_place:
            spec = self.spec[self.input_array].copy()
            self.provides(self.output_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        if self.in_place:
            deps[self.input_array] = request[self.input_array].copy()
        else:
            deps[self.input_array] = request[self.output_array].copy()
        return deps

    def process(self, batch, request):
        input_data = batch[self.input_array].data

        # Convert to torch tensor for activation application
        input_tensor = torch.from_numpy(input_data)

        # Apply activations per channel in-place
        for c, activation in enumerate(self.activations):
            if activation is not None:
                # Apply activation to this channel
                input_data[:, c] = activation(input_tensor[:, c]).numpy()

        if self.in_place:
            # Already modified in place, no need to create new batch
            return gp.Batch()
        else:
            # Create output array
            spec = self.spec[self.input_array].copy()
            spec.roi = request[self.output_array].roi
            output_array = gp.Array(input_data.copy(), spec)

            output = gp.Batch()
            output.arrays[self.output_array] = output_array
            return output
