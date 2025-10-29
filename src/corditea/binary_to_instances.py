import logging
from typing import Optional, Union

import gunpowder as gp
import numpy as np
from scipy.ndimage import label, generate_binary_structure

logger = logging.getLogger(__name__)


class BinaryToInstances(gp.BatchFilter):
    """Convert binary segmentation to instance labels using connected components.

    This node takes a binary segmentation array (where foreground pixels are > 0)
    and converts it to instance labels by running connected components analysis.
    Each connected component gets a unique label starting from 1.

    Arrays with extra dimensions are supported only if they are singleton dimensions
    at the beginning (channels-first format). For example, (1, H, W) for 2D spatial
    data is accepted, but (H, W, 1) or (2, H, W) will raise an error.

    Args:

        source (:class:`ArrayKey`):
            The key of the binary segmentation array to convert.

        target (:class:`ArrayKey`, optional):
            The key to store the instance labels. If None, the source array
            will be modified in place.

        mask (:class:`ArrayKey`, optional):
            The key of an input mask array. Only pixels where the mask is > 0
            will be considered for connected components analysis. If None,
            all non-background pixels are processed.

        connectivity (``int``):
            Connectivity for connected components. For 2D: 1 (4-connectivity) or
            2 (8-connectivity). For 3D: 1 (6-connectivity), 2 (18-connectivity),
            or 3 (26-connectivity). Default is 1.

        background (``int``, optional):
            Value to consider as background. Default is 0.
    """

    def __init__(
        self,
        source: gp.ArrayKey,
        target: Optional[gp.ArrayKey] = None,
        mask: Optional[gp.ArrayKey] = None,
        connectivity: Optional[int] = 1,
        background: Union[int, float] = 0,
    ) -> None:
        assert isinstance(source, gp.ArrayKey)
        self.source = source
        self.background = background
        self.connectivity = connectivity

        if target is None:
            self.target = source
        else:
            assert isinstance(target, gp.ArrayKey)
            self.target = target

        if mask is not None:
            assert isinstance(mask, gp.ArrayKey)
        self.mask: Optional[gp.ArrayKey] = mask

    def setup(self) -> None:
        # Get the source spec
        source_spec = self.spec[self.source]

        # Create target spec with same properties as source
        target_spec = source_spec.copy()

        # Ensure we're not working with interpolatable arrays for instances
        target_spec.interpolatable = False

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

        # Request the mask array if provided
        if self.mask is not None:
            deps[self.mask] = request[self.target].copy()

        return deps

    def process(self, batch: gp.Batch, request: gp.BatchRequest) -> gp.Batch:
        # Get the source array
        source_array = batch.arrays[self.source]
        data = source_array.data

        # Get the mask array if provided
        mask_data: Optional[np.ndarray] = None
        if self.mask is not None:
            mask_data = batch.arrays[self.mask].data

        logger.debug(f"Converting binary segmentation to instances, input shape: {data.shape}")

        # Determine spatial dimensions
        spatial_dims = source_array.spec.roi.dims

        # Handle arrays with extra dimensions (must be singleton channels at the beginning)
        if len(data.shape) > spatial_dims:
            # Check that all non-spatial dimensions are singleton
            channel_dims = len(data.shape) - spatial_dims
            channels_shape = data.shape[:channel_dims]

            if not all(dim == 1 for dim in channels_shape):
                msg = (
                    f"Non-spatial dimensions must be singleton, got shape {data.shape} "
                    f"with {spatial_dims}D spatial data. Non-spatial shape: {channels_shape}"
                )
                raise ValueError(msg)

            # Squeeze out only the singleton channel dimensions
            channel_axes = tuple(range(channel_dims))
            spatial_data = data.squeeze(axis=channel_axes)
            spatial_mask: Optional[np.ndarray] = None
            if mask_data is not None:
                spatial_mask = mask_data.squeeze(axis=channel_axes)

            # Process the spatial data
            processed_spatial, num_features = self._process_single_array(
                spatial_data, spatial_dims, spatial_mask
            )

            # Restore original shape
            output_data = processed_spatial.reshape(data.shape)
        else:
            # Exact spatial dimensions, process directly
            output_data, num_features = self._process_single_array(data, spatial_dims, mask_data)

        # Preserve original data type if it can hold the labels
        # The maximum label value is num_features (since labels go from 0 to num_features)
        if np.issubdtype(data.dtype, np.integer):
            # Check if the original dtype can hold the maximum label
            if num_features <= np.iinfo(data.dtype).max:
                output_data = output_data.astype(data.dtype)
            else:
                logger.warning(
                    f"Original dtype {data.dtype} cannot hold max label {num_features}, "
                    f"using {output_data.dtype}"
                )
        else:
            # For floating point inputs, convert to appropriate integer type
            if num_features <= np.iinfo(np.uint16).max:
                output_data = output_data.astype(np.uint16)
            elif num_features <= np.iinfo(np.uint32).max:
                output_data = output_data.astype(np.uint32)
            # else keep as int64

        logger.debug(
            f"Connected components found: {num_features}, "
            f"output dtype: {output_data.dtype}"
        )

        # Create output array with same spec as source
        target_spec = source_array.spec.copy()
        target_spec.interpolatable = False
        target_spec.dtype = output_data.dtype

        # Create the output batch
        output_batch = gp.Batch()
        output_batch.arrays[self.target] = gp.Array(output_data, target_spec)

        return output_batch

    def _process_single_array(
        self,
        data: np.ndarray,
        spatial_dims: int,
        mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, int]:
        """Process a single spatial array (no channel dimensions)."""

        # Create binary mask (anything not background is foreground)
        binary_mask = data != self.background

        # Apply input mask if provided
        if mask is not None:
            # Only process pixels where both binary_mask and input mask are > 0
            binary_mask = binary_mask & (mask > 0)

        # Generate structure for connectivity
        structure = generate_binary_structure(spatial_dims, self.connectivity)

        # Run connected components
        labeled_array, max_instance = label(binary_mask, structure=structure)

        # Preserve background value in output
        if self.background > 0 and max_instance > 0 and self.background <= max_instance:
            # Collision: shift instance labels up
            offset = max(1, int(self.background) + 1)
            labeled_array = np.where(labeled_array > 0, labeled_array + offset - 1, 0)
            max_instance = max_instance + offset - 1

        # Set background pixels to background value
        if self.background != 0:
            labeled_array = np.where(labeled_array == 0, self.background, labeled_array)

        logger.debug(f"Found {max_instance} connected components")

        return labeled_array, max_instance