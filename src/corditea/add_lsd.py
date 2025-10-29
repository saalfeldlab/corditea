from __future__ import absolute_import
from lsd_lite import get_lsds
from gunpowder import BatchFilter, Array, BatchRequest, Batch, Coordinate
import logging
import numpy as np
from typing import Literal, Union

logger = logging.getLogger(__name__)


class AddLSD(BatchFilter):

    """Create a local segmentation shape descriptor to each voxel.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.

        lsds_mask (:class:`ArrayKey`, optional): The array to store a binary mask
            the size of the descriptors. Background voxels, which do not have a
            descriptor, will be set to 0. This can be used as a loss scale
            during training, such that background is ignored.

        labels_mask (:class:`ArrayKey`, optional): The array to use as a mask
            for labels.

        background_mode (Literal["exclude", "zero", "label"]): How to handle background
            voxels in the segmentation. "exclude" will add background voxels to the
            mask to be ignored during training.

        background_value (int or tuple of int): The label value(s) to consider as
            background. Default is 255.

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel. Default is 5.0.

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factor. Default is 1 (no
            downsampling).
    """

    def __init__(
        self,
        segmentation,
        descriptor,
        lsds_mask=None,
        labels_mask=None,
        background_mode: Literal["exclude", "zero", "label"] = "exclude",
        background_value: Union[int, tuple[int, ...]] = 255,
        sigma=5.0,
        downsample=1,
    ):

        self.segmentation = segmentation
        self.descriptor = descriptor
        self.lsds_mask = lsds_mask
        self.labels_mask = labels_mask
        self.sigma = sigma


        self.downsample = downsample
        self.voxel_size = None
        self.background_mode = background_mode
        if isinstance(background_value, int):
            background_value = (background_value,)
        self.background_value = background_value


    def setup(self):

        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32

        self.voxel_size = spec.voxel_size
        self.provides(self.descriptor, spec)

        if self.lsds_mask:
            self.provides(self.lsds_mask, spec.copy())

        # Context will be computed in prepare() based on actual dimensions
        self.enable_autoskip()
        
    def prepare(self, request):
        deps = BatchRequest()

        dims = len(request[self.descriptor].roi.get_shape())

        # Handle sigma dimensions
        if isinstance(self.sigma, (float, int)):
            sigma = (self.sigma,) * dims
        else:
            sigma = self.sigma
            if len(sigma) != dims:
                raise ValueError(f"Sigma tuple length ({len(sigma)}) must match spatial dimensions ({dims})")
        context = tuple(s*3 for s in sigma)
        # increase segmentation ROI to fit Gaussian
        context_roi = request[self.descriptor].roi.grow(context, context)

        # ensure context roi is multiple of voxel size
        context_roi = context_roi.snap_to_grid(self.voxel_size, mode="shrink")

        grown_roi = request[self.descriptor].roi.union(context_roi)

        # If downsampling, ensure ROI dimensions are divisible by downsample factor
        if self.downsample > 1:
            # Work in voxel space to check divisibility
            roi_voxel_shape = grown_roi.get_shape() / self.voxel_size

            downsample_padding_voxels = []
            for voxel_dim in roi_voxel_shape:
                remainder = int(voxel_dim) % self.downsample
                if remainder != 0:
                    pad_needed = self.downsample - remainder
                    downsample_padding_voxels.append(pad_needed)
                else:
                    downsample_padding_voxels.append(0)

            # Convert back to world coordinates and apply padding
            if any(p > 0 for p in downsample_padding_voxels):
                downsample_padding_world = Coordinate(downsample_padding_voxels) * self.voxel_size
                grown_roi = grown_roi.grow(downsample_padding_world, Coordinate([0] * len(downsample_padding_world)))

        deps[self.segmentation] = request[self.descriptor].copy()
        deps[self.segmentation].roi = grown_roi

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.segmentation].copy()

        return deps

    def process(self, batch, request):


        dims = len(self.voxel_size)

        segmentation_array = batch[self.segmentation]

        # get voxel roi of requested descriptors
        # this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi
        descriptor_roi = request[self.descriptor].roi
        voxel_roi_in_seg = (
            seg_roi.intersect(descriptor_roi) - seg_roi.get_offset()
        ) / self.voxel_size

        crop = voxel_roi_in_seg.get_bounding_box()
        labels = list(np.unique(segmentation_array.data))
        if self.background_mode == "exclude" or self.background_mode == "zero":
            labels = [l for l in labels if l not in self.background_value]
        elif self.background_mode == "label":
            # Relabel all background values to a single label
            unified_bg_label = self.background_value[0]
            # Ensure unified label is in the labels list
            if unified_bg_label not in labels:
                labels.append(unified_bg_label)
            for bg_val in self.background_value[1:]:
                if bg_val in labels:
                    segmentation_array.data[segmentation_array.data == bg_val] = unified_bg_label
                    labels.remove(bg_val)
        if 0 in labels:
            new_label = max(labels) + 1
            segmentation_array.data[segmentation_array.data == 0] = new_label
            labels.remove(0)
            labels.append(new_label)

        # Prepare segmentation data for get_lsds
        # Remove singleton dimensions to match voxel_size dimensions
        seg_data = segmentation_array.data

        # If segmentation has more dimensions than voxel_size, squeeze singleton dims
        if seg_data.ndim > dims:
            # Find singleton dimensions at the beginning
            leading_singletons = 0
            for i in range(seg_data.ndim - dims):
                if seg_data.shape[i] == 1:
                    leading_singletons += 1
                else:
                    break

            if leading_singletons > 0:
                # Squeeze leading singleton dimensions
                squeeze_axes = tuple(range(leading_singletons))
                seg_data = np.squeeze(seg_data, axis=squeeze_axes)
                logger.debug(f"Squeezed segmentation from {segmentation_array.data.shape} to {seg_data.shape}")

        descriptor = get_lsds(
            segmentation=seg_data,
            sigma=self.sigma,
            voxel_size=self.voxel_size,
            downsample=self.downsample,
            labels=labels
        )

        # get_lsds returns (channels, spatial...) format, so we need to slice only spatial dimensions
        spatial_slices = (slice(None),) + voxel_roi_in_seg.to_slices()
        descriptor = descriptor[spatial_slices]

        # create descriptor array
        descriptor_spec = self.spec[self.descriptor].copy()
        descriptor_spec.roi = request[self.descriptor].roi.copy()
        descriptor_array = Array(descriptor, descriptor_spec)

        # Create new batch for descriptor:
        output = Batch()

        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:

            if self.labels_mask:

                mask = self._create_mask(batch, self.labels_mask, descriptor, crop)

            else:
                mask = np.ones_like(descriptor, dtype=np.float32)
            if self.background_mode == "exclude":
                seg_crop = segmentation_array.data[voxel_roi_in_seg.to_slices()]
                for bv in self.background_value:
                    # Create spatial mask and broadcast across all channels
                    spatial_mask = seg_crop == bv
                    # Multiply mask by inverted spatial mask (broadcast automatically)
                    mask = mask * (~spatial_mask)[None, ...]
            output[self.lsds_mask] = Array(
                mask.astype(descriptor.dtype), descriptor_spec.copy()
            )

        output[self.descriptor] = descriptor_array

        return output

    def _create_mask(self, batch, mask, lsds, crop):

        mask = batch.arrays[mask].data

        mask = np.array([mask] * lsds.shape[0])

        mask = mask[(slice(None),) + crop]

        return mask