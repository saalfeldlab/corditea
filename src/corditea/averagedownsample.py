import logging

import gunpowder as gp
from skimage.transform import downscale_local_mean

logger = logging.getLogger(__name__)


class AverageDownSample(gp.BatchFilter):
    """Downsample arrays in a batch by given factors.

    Args:

        source (:class:`ArrayKey`):

            The key of the array to downsample.

        factor (``int`` or ``tuple`` of ``int``):

            The factor to downsample with.

        target (:class:`ArrayKey`):

            The key of the array to store the downsampled ``source``.
    """

    def __init__(self, source, target_voxel_size, target=None):
        assert isinstance(source, gp.ArrayKey)
        self.source = source
        self.target_voxel_size = gp.Coordinate(target_voxel_size)
        if target is None:
            self.target = source
        else:
            assert isinstance(target, gp.ArrayKey)
            self.target = target

    def setup(self):

        self.source_voxel_size = self.get_upstream_provider().spec.array_specs[self.source].voxel_size
        self.factor = self.target_voxel_size / self.source_voxel_size

        spec = self.spec[self.source].copy()
        spec.voxel_size = self.target_voxel_size
        source_roi = spec.roi
        spec.roi = (spec.roi / self.target_voxel_size) * self.target_voxel_size
        logger.debug(f"Updating {source_roi} to {spec.roi}")
        assert self.target_voxel_size % self.source_voxel_size == gp.Coordinate(
            (0,) * len(self.target_voxel_size)
        ), f"{self.target_voxel_size % self.source_voxel_size}"
        assert self.target_voxel_size > self.source_voxel_size
        if not spec.interpolatable:
            msg = "can't use average downsampling for non-interpolatable arrays"
            raise ValueError(msg)

        if self.target == self.source:
            self.updates(self.target, spec)
        else:
            self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):
        # intialize source request with existing request for target
        source_request = request[self.target].copy()
        # correct the voxel size for source
        logger.debug(f"Initializing source request with {source_request}")
        # source_voxel_size = self.spec[self.source].voxel_size
        source_request.voxel_size = self.source_voxel_size
        deps = gp.BatchRequest()
        deps[self.source] = source_request
        return deps

    def process(self, batch, request):
        source = batch.arrays[self.source]
        data = source.data
        src_dtype = data.dtype

        channel_dims = len(data.shape) - source.spec.roi.dims
        factor = (1,) * channel_dims + self.factor
        resampled_data = downscale_local_mean(data, factor).astype(src_dtype)
        logger.debug(f"Downsampling turns shape {data.shape} into {resampled_data.shape}")
        target_spec = source.spec.copy()
        target_spec.roi = gp.Roi(
            source.spec.roi.get_begin(),
            self.target_voxel_size * gp.Coordinate(resampled_data.shape[-self.target_voxel_size.dims :]),
        )
        target_spec.voxel_size = self.target_voxel_size
        logger.debug(f"returning array with spec {target_spec}")

        # create output array
        outputs = gp.Batch()
        outputs.arrays[self.target] = gp.Array(resampled_data, target_spec)

        return outputs
