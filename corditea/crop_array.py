import gunpowder as gp
import logging

logger = logging.getLogger(__name__)


class CropArray(gp.BatchFilter):

    '''Crop an array

    Args:

        key (:class:`ArrayKey`):

            The key of the array to crop.

        cropping_widths_neg (``int`` or ``tuple`` of ``int``):

            The amount to cut away at the beginning.

        cropping_widths_pos (``int`` or ``tuple`` of ``int``):

            The amount to cut away at the end.
    '''

    def __init__(self, key, cropping_widths_neg, cropping_widths_pos):

        assert isinstance(key, gp.ArrayKey)
        self.key = key
        self.cropping_widths_neg = cropping_widths_neg
        self.cropping_widths_pos = cropping_widths_pos

    def setup(self):
        assert self.key in self.spec, ("Asked to crop %s, but is not provided upstream."%self.key)
        assert self.spec[self.key].roi is not None, ("Asked to crop %s, but upstream provider doesn't have a ROI for it"
                                                      "."%self.key)

    def prepare(self, request):
        cropped_roi = request[self.key].roi.grow(gp.Coordinate(self.cropping_widths_neg),
                                                 gp.Coordinate(self.cropping_widths_pos))

        request[self.key].roi = cropped_roi

    def process(self, batch, request):
        request_roi = request[self.key].roi
        batch.arrays[self.key] = batch.arrays[self.key].crop(request_roi)
