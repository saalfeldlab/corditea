import copy
import logging
import random

import gunpowder as gp

logger = logging.getLogger(__name__)


class RejectEfficiently(gp.BatchFilter):
    '''Reject batches based on the masked-in vs. masked-out ratio.
    Args:

        mask (:class:`ArrayKey`):

            The mask to use.

        min_masked (``float``, optional):


            The minimal required ratio of masked-in vs. masked-out voxels.
            Defaults to 0.5.

        reject_probability (``float``, optional):

            The probability by which a batch that is not valid (less than
            min_masked) is actually rejected. Defaults to 1., i.e. strict
            rejection.
    '''

    def __init__(self, mask, min_masked=0.5, reject_probability=1.0):

        self.mask = mask
        self.min_masked = min_masked
        self.reject_probability = reject_probability

    def setup(self):

        assert self.mask in self.spec, "Reject can only be used if %s is provided" % self.mask
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = gp.profiling.Timing(self)
        timing.start()

        assert self.mask in request, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        while not have_good_batch:
            upstream_spec = self.upstream_provider.spec
            mask_spec = upstream_spec.array_specs[self.mask]
            mask_request = gp.BatchRequest({self.mask: mask_spec})
            logger.info("requesting only mask: ", mask_spec)
            mask_batch = self.upstream_provider.request_batch(mask_request)

            mask_ratio = mask_batch.arrays[self.mask].data.mean()
            have_good_batch = mask_ratio > self.min_masked

            if not have_good_batch and self.reject_probability < 1.0:
                have_good_batch = random.random() > self.reject_probability

            if not have_good_batch:

                logger.debug("reject batch with mask ratio %f at %s", mask_ratio, mask_batch.arrays[self.mask].spec.roi)

                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning(
                        "rejected %d batches, been waiting for a good one " "since %ds",
                        num_rejected,
                        report_next_timeout,
                    )
                    report_next_timeout *= 2
            else:

                logger.debug(
                    "accepted batch with mask ratio %f at %s", mask_ratio, mask_batch.arrays[self.mask].spec.roi
                )
        batch = self.upstream_provider.request_batch(request)
        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
