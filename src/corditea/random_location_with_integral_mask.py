import logging
import math
from random import choice, randint, random

import gunpowder as gp
import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class RandomLocationWithIntegralMask(gp.BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream
    provider.

    The random location is chosen such that the batch request ROI lies entirely
    inside the provider's ROI.

    If ``min_masked`` and ``mask`` are set, only batches are returned that have
    at least the given ratio of masked-in voxels. This is in general faster
    than using the :class:`Reject` node, at the expense of storing an integral
    array of the complete mask.

    If ``ensure_nonempty`` is set to a :class:`PointsKey`, only batches are
    returned that have at least one point of this point collection within the
    requested ROI.

    Additional tests for randomly picked locations can be implemented by
    subclassing and overwriting of :func:`accepts`. This method takes the
    randomly shifted request that meets all previous criteria (like
    ``min_masked`` and ``ensure_nonempty``) and should return ``True`` if the
    request is acceptable.

    Args:

        min_masked (``float``, optional):

            If non-zero, require that the random sample contains at least that
            ratio of masked-in voxels.

        mask (:class:`ArrayKey`, optional):

            The array to use for mask checks.

        ensure_nonempty (:class:`PointsKey`, optional):

            Ensures that when finding a random location, a request for
            ``ensure_nonempty`` will contain at least one point.

        p_nonempty (``float``, optional):

            If ``ensure_nonempty`` is set, it defines the probability that a
            request for ``ensure_nonempty`` will contain at least one point.
            Default value is 1.0.
    '''

    def __init__(self, min_masked=0, integral_mask=None, ensure_nonempty=None, p_nonempty=1.0):

        self.min_masked = min_masked
        self.mask_integral = integral_mask
        self.mask_spec = None
        self.ensure_nonempty = ensure_nonempty
        self.points = None
        self.p_nonempty = p_nonempty
        self.upstream_spec = None
        self.random_shift = None

    def setup(self):
        upstream = self.get_upstream_provider()
        self.upstream_spec = upstream.spec
        if self.mask_integral and self.min_masked > 0:
            assert self.mask_integral in self.upstream_spec, "Upstream provider does not have %s" % self.mask_integral
            self.mask_spec = self.upstream_spec.array_specs[self.mask_integral]

            # logger.info("requesting complete mask...")

            # mask_request = BatchRequest({self.mask: self.mask_spec})
            # mask_batch = upstream.request_batch(mask_request)

            # logger.info("allocating mask integral array...")

            # mask_data = mask_batch.arrays[self.mask].data
            # mask_integral_dtype = np.uint64
            # logger.debug("mask size is %s", mask_data.size)
            # if mask_data.size < 2**32:
            #     mask_integral_dtype = np.uint32
            # if mask_data.size < 2**16:
            #     mask_integral_dtype = np.uint16
            # logger.debug("chose %s as integral array dtype", mask_integral_dtype)

            # self.mask_integral = np.array(mask_data > 0, dtype=mask_integral_dtype)
            # self.mask_integral = integral_image(self.mask_integral)

        if self.ensure_nonempty:

            assert self.ensure_nonempty in self.upstream_spec, (
                "Upstream provider does not have %s" % self.ensure_nonempty
            )
            points_spec = self.upstream_spec.points_specs[self.ensure_nonempty]

            logger.info("requesting all %s points...", self.ensure_nonempty)

            points_request = gp.BatchRequest({self.ensure_nonempty: points_spec})
            points_batch = upstream.request_batch(points_request)

            self.points = KDTree([p.location for p in points_batch[self.ensure_nonempty].data.values()])

            logger.info("retrieved %d points", len(self.points.data))

        # clear bounding boxes of all provided arrays and points --
        # RandomLocation does not have limits (offsets are ignored)
        for key, spec in self.spec.items():
            spec.roi.set_shape(None)
            self.updates(key, spec)

    def prepare(self, request):

        logger.debug("request: %s", request.array_specs)
        logger.debug("my spec: %s", self.spec)

        shift_roi = self.__get_possible_shifts(request)
        if request.array_specs.keys():

            lcm_voxel_size = self.spec.get_lcm_voxel_size(request.array_specs.keys())
            shift_roi = shift_roi.snap_to_grid(lcm_voxel_size, mode='shrink')
            lcm_shift_roi = shift_roi / lcm_voxel_size
            logger.debug("lcm voxel size: %s", lcm_voxel_size)

            logger.debug("restricting random locations to multiples of voxel size %s", lcm_voxel_size)

        else:

            lcm_voxel_size = gp.Coordinate((1,) * shift_roi.dims())
            lcm_shift_roi = shift_roi

        random_shift = self.__select_random_shift(request, lcm_shift_roi, lcm_voxel_size)

        self.random_shift = random_shift
        self.__shift_request(request, random_shift)

    def process(self, batch, request):

        # reset ROIs to request
        for array_key, spec in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi
        for points_key, spec in request.points_specs.items():
            batch.points[points_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for points_key in request.points_specs.keys():
            for point_id, _ in batch.points[points_key].data.items():
                batch.points[points_key].data[point_id].location -= self.random_shift

    def accepts(self, request):
        '''Should return True if the randomly chosen location is acceptable
        (besided meeting other criteria like ``min_masked`` and/or
        ``ensure_nonempty``). Subclasses can overwrite this method to implement
        additional tests for acceptable locations.'''

        return True

    def __get_possible_shifts(self, request):

        total_shift_roi = None

        for key, spec in request.items():

            request_roi = spec.roi
            provided_roi = self.upstream_spec[key].roi

            shift_roi = provided_roi.shift(-request_roi.get_begin()).grow(
                (0,) * request_roi.dims(), -request_roi.get_shape()
            )

            if total_shift_roi is None:
                total_shift_roi = shift_roi
            else:
                total_shift_roi = total_shift_roi.intersect(shift_roi)

        logger.debug("valid shifts for request in " + str(total_shift_roi))

        assert not total_shift_roi.unbounded(), (
            "Can not pick a random location, intersection of upstream ROIs is " "unbounded."
        )
        assert total_shift_roi.size() > 0, "Can not satisfy batch request, no location covers all requested " "ROIs."

        return total_shift_roi

    def __select_random_shift(self, request, lcm_shift_roi, lcm_voxel_size):
        ensure_points = self.ensure_nonempty is not None and random() <= self.p_nonempty

        while True:

            if ensure_points:
                random_shift = self.__select_random_location_with_points(request, lcm_shift_roi, lcm_voxel_size)
            else:
                random_shift = self.__select_random_location(lcm_shift_roi, lcm_voxel_size)

            logger.debug("random shift: " + str(random_shift))

            if not self.__is_min_masked(random_shift, request):
                logger.debug("random location does not meet 'min_masked' criterium")
                continue

            if not self.__accepts(random_shift, request):
                logger.debug("random location does not meet user-provided criterium")
                continue

            return random_shift

    def __is_min_masked(self, random_shift, request):
        if not self.mask_integral or self.min_masked == 0:
            return True

        # get randomly chosen mask ROI
        request_mask_roi = request.array_specs[self.mask_integral].roi
        request_mask_roi = request_mask_roi.shift(random_shift)

        # get number of masked-in voxels
        num_masked_in = self.__integrate(
            request_mask_roi.get_begin(), request_mask_roi.get_end() - self.spec[self.mask_integral].voxel_size
        )[0]
        mask_ratio = float(num_masked_in) / (
            request_mask_roi.size() / np.prod(self.spec[self.mask_integral].voxel_size)
        )
        logger.debug("mask ratio is %f", mask_ratio)
        return mask_ratio >= self.min_masked

    def __integrate(self, start, end):
        """Use an integral image to integrate over a given window.

        Parameters
        ----------
        ii : ndarray
            Integral image.
        start : List of tuples, each tuple of length equal to dimension of `ii`
            Coordinates of top left corner of window(s).
            Each tuple in the list contains the starting row, col, ... index
            i.e `[(row_win1, col_win1, ...), (row_win2, col_win2,...), ...]`.
        end : List of tuples, each tuple of length equal to dimension of `ii`
            Coordinates of bottom right corner of window(s).
            Each tuple in the list containing the end row, col, ... index i.e
            `[(row_win1, col_win1, ...), (row_win2, col_win2, ...), ...]`.

        Returns
        -------
        S : scalar or ndarray
            Integral (sum) over the given window(s).


        Examples
        --------
        >>> arr = np.ones((5, 6), dtype=np.float)
        >>> ii = integral_image(arr)
        >>> integrate(ii, (1, 0), (1, 2))  # sum from (1, 0) to (1, 2)
        array([ 3.])
        >>> integrate(ii, [(3, 3)], [(4, 5)])  # sum from (3, 3) to (4, 5)
        array([ 6.])
        >>> # sum from (1, 0) to (1, 2) and from (3, 3) to (4, 5)
        >>> integrate(ii, [(1, 0), (3, 3)], [(1, 2), (4, 5)])
        array([ 3.,  6.])
        """
        start = np.atleast_2d(np.array(start))
        end = np.atleast_2d(np.array(end))
        rows = start.shape[0]

        if np.any((end - start) < 0):
            raise IndexError('end coordinates must be greater or equal to start')

        # bit_perm is the total number of terms in the expression
        # of S. For example, in the case of a 4x4 2D image
        # sum of image from (1,1) to (2,2) is given by
        # S = + ii[2, 2]
        #     - ii[0, 2] - ii[2, 0]
        #     + ii[0, 0]
        # The total terms = 4 = 2 ** 2(dims)

        S = np.zeros(rows)
        bit_perm = 2 ** len(self.spec[self.mask_integral].voxel_size)
        width = len(bin(bit_perm - 1)[2:])

        # Sum of a (hyper)cube, from an integral image is computed using
        # values at the corners of the cube. The corners of cube are
        # selected using binary numbers as described in the following example.
        # In a 3D cube there are 8 corners. The corners are selected using
        # binary numbers 000 to 111. Each number is called a permutation, where
        # perm(000) means, select end corner where none of the coordinates
        # is replaced, i.e ii[end_row, end_col, end_depth]. Similarly, perm(001)
        # means replace last coordinate by start - 1, i.e
        # ii[end_row, end_col, start_depth - 1], and so on.
        # Sign of even permutations is positive, while those of odd is negative.
        # If 'start_coord - 1' is -ve it is labeled bad and not considered in
        # the final sum.

        for i in range(bit_perm):  # for all permutations
            # boolean permutation array eg [True, False] for '10'
            binary = bin(i)[2:].zfill(width)
            bool_mask = [bit == '1' for bit in binary]
            sign = (-1) ** sum(bool_mask)  # determine sign of permutation
            bad = [
                np.any(
                    (
                        (start[r] - self.spec[self.mask_integral].voxel_size) * bool_mask
                        + end[r] * np.invert(bool_mask)
                        - self.mask_spec.roi.get_offset()
                    )
                    < 0
                )
                for r in range(rows)
            ]  # find out bad start rows

            corner_points = (end * (np.invert(bool_mask))) + (
                (start - self.spec[self.mask_integral].voxel_size) * bool_mask
            )  # find corner for each row

            ii_cp = []
            for cp, b in zip(corner_points, bad):
                if b:
                    ii_cp.append(0)
                else:
                    corner_spec_roi = self.spec[self.mask_integral].copy()
                    corner_spec_roi.roi = gp.Roi(offset=tuple(cp), shape=corner_spec_roi.voxel_size)
                    corner_request = gp.BatchRequest({self.mask_integral: corner_spec_roi})
                    corner = self.get_upstream_provider().request_batch(corner_request)
                    ii_cp.append(corner.arrays[self.mask_integral].data[0, 0, 0])
            S += [sign * ii_cp[r] for r in range(rows)]  # add only good rows
        return S

    def __accepts(self, random_shift, request):

        # create a shifted copy of the request
        shifted_request = request.copy()
        self.__shift_request(shifted_request, random_shift)

        return self.accepts(shifted_request)

    def __shift_request(self, request, shift):

        # shift request ROIs
        for specs_type in [request.array_specs, request.points_specs]:
            for key, spec in specs_type.items():
                roi = spec.roi.shift(shift)
                specs_type[key].roi = roi

    def __select_random_location_with_points(self, request, lcm_shift_roi, lcm_voxel_size):

        request_points_roi = request[self.ensure_nonempty].roi

        while True:

            # How to pick shifts that ensure that a randomly chosen point is
            # contained in the request ROI:
            #
            #
            # request          point
            # [---------)      .
            # 0        +10     17
            #
            #         least shifted to contain point
            #         [---------)
            #         8        +10
            #         ==
            #         point-request.begin-request.shape+1
            #
            #                  most shifted to contain point:
            #                  [---------)
            #                  17       +10
            #                  ==
            #                  point-request.begin
            #
            #         all possible shifts
            #         [---------)
            #         8        +10
            #         ==
            #         point-request.begin-request.shape+1
            #                   ==
            #                   request.shape
            #
            # In the most shifted scenario, it could happen that the point lies
            # exactly at the lower boundary (17 in the example). This will cause
            # trouble if later we mirror the batch. The point would end up lying
            # on the other boundary, which is exclusive and thus not part of the
            # ROI. Therefore, we have to ensure that the point is well inside
            # the shifted ROI, not just on the boundary:
            #
            #         all possible shifts
            #         [--------)
            #         8       +9
            #                 ==
            #                 request.shape-1

            # pick a random point
            point = choice(self.points.data)

            logger.debug("select random point at %s", point)

            # get the lcm voxel that contains this point
            lcm_location = gp.Coordinate(point / lcm_voxel_size)
            logger.debug("belongs to lcm voxel %s", lcm_location)

            # mark all dimensions in which the point lies on the lower boundary
            # of the lcm voxel
            on_lower_boundary = lcm_location * lcm_voxel_size == point
            logger.debug("lies on the lower boundary of the lcm voxel in dimensions %s", on_lower_boundary)

            # for each of these dimensions, we have to change the shape of the
            # shift ROI using the following correction
            lower_boundary_correction = gp.Coordinate((-1 if o else 0 for o in on_lower_boundary))
            logger.debug("lower bound correction for shape of shift ROI %s", lower_boundary_correction)

            # get the request ROI's shape in lcm
            lcm_roi_begin = request_points_roi.get_begin() / lcm_voxel_size
            lcm_roi_shape = request_points_roi.get_shape() / lcm_voxel_size
            logger.debug("Point request ROI: %s", request_points_roi)
            logger.debug("Point request lcm ROI shape: %s", lcm_roi_shape)

            # get all possible starting points of lcm_roi_shape that contain
            # lcm_location
            lcm_shift_roi_begin = lcm_location - lcm_roi_begin - lcm_roi_shape + gp.Coordinate((1,) * len(lcm_location))
            lcm_shift_roi_shape = lcm_roi_shape + lower_boundary_correction
            lcm_point_shift_roi = gp.Roi(lcm_shift_roi_begin, lcm_shift_roi_shape)
            logger.debug("lcm point shift roi: %s", lcm_point_shift_roi)

            # intersect with total shift ROI
            if not lcm_point_shift_roi.intersects(lcm_shift_roi):
                logger.debug(
                    "reject random shift, random point %s shift ROI %s does " "not intersect total shift ROI %s",
                    point,
                    lcm_point_shift_roi,
                    lcm_shift_roi,
                )
                continue
            lcm_point_shift_roi = lcm_point_shift_roi.intersect(lcm_shift_roi)

            # select a random shift from all possible shifts
            random_shift = self.__select_random_location(lcm_point_shift_roi, lcm_voxel_size)
            logger.debug("random shift: %s", random_shift)

            # count all points inside the shifted ROI
            points = self.__get_points_in_roi(request_points_roi.shift(random_shift))
            assert point in points, "Requested batch to contain point %s, but got points " "%s" % (point, points)
            num_points = len(points)

            # accept this shift with p=1/num_points
            #
            # This is to compensate the bias introduced by close-by points.
            accept = random() <= 1.0 / num_points
            if accept:
                return random_shift

    def __select_random_location(self, lcm_shift_roi, lcm_voxel_size):

        # select a random point inside ROI
        random_shift = gp.Coordinate(
            randint(int(begin), int(end - 1)) for begin, end in zip(lcm_shift_roi.get_begin(), lcm_shift_roi.get_end())
        )

        random_shift *= lcm_voxel_size

        return random_shift

    def __get_points_in_roi(self, roi):

        points = []

        center = roi.get_center()
        radius = math.ceil(float(max(roi.get_shape())) / 2)
        candidates = self.points.query_ball_point(center, radius, p=np.inf)

        for i in candidates:
            if roi.contains(self.points.data[i]):
                points.append(self.points.data[i])

        return np.array(points)
