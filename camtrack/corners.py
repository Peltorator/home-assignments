#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import math

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

MAX_CORNERS = 3000
PYRAMID_DEPTH = 3
WIN_SIZE = (21, 21)
BLOCK_SIZE = 7

def filter_old_corners(ids, points, sizes, depth, image_0_pyramid, image_1_pyramid):
    next_pts = None
    status = None
    err = None
    prev_pts = None
    prev_status = None
    prev_err = None

    for cur_depth in range(depth - 1, -1, -1):
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            image_0_pyramid[cur_depth],
            image_1_pyramid[cur_depth],
            np.asarray(points, dtype=np.float32) / (2 ** cur_depth),
            None,
            status,
            err,
            WIN_SIZE,
            PYRAMID_DEPTH,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.002)
        )
        prev_pts, prev_status, prev_err = cv2.calcOpticalFlowPyrLK(
            image_1_pyramid[cur_depth],
            image_0_pyramid[cur_depth],
            np.asarray(next_pts, dtype=np.float32) / (2 ** cur_depth),
            None,
            status,
            err,
            WIN_SIZE,
            PYRAMID_DEPTH,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.002)
        )

    status = status.ravel()
    prev_status = prev_status.ravel()

    distances = np.array([math.sqrt(x * x + y * y) for (x, y) in np.reshape(points - prev_pts, (-1, 2))])

    good_corners = (status == 1) & (prev_status == 1) & (distances < 0.7)
    return (list(np.asarray(ids)[good_corners]), list(np.asarray(next_pts)[good_corners]), list(np.asarray(sizes)[good_corners]))

def make_points_mask(points, sizes, height, width):
    mask = np.full((height, width), 255).astype(np.uint8)
    for i in range(len(points)):
        x, y = points[i]
        size = sizes[i]
        mask = cv2.circle(mask, (int(x), int(y)), int(size), 0, -1)
    return mask


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = None
    image_0_pyramid = None
    ids = []
    points = []
    sizes = []
    height = frame_sequence.frame_shape[0]
    width = frame_sequence.frame_shape[1]
    cur_id = 0
    first_frame = True

    for frame, image_1 in enumerate(frame_sequence):
        image_1 = (image_1 * 255).astype(np.uint8);
        depth, image_1_pyramid = cv2.buildOpticalFlowPyramid(image_1, WIN_SIZE, PYRAMID_DEPTH, None, False)
        if len(points) > 0:
            ids, points, sizes = filter_old_corners(ids, points, sizes, depth, image_0_pyramid, image_1_pyramid)

        mask = make_points_mask(points, sizes, height, width)
        for depth in range(len(image_1_pyramid)):
            new_corners = cv2.goodFeaturesToTrack(
                image_1_pyramid[depth],
                MAX_CORNERS - len(points),
                0.00075 if first_frame else 0.075,
                8 << depth,
                mask=mask,
                blockSize=BLOCK_SIZE
            )
            if new_corners is not None:
                new_corners = new_corners.reshape(-1, 2).astype(np.float32)
                for (x, y) in new_corners:
                    if len(points) < MAX_CORNERS:
                        if mask[int(y), int(x)] != 0:
                            ids.append(cur_id)
                            cur_id += 1
                            points.append((x * (2 ** depth), y * (2 ** depth)))
                            sizes.append(BLOCK_SIZE << depth)
                            cv2.circle(mask, (int(x), int(y)), BLOCK_SIZE, 0, -1)
            mask = cv2.pyrDown(mask).astype(np.uint8)


        builder.set_corners_at_frame(frame, FrameCorners(np.array(ids), np.array(points), np.array(sizes)))
        image_0 = image_1
        image_0_pyramid = image_1_pyramid
        first_frame = False


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
