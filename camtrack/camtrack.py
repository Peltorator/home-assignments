#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from cv2 import solvePnPRansac
from corners import CornerStorage
from _corners import filter_frame_corners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    
    known_view = [known_view_1, known_view_2]

    if known_view[0] is None or known_view[1] is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    pose_to_view = [None for _ in rgb_sequence]
    for i in range(2):
        pose_to_view[known_view[i][0]] = pose_to_view_mat3x4(known_view[i][1])

    corrs = build_correspondences(corner_storage[known_view[0][0]], corner_storage[known_view[1][0]])
    points, corrs_ids, med_cos = triangulate_correspondences(corrs, pose_to_view[known_view[0][0]], pose_to_view[known_view[1][0]], intrinsic_mat, TriangulationParameters(1, 1.1, .1))
    pointCloudBuilder = PointCloudBuilder(corrs_ids, points)

    while True:
        new_views = 0
        for i in range(len(pose_to_view)):
            if pose_to_view[i] is None:
                arr1 = pointCloudBuilder.ids.flatten()
                arr2 = corner_storage[i].ids.flatten()
                i2 = 0
                p3d = []
                p2d = []
                for i1 in range(len(arr1)):
                    while i2 < len(arr2) and arr2[i2] < arr1[i1]:
                        i2 += 1
                    if i2 < len(arr2) and arr2[i2] == arr1[i1]:
                        p3d.append(pointCloudBuilder.points[i1])
                        p2d.append(corner_storage[i].points[i2])
                        i2 += 1
                #print(len(p3d))
                p3d = np.array(p3d)
                p2d = np.array(p2d)
                try:
                    retval, rvec, tvec, inliers = solvePnPRansac(p3d, p2d, intrinsic_mat, None)
                    if inliers is None or len(inliers) == 0:
                        continue
                    new_views += 1
                    pose_to_view[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    fil_frame_cors = filter_frame_corners(corner_storage[i], np.array(inliers, dtype=int).flatten())

                    for i2 in range(len(pose_to_view)):
                        if pose_to_view[i2] is not None:
                            inner_corrs = build_correspondences(corner_storage[i2], fil_frame_cors)
                            if inner_corrs is None or len(inner_corrs) == 0:
                                continue
                            inner_points, inner_corrs_ids, inner_med_cos = triangulate_correspondences(inner_corrs, pose_to_view[i2], pose_to_view[i], intrinsic_mat, TriangulationParameters(1, 1.1, .1))
                            pointCloudBuilder.add_points(inner_corrs_ids, inner_points)
                except Exception:
                    pass
        if new_views == 0:
            break

    for i in range(len(pose_to_view)):
        if i == 0:
            for j in range(len(pose_to_view)):
                if pose_to_view[j] is not None:
                    pose_to_view[i] = pose_to_view[j]
                    break
        else:
            pose_to_view[i] = pose_to_view[i - 1] if pose_to_view[i] is None else pose_to_view[i]

    calc_point_cloud_colors(
        pointCloudBuilder,
        rgb_sequence,
        np.array(pose_to_view),
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = pointCloudBuilder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, pose_to_view))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()



