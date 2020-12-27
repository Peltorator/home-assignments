#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from cv2 import solvePnPRansac, findEssentialMat, findHomography, decomposeEssentialMat, RANSAC
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
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences,
    eye3x4
)

INF = 1e9
TrPams = TriangulationParameters(6, 1.1, 0.1)

def test_corrs(corrs):
    return len(corrs[0]) >= 50

def check_mask_essential(mask):
    return sum(mask) <= len(mask) * 0.8

def check_mask_homography(mask):
    return sum(mask) <= len(mask) * 0.8

def get_quality(c1, c2, intrinsic_mat):
    corrs = build_correspondences(c1, c2)
    if not test_corrs(corrs):
        return -INF, None, None
 
    matE, mask2 = findEssentialMat(corrs.points_1, corrs.points_2, intrinsic_mat)
    mask2 = mask2.flatten()
    if matE is None or (not check_mask_essential(mask2)):
        return -INF, None, None
    filtered_corrs = Correspondences(corrs.ids[mask2], corrs.points_1[mask2], corrs.points_2[mask2])

    matH, mask = findHomography(filtered_corrs.points_1, filtered_corrs.points_2, RANSAC)
    mask = mask.flatten()
    if not check_mask_homography(mask):
        return -INF, None, None

    R1, R2, t = decomposeEssentialMat(matE)
    vm1 = None
    vm2 = None
    quality = -INF
    for M1, M2 in ((R1, t), (R1, -t), (R2, t), (R2, -t)):
        M = pose_to_view_mat3x4(Pose(M1.transpose(), M1.transpose() @ M2))
        N = eye3x4()
        points, corrs_ids, med_cos = triangulate_correspondences(filtered_corrs, N, M, intrinsic_mat, TrPams)
        if quality < len(points):
            vm1 = view_mat3x4_to_pose(N)
            vm2 = view_mat3x4_to_pose(M)
            quality = len(points)
    return quality, vm1, vm2

GOOD_ENOUGH = 1000

def get_known_views(intrinsic_mat, corner_storage):
    ans1 = None
    ans2 = None
    best_quality = -INF
    for i, ci in enumerate(corner_storage):
        for j in range(i + 1, len(corner_storage)):
            cj = corner_storage[j]
            cur_qual, vm1, vm2 = get_quality(ci, cj, intrinsic_mat)
            print('Frames {0} and {1}. Quality: {2}'.format(i, j, cur_qual))
            if cur_qual > GOOD_ENOUGH:
                return ((i, vm1), (j, vm2))
            elif cur_qual > best_quality:
                ans1 = (i, vm1)
                ans2 = (j, vm2)
                best_quality = cur_qual
    return ans1, ans2


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    
    known_view = [known_view_1, known_view_2]

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view[0] is None or known_view[1] is None:
        print('No known views. Finding our own.')
        known_view[0], known_view[1] = get_known_views(intrinsic_mat, corner_storage)
        print('Found known views: {0}, {1}'.format(known_view[0][0], known_view[1][0]))

    if known_view[0][0] > known_view[1][0]:
        known_view[0], known_view[1] = known_view[1], known_view[0]

    pose_to_view = [None for _ in rgb_sequence]
    for i in range(2):
        pose_to_view[known_view[i][0]] = pose_to_view_mat3x4(known_view[i][1])

    corrs = build_correspondences(corner_storage[known_view[0][0]], corner_storage[known_view[1][0]])
    points, corrs_ids, med_cos = triangulate_correspondences(corrs, pose_to_view[known_view[0][0]], pose_to_view[known_view[1][0]], intrinsic_mat, TrPams)
    pointCloudBuilder = PointCloudBuilder(corrs_ids, points)

    while True:
        found = sum(1 for mat in pose_to_view if mat is not None)
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
                    print('inliers: {0} for frame {1}. There are {2} points in cloud. Processing frame {3} out of {4}'.format(len(inliers), i, len(pointCloudBuilder.points), found + new_views, len(pose_to_view)))
                    new_views += 1
                    pose_to_view[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    fil_frame_cors = filter_frame_corners(corner_storage[i], np.array(inliers, dtype=int).flatten())

                    for i2 in range(len(pose_to_view)):
                        if pose_to_view[i2] is not None:
                            inner_corrs = build_correspondences(corner_storage[i2], fil_frame_cors)
                            if inner_corrs is None or len(inner_corrs) == 0:
                                continue
                            inner_points, inner_corrs_ids, inner_med_cos = triangulate_correspondences(inner_corrs, pose_to_view[i2], pose_to_view[i], intrinsic_mat, TrPams)
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

