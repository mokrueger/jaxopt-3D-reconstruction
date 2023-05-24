import gtsam
import numpy as np
from gtsam import symbol_shorthand, PriorFactorPinholeCameraCal3_S2, GeneralSFMFactorCal3_S2
from gtsam import (LevenbergMarquardtOptimizer, NonlinearFactorGraph, PriorFactorPoint3, Values)

from src.dataset.camera_pose.enums_and_types import CoordinateSystem, TransformationDirection
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset

L = symbol_shorthand.L
X = symbol_shorthand.X


def benchmark_gtsam_bundle_adjustment(dataset: Dataset, add_noise=True, args=None):
    noise_dataset = Dataset.with_noise(dataset) if add_noise else dataset
    graph = NonlinearFactorGraph()

    initial_estimate = Values()
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

    error = 0
    count = 0
    for index, de in enumerate(noise_dataset.datasetEntries, start=0):
        ci = de.camera.camera_intrinsics
        #  calibration = Cal3_S2(ci.focal_x, ci.focal_y, ci.skew_factor, ci.center_x, ci.center_y)
        pose = gtsam.Pose3(
            de.camera.camera_pose
            .in_coordinate_system(CoordinateSystem.COLMAP)
            .in_direction(TransformationDirection.C2W).transformation_translation_matrix
        )
        #  camera = gtsam.gtsam.PinholeCameraCal3Bundler(pose, calibration)

        calibration = gtsam.gtsam.Cal3_S2(ci.focal_x, ci.focal_y, ci.skew_factor, ci.center_x, ci.center_y)
        camera = gtsam.gtsam.PinholeCameraCal3_S2(pose, calibration)

        initial_estimate.insert(X(index), camera)
        #  initial_estimate.insert(X(index), pose)

        if index == 0:
            # Add a prior on pose x1. This indirectly specifies where the origin is.
            # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
            #  pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
            #  factor = PriorFactorPose3(X(index), pose, pose_noise)
            cam_noise = gtsam.noiseModel.Isotropic.Sigma(11, 0.1)
            factor = PriorFactorPinholeCameraCal3_S2(X(index), camera, cam_noise)
            graph.push_back(factor)

        for j, p2d_p3d in enumerate(de.map2d_3d(noise_dataset.points3D_mapped, zipped=True), start=0):
            #  measurement = camera.project(point)
            #  factor = GenericProjectionFactorCal3_S2(
            #      p2d_p3d[0].xy, measurement_noise, X(index), L(p2d_p3d[1].identifier), calibration
            #  )
            error += np.linalg.norm(p2d_p3d[0].xy - camera.project(p2d_p3d[1].xyz)) ** 2
            count += 1
            factor = GeneralSFMFactorCal3_S2(p2d_p3d[0].xy, measurement_noise, X(index), L(p2d_p3d[1].identifier))
            graph.push_back(factor)

    for j, point in enumerate(noise_dataset.points3D):
        if j == 0:
            # Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
            # Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
            # between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
            point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            factor = PriorFactorPoint3(L(point.identifier), point.xyz, point_noise)
            graph.push_back(factor)
        initial_estimate.insert(L(point.identifier), point.xyz)

    initial_estimate.print('Initial Estimates:\n')

    # Optimize the graph and print results
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity('TERMINATION')
        optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        print('Optimizing:')
        result = optimizer.optimize()
    except RuntimeError:
        # TODO: Something here
        return ...

    result.print('Final results:\n')
    print('initial error = {}'.format(graph.error(initial_estimate)))
    print('final error = {}'.format(graph.error(result)))

    """ interpret result """  # TODO: Parse into dataset to visualize
    corresponding_cameras = []
    for index, de in enumerate(dataset.datasetEntries, start=0):
        camera = initial_estimate.atPinholeCameraCal3_S2(X(index))
        corresponding_cameras.append(
            (camera.pose(), camera.calibration(),
             (de.camera.camera_pose.in_direction(TransformationDirection.C2W), de.camera.camera_intrinsics))
        )

    corresponding_points = []
    for p in dataset.points3D:
        corresponding_points.append((initial_estimate.atPoint3(L(dataset.points3D[0].identifier)), p.xyz))

    return ...


if __name__ == "__main__":
    path = "/home/morkru/Downloads/reichstag/dense/sparse/"
    image_path = "/home/morkru/Downloads/reichstag/dense/images"
    ds = load_colmap_dataset(path, image_path, binary=True)
    benchmark_gtsam_bundle_adjustment(ds)
