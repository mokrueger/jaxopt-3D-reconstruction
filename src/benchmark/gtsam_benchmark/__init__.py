import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from gtsam import symbol_shorthand, SmartProjectionPose3Factor, PriorFactorPinholeCameraCal3_S2, \
    GeneralSFMFactorCal3_S2

from dataset.camera_pose import CoordinateSystem, TransformationDirection

L = symbol_shorthand.L
X = symbol_shorthand.X

from gtsam.examples import SFMdata
from gtsam import (Cal3_S2, LevenbergMarquardtOptimizer, Marginals,
                   NonlinearFactorGraph, PinholeCameraCal3_S2, PriorFactorPoint3, PriorFactorPose3, Values)
from gtsam.utils import plot

from dataset import Dataset
from dataset.loaders.colmap_dataset_loader import load_colmap_dataset


def _prepare_dataset(dataset):  # TODO: Perhaps integrate into dataset
    mapping = {}
    for index, e in enumerate(dataset.datasetEntries):
        mapping.update({index: e.map2d_3d_np(dataset.points3D_mapped, zipped=False)})
    return mapping


def _prepare_cameras(dataset):  # TODO: Perhaps integrate into dataset
    mapping = {}
    for index, e in enumerate(dataset.datasetEntries):
        mapping.update({index: e.camera})
    return mapping


def _prepare_gtsam_cameras(mapping_cameras, factor_graph):  # TODO: use refine_focal_length stuff
    mapping = {}
    for image_id, c in list(mapping_cameras.items()):
        # Note: by default colmap uses focal length of 1.2*max(width, height) to start
        colmap_camera = pycolmap.Camera(model="PINHOLE", width=c.width, height=c.height,  # TODO: could be radial with s
                                        params=[c.camera_intrinsics.focal_x, c.camera_intrinsics.focal_y,
                                                c.camera_intrinsics.center_x, c.camera_intrinsics.center_y])
        mapping[image_id] = colmap_camera
    return mapping


def benchmark_gtsam_absolute_pose(dataset: Dataset, args=None):
    pass


def ttest():
    """
    GTSAM Copyright 2010, Georgia Tech Research Corporation,
    Atlanta, Georgia 30332-0415
    All Rights Reserved
    Authors: Frank Dellaert, et al. (see THANKS for the full author list)

    See LICENSE for the license information

    A structure-from-motion problem on a simulated dataset
    """

    """
    Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).

    Each variable in the system (poses and landmarks) must be identified with a unique key.
    We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
    Here we will use Symbols

    In GTSAM, measurement functions are represented as 'factors'. Several common factors
    have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
    Here we will use Projection factors to model the camera's landmark observations.
    Also, we will initialize the robot at some location using a Prior factor.

    When the factors are created, we will add them to a Factor Graph. As the factors we are using
    are nonlinear factors, we will need a Nonlinear Factor Graph.

    Finally, once all of the factors have been added to our factor graph, we will want to
    solve/optimize to graph to find the best (Maximum A Posteriori) set of variable values.
    GTSAM includes several nonlinear optimizers to perform this step. Here we will use a
    trust-region method known as Powell's Degleg

    The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
    nonlinear functions around an initial linearization point, then solve the linear system
    to update the linearization point. This happens repeatedly until the solver converges
    to a consistent set of variable values. This requires us to specify an initial guess
    for each variable, held in a Values container.
    """

    # Define the camera calibration parameters
    K = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    # Define the camera observation noise model
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

    # Create the set of ground-truth landmarks
    points = SFMdata.createPoints()

    # Create the set of ground-truth poses
    poses = SFMdata.createPoses(K)

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Add a prior on pose x1. This indirectly specifies where the origin is.
    # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    factor = PriorFactorPose3(X(0), poses[0], pose_noise)
    graph.push_back(factor)
    #   // Because the structure-from-motion problem has a scale ambiguity, the problem is
    #   // still under-constrained. Here we add a prior on the second pose x1, so this will
    #   // fix the scale by indicating the distance between x0 and x1.
    #   // Because these two are fixed, the rest of the poses will be also be fixed.
    factor = PriorFactorPose3(X(1), poses[1], pose_noise)
    graph.push_back(factor)

    for i, point in enumerate(points):
        factor = SmartProjectionPose3Factor(measurement_noise, K)
        for j, pose in enumerate(poses):
            camera = PinholeCameraCal3_S2(pose, K)
            measurement = camera.project(point)
            factor.add(measurement, poseKey_i=X(j), noise_i=measurement_noise, K_i=K)
        graph.push_back(factor)

    # Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
    # Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
    # between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
    #  point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    #  factor = PriorFactorPoint3(L(0), points[0], point_noise)
    #  graph.push_back(factor)
    graph.print('Factor Graph:\n')

    # Create the data structure to hold the initial estimate to the solution
    # Intentionally initialize the variables off from the ground truth
    initial_estimate = Values()
    for i, pose in enumerate(poses):
        transformed_pose = pose.retract(0.1 * np.random.randn(6, 1))
        initial_estimate.insert(X(i), transformed_pose)
    #  for j, point in enumerate(points):
    #      transformed_point = point + 0.1 * np.random.randn(3)
    #      initial_estimate.insert(L(j), transformed_point)
    initial_estimate.print('Initial Estimates:\n')

    # Optimize the graph and print results
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity('TERMINATION')
    optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    print('Optimizing:')
    result = optimizer.optimize()
    result.print('Final results:\n')
    print('initial error = {}'.format(graph.error(initial_estimate)))
    print('final error = {}'.format(graph.error(result)))

    marginals = Marginals(graph, result)
    plot.plot_3d_points(1, result, marginals=marginals)
    plot.plot_trajectory(1, result, marginals=marginals, scale=8)
    plot.set_axes_equal(1)
    plt.show()


def benchmark_gtsam_bundle_adjustment(dataset: Dataset, args=None):
    noise_dataset = Dataset.with_noise(dataset)
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
    #  ttest()
    path = "/home/morkru/Downloads/reichstag/dense/sparse/"
    image_path = "/home/morkru/Downloads/reichstag/dense/images"
    ds = load_colmap_dataset(path, image_path, binary=True)
    benchmark_gtsam_bundle_adjustment(ds)
    #  benchmark_gtsam_absolute_pose(ds)
