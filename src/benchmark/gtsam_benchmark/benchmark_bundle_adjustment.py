import os
import time

import gtsam
import numpy as np
from gtsam import (
    GeneralSFMFactorCal3_S2,
    LevenbergMarquardtOptimizer,
    NonlinearFactorGraph,
    PriorFactorPinholeCameraCal3_S2,
    PriorFactorPoint3,
    Values,
    symbol_shorthand,
)

from src.benchmark.benchmark import (
    BundleAdjustmentBenchmark,
    BundleAdjustmentBenchmarkResults,
)
from src.config import DATASETS_PATH
from src.dataset.camera import Camera, CameraIntrinsics
from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import (
    CoordinateSystem,
    TransformationDirection,
)
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import (
    load_colmap_dataset,
    params_to_intrinsics,
)
from src.dataset.point import Point3D

L = symbol_shorthand.L
X = symbol_shorthand.X


class GtsamBundleAdjustmentBenchmark(BundleAdjustmentBenchmark):
    def benchmark(self, *args, **kwargs):
        self.benchmark_args_kwargs = (args, kwargs)
        graph = NonlinearFactorGraph()

        initial_estimate = Values()
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        error = 0
        count = 0
        for index, de in enumerate(self.dataset.datasetEntries, start=0):
            ci = de.camera.camera_intrinsics
            #  calibration = Cal3_S2(ci.focal_x, ci.focal_y, ci.skew_factor, ci.center_x, ci.center_y)
            pose = gtsam.Pose3(
                de.camera.camera_pose.in_coordinate_system(CoordinateSystem.COLMAP)
                .in_direction(TransformationDirection.C2W)
                .transformation_translation_matrix
            )
            #  camera = gtsam.gtsam.PinholeCameraCal3Bundler(pose, calibration)

            calibration = gtsam.gtsam.Cal3_S2(
                ci.focal_x, ci.focal_y, ci.skew_factor, ci.center_x, ci.center_y
            )
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

            for j, p2d_p3d in enumerate(
                de.map2d_3d(self.dataset.points3D_mapped, zipped=True), start=0
            ):
                #  measurement = camera.project(point)
                #  factor = GenericProjectionFactorCal3_S2(
                #      p2d_p3d[0].xy, measurement_noise, X(index), L(p2d_p3d[1].identifier), calibration
                #  )
                error += (
                    np.linalg.norm(p2d_p3d[0].xy - camera.project(p2d_p3d[1].xyz)) ** 2
                )
                count += 1
                factor = GeneralSFMFactorCal3_S2(
                    p2d_p3d[0].xy, measurement_noise, X(index), L(p2d_p3d[1].identifier)
                )
                graph.push_back(factor)

        for j, point in enumerate(self.dataset.points3D):
            if j == 0:
                # Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
                # Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
                # between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
                point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
                factor = PriorFactorPoint3(L(point.identifier), point.xyz, point_noise)
                graph.push_back(factor)
            initial_estimate.insert(L(point.identifier), point.xyz)

        initial_estimate.print("Initial Estimates:\n")

        # Optimize the graph and print results
        try:
            start = time.perf_counter()
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosity("TERMINATION")
            optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            print("Optimizing:")
            result = optimizer.optimize()
            total_time = time.perf_counter() - start
        except RuntimeError as e:
            # TODO: Something here
            raise e

        result.print("Final results:\n")
        print("initial error = {}".format(graph.error(initial_estimate)))
        print("final error = {}".format(graph.error(result)))

        """ interpret result """  # TODO: Parse into dataset to visualize
        corresponding_cameras = {}
        for index, de in enumerate(self.dataset.datasetEntries):
            camera = initial_estimate.atPinholeCameraCal3_S2(X(index))
            camera_intrinsics = camera.calibration()
            corresponding_cameras.update(
                {
                    #  de.identifier:
                    index: Camera(
                        camera_pose=CameraPose.from_string_transformation_translation(
                            string=str(camera.pose().matrix())
                            .replace("[", "")
                            .replace("]", ""),
                            identifier=de.identifier,
                            coordinate_system=CoordinateSystem.COLMAP,
                            direction=TransformationDirection.C2W,
                        ),
                        camera_intrinsics=params_to_intrinsics(
                            fx=camera_intrinsics.fx(),
                            fy=camera_intrinsics.fy(),
                            cx=camera_intrinsics.px(),
                            cy=camera_intrinsics.py(),
                            s=camera_intrinsics.skew(),
                        ),
                        width=de.camera.width,
                        height=de.camera.height,
                    )
                }
            )

        corresponding_points = {}
        for p in self.dataset.points3D:
            new_point = initial_estimate.atPoint3(L(p.identifier))
            corresponding_points.update(
                {
                    p.identifier: Point3D(
                        identifier=p.identifier,
                        x=new_point[0],
                        y=new_point[1],
                        z=new_point[2],
                        metadata={
                            **p.metadata,
                            "note": "has been copied from original",
                        },
                    )
                }
            )

        self._results = BundleAdjustmentBenchmarkResults(
            camera_mapping=corresponding_cameras, point_mapping=corresponding_points
        )
        self._time = total_time
        self._iterations = optimizer.iterations()


if __name__ == "__main__":
    path = os.path.join(DATASETS_PATH, "reichstag/sparse/")
    image_path = os.path.join(DATASETS_PATH, "reichstag/images")
    ds = load_colmap_dataset(path, image_path, binary=True)

    gtsamBundleAdjustmentBenchmark = GtsamBundleAdjustmentBenchmark(ds)
    gtsamBundleAdjustmentBenchmark.benchmark()
