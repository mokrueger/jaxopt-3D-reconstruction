import collections
import os
import re
import time

import numpy as np
import pycolmap
from pycolmap import AbsolutePoseRefinementOptions

from src.benchmark.benchmark import SinglePoseBenchmark, SinglePoseBenchmarkResults
from src.benchmark.colmap_benchmark.utils import OutputGrabber
from src.config import DATASETS_PATH
from src.dataset.camera import Camera
from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import (
    CoordinateSystem,
    TransformationDirection,
)
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset

PoseRefinementReport = collections.namedtuple(  # TODO: maybe just a dataclass
    "PoseRefinementReport",
    [
        "residuals",
        "parameters",
        "iterations",
        "time",
        "initial_cost",
        "final_cost",
        "termination",
    ],
)


def _process_std_out(std_out):
    return PoseRefinementReport(
        residuals=int(re.search(r".*Residuals : (\d+)\n", std_out).group(1)),
        parameters=int(re.search(r".*Parameters : (\d+)\n", std_out).group(1)),
        iterations=int(re.search(r".*Iterations : (\d+)\n", std_out).group(1)),
        time=float(re.search(r".*Time : (\d+\.\d+) \[s]", std_out).group(1)),
        initial_cost=float(
            re.search(r".*Initial cost : (\d+\.\d+) \[px]", std_out).group(1)
        ),
        final_cost=float(
            re.search(r".*Final cost : (\d+\.\d+) \[px]", std_out).group(1)
        ),
        termination=re.search(
            r".*Termination : (Convergence|No convergence)\n", std_out
        ).group(1),
    )


class ColmapSinglePoseBenchmark(SinglePoseBenchmark):
    FRAMEWORK = "Colmap"

    def _prepare_dataset(self):
        mapping = {}
        for index, e in enumerate(self.dataset.datasetEntries):
            mapping.update(
                {index: e.map2d_3d(self.dataset.points3D_mapped, zipped=False, np=True)}
            )
        return mapping

    def _prepare_cameras(self):
        mapping = {}
        for index, e in enumerate(self.dataset.datasetEntries):
            mapping.update({index: e.camera})
        return mapping

    def _prepare_colmap_cameras(self, mapping_cameras):
        mapping = {}
        for image_id, c in list(mapping_cameras.items()):
            # Note: by default colmap uses focal length of 1.2*max(width, height) to start
            colmap_camera = pycolmap.Camera(
                model="PINHOLE",
                width=c.width,
                height=c.height,
                # TODO: could be radial with s
                params=[
                    c.camera_intrinsics.focal_x,
                    c.camera_intrinsics.focal_y,
                    c.camera_intrinsics.center_x,
                    c.camera_intrinsics.center_y,
                ],
            )
            mapping[image_id] = colmap_camera
        return mapping

    def benchmark_absolute_pose(
        self,
        tvecs,
        qvecs,
        p2d_list,
        p3d_list,
        inlier_mask_list,
        camera_list,
        absolute_pose_refinement_options,
        verbose,
    ):
        assert len(p2d_list) == len(p3d_list) == len(camera_list)
        outputs = []
        reports = []
        for index in range(len(p2d_list)):
            output_grabber = OutputGrabber()
            output_grabber.start()
            o = pycolmap.pose_refinement(
                tvec=tvecs[index],
                qvec=qvecs[index],
                points2D=p2d_list[index],
                points3D=p3d_list[index],
                inlier_mask=inlier_mask_list[index],
                camera=camera_list[index],
                refinement_options=absolute_pose_refinement_options,
            )
            output_grabber.stop()
            reports.append(_process_std_out(output_grabber.capturedtext))
            if verbose:
                print(output_grabber.capturedtext)
            outputs.append(o)
        return outputs, [r.time for r in reports], reports

    def validate_output(
        self,
        output,
        camera_poses_list,
        validation_error_position,
        validation_error_rotation,
    ):
        output_camera_poses = list(
            map(  # TODO: Remove this function
                lambda x: CameraPose.from_string_wxyz_quaternion_translation(
                    string=f"{x.get('qvec')[0]} "
                    f"{x.get('qvec')[1]} "
                    f"{x.get('qvec')[2]} "
                    f"{x.get('qvec')[3]} "
                    f"{x.get('tvec')[0]} "
                    f"{x.get('tvec')[1]} "
                    f"{x.get('tvec')[2]}",
                    coordinate_system=CoordinateSystem.COLMAP,
                    direction=TransformationDirection.W2C,
                ),
                output,
            )
        )

        expected_result = list(zip(camera_poses_list, output_camera_poses))
        position_errors = np.array(
            list(
                map(
                    lambda cp1_cp2: CameraPose.compute_position_error(
                        cp1_cp2[0], cp1_cp2[1]
                    ),
                    expected_result,
                )
            )
        )
        rotation_errors = np.array(
            list(
                map(
                    lambda cp1_cp2: CameraPose.compute_rotation_error(
                        cp1_cp2[0], cp1_cp2[1]
                    ),
                    expected_result,
                )
            )
        )

        assert all(map(lambda o: o["success"], output))
        #  assert np.max(position_errors) <= validation_error_position
        #  assert np.max(rotation_errors) <= validation_error_rotation

    @staticmethod
    def _parse_colmap_output(output):
        return list(
            map(
                lambda x: CameraPose.from_string_wxyz_quaternion_translation(
                    string=f"{x.get('qvec')[0]} "
                    f"{x.get('qvec')[1]} "
                    f"{x.get('qvec')[2]} "
                    f"{x.get('qvec')[3]} "
                    f"{x.get('tvec')[0]} "
                    f"{x.get('tvec')[1]} "
                    f"{x.get('tvec')[2]}",
                    coordinate_system=CoordinateSystem.COLMAP,
                    direction=TransformationDirection.W2C,
                ),
                output,
            )
        )

    def benchmark(self, *args, **kwargs):
        """
        @type verbose: bool; specify verbosity
        """
        verbose = kwargs.get("verbose", False)
        self.benchmark_args_kwargs = (args, kwargs)
        # TODO: Different ids could be a problem, perhaps switch to datasetEntry.identifier-based mapping
        self.benchmark_args_kwargs = (args, kwargs)
        mapping_2d_3d_by_id = self._prepare_dataset()
        mapping_cameras_by_id = self._prepare_cameras()
        mapping_colmap_cameras_by_id = self._prepare_colmap_cameras(
            mapping_cameras_by_id
        )

        absolute_pose_refinement_options = AbsolutePoseRefinementOptions()
        absolute_pose_refinement_options.refine_extra_params = True
        absolute_pose_refinement_options.refine_focal_length = True
        absolute_pose_refinement_options.print_summary = True  # Set to false for now

        """input preparation"""
        (
            tvecs,
            qvecs,
            p2d_list,
            p3d_list,
            inlier_mask_list,
            colmap_camera_list,
            camera_poses_list,
        ) = ([], [], [], [], [], [], [])
        # Note: sorting just for peace of mind
        for index, v in sorted(list(mapping_2d_3d_by_id.items()), key=lambda x: x[0]):
            p2d_list.append(v[0])
            p3d_list.append(v[1])
            colmap_camera_list.append(mapping_colmap_cameras_by_id.get(index))
            camera_poses_list.append(mapping_cameras_by_id.get(index).camera_pose)
            tvecs.append(camera_poses_list[index].translation)
            qvecs.append(camera_poses_list[index].wxyz_quaternion)
            inlier_mask_list.append(
                [True for _ in v[0]]
            )  # Set all points to be inliers since it's close enough to GT
            # inlier_mask_list.append(mapping_cameras_by_id.get(index).compute_inlier_mask_mod(
            #     v[0],
            #     v[1],
            #     #  RANSACOptions().max_error
            # ))

        """benchmark"""
        elapsed_time = 0.0
        output, times, reports = self.benchmark_absolute_pose(
            tvecs,
            qvecs,
            p2d_list,
            p3d_list,
            inlier_mask_list,
            colmap_camera_list,
            absolute_pose_refinement_options,
            verbose=verbose,
        )
        elapsed_time += sum(times)
        iterations = [r.iterations for r in reports]

        """parsing output"""
        parsed_camera_pose = self._parse_colmap_output(output)
        # TODO: BIG NOTE: Since there is a bug(?) with the camera intrinsics we use the same as before(!!!!!)
        camera_mapping = {
            index: Camera(
                camera_pose=camera_pose,
                camera_intrinsics=mapping_cameras_by_id.get(
                    index
                ).camera_intrinsics,  # TODO: FIX THIS(!!)
                width=mapping_cameras_by_id.get(index).width,  # TODO: FIX THIS(!!)
                height=mapping_cameras_by_id.get(index).height,  # TODO: FIX THIS(!!)
            )
            for index, camera_pose in enumerate(parsed_camera_pose)
        }

        # if validate_result:
        #     print("validation run...")
        #     output = self.benchmark_absolute_pose(tvecs, qvecs, p2d_list, p3d_list, inlier_mask_list,
        #                                           colmap_camera_list, absolute_pose_refinement_options)
        #     self.validate_output(output, camera_poses_list, validation_error_position, validation_error_rotation)

        self._results = SinglePoseBenchmarkResults(camera_mapping=camera_mapping)
        self._time = elapsed_time
        self._single_times = times
        self._iterations = iterations


if __name__ == "__main__":
    path = os.path.join(DATASETS_PATH, "reichstag/sparse/TXT")
    image_path = os.path.join(DATASETS_PATH, "reichstag/images")
    ds = load_colmap_dataset(path, image_path, binary=False)

    colmapSinglePoseBenchmark = ColmapSinglePoseBenchmark(ds)
    colmapSinglePoseBenchmark.benchmark()
    time2 = colmapSinglePoseBenchmark.time
    print("finished")
