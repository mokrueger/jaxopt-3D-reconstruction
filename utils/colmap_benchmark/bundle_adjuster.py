import collections
import os
import re
import subprocess

from dataclasses import dataclass

from dataset import Dataset
from dataset.colmap_dataset_loader import read_write_model

COLMAP_PATH = "/usr/local/bin/colmap"

BundleAdjustmentReport = collections.namedtuple(  # TODO: maybe just a dataclass
    "BundleAdjustmentReport",
    ["residuals", "parameters", "iterations", "time", "initial_cost", "final_cost", "termination", "elapsed_time"]
)


@dataclass
class BundleAdjustmentOptions:
    max_num_iterations: int = 100
    max_linear_solver_iterations: int = 200
    function_tolerance: float = 0.0
    gradient_tolerance: float = 0.0
    parameter_tolerance: float = 0.0
    #  max_num_consecutive_invalid_steps: int = 10
    #  max_consecutive_nonmonotonic_steps: int = 10
    #  num_threads: int = -1
    refine_focal_length: int = 1
    refine_principal_point: int = 0  # Colmap default == 0
    refine_extra_params: int = 1
    refine_extrinsics: int = 1


def perform_bundle_adjustment(input_path, output_path, bundle_adjustment_options: BundleAdjustmentOptions = None):
    os.makedirs(output_path, exist_ok=True)
    input_path = input_path if input_path else "."
    output_path = output_path if output_path else "."
    if not bundle_adjustment_options:
        bundle_adjustment_options = BundleAdjustmentOptions()
    p = subprocess.run([COLMAP_PATH, "bundle_adjuster",
                        "--input_path", input_path,
                        "--output_path", output_path,
                        "--BundleAdjustment.max_num_iterations", str(bundle_adjustment_options.max_num_iterations),
                        "--BundleAdjustment.max_linear_solver_iterations",
                        str(bundle_adjustment_options.max_linear_solver_iterations),
                        "--BundleAdjustment.function_tolerance", str(bundle_adjustment_options.function_tolerance),
                        "--BundleAdjustment.gradient_tolerance", str(bundle_adjustment_options.gradient_tolerance),
                        "--BundleAdjustment.parameter_tolerance", str(bundle_adjustment_options.parameter_tolerance),
                        "--BundleAdjustment.refine_focal_length", str(bundle_adjustment_options.refine_focal_length),
                        "--BundleAdjustment.refine_principal_point",
                        str(bundle_adjustment_options.refine_principal_point),
                        "--BundleAdjustment.refine_extra_params", str(bundle_adjustment_options.refine_extra_params),
                        "--BundleAdjustment.refine_extrinsics", str(bundle_adjustment_options.refine_extrinsics)
                        ], stdout=subprocess.PIPE)
    std_out = p.stdout.decode("utf-8")
    assert p.returncode == 0
    return std_out


def _process_std_out(std_out):
    return BundleAdjustmentReport(
        residuals=int(re.search(r".*Residuals : (\d+)\n", std_out).group(1)),
        parameters=int(re.search(r".*Parameters : (\d+)\n", std_out).group(1)),
        iterations=int(re.search(r".*Iterations : (\d+)\n", std_out).group(1)),
        time=float(re.search(r".*Time : (\d+\.\d+) \[s]", std_out).group(1)),
        initial_cost=float(re.search(r".*Initial cost : (\d+\.\d+) \[px]", std_out).group(1)),
        final_cost=float(re.search(r".*Final cost : (\d+\.\d+) \[px]", std_out).group(1)),
        termination=re.search(r".*Termination : (Convergence|No convergence)\n", std_out).group(1),
        elapsed_time=float(re.search(r".*Elapsed time: (\d+\.\d+) \[minutes]", std_out).group(1))
    )
