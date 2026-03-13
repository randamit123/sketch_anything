from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Path1Config:
    max_retries: int = 5
    max_action_steps: int = 500      # hard cap; simple pick-place ~100 steps, complex tasks (rack insertion) ~350+
    goal_threshold_m: float = 0.05   # 5 cm geometric success threshold
    use_mock_planner: bool = True     # True = mock, False = o3. Default is True.
    llm_model: str = "o3"            # only used when use_mock_planner=False
    render_scale: int = 2
    camera_names: list = field(
        default_factory=lambda: ["agentview", "frontview", "robot0_eye_in_hand"]
    )
    output_dir: str = "./outputs/path1"
    image_width: int = 256
    image_height: int = 256
    video_fps: int = 20


@dataclass
class ObjectInfo:
    label: str
    centroid_3d: np.ndarray          # shape (3,), world frame
    bbox_3d_corners: np.ndarray      # shape (8, 3), world frame
    orientation: np.ndarray          # shape (3, 3), rotation matrix
    bbox_2d_per_camera: dict         # camera_name → [x_min, y_min, x_max, y_max] in [0,1]


@dataclass
class SceneContext:
    point_cloud: np.ndarray          # shape (N, 3), world frame
    object_registry: dict            # label → ObjectInfo
    camera_images: dict              # camera_name → np.ndarray (H, W, 3) uint8
    camera_matrices: dict            # camera_name → tuple(K, R, t)
    task_instruction: str
    initial_eef_pose: np.ndarray     # shape (7,) from env observation


@dataclass
class Trajectory:
    actions: list                    # list of np.ndarray, each shape (7,)
                                     # [x, y, z, roll, pitch, yaw, gripper]
    sub_goals: list                  # list of str, one label per phase
    grasp_indices: list              # list of int — which action indices are grasps
    metadata: dict                   # planner_type, model, retry_num, etc.


@dataclass
class PlannedSketch:
    primitives_per_camera: dict      # camera_name → SketchPrimitives
    rendered_images: dict            # camera_name → np.ndarray (H, W, 3)


@dataclass
class VerifyResult:
    verified: bool
    sim_success: bool
    geometric_success: bool
    final_eef_pose: np.ndarray       # shape (7,)
    distance_to_goal: float          # metres
    feedback: Optional[str]          # None if verified=True
    planned_sketch: PlannedSketch


@dataclass
class OrchestratorResult:
    success: bool
    attempts: int
    final_trajectory: Optional[Trajectory] = None
    final_verify_result: Optional[VerifyResult] = None
