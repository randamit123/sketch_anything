"""Stage 2 — Object Extraction.

Calls build_object_registry from sketch_anything, then enriches entries
into ObjectInfo dataclasses. Applies a task-relevance filter so only
objects mentioned in the task instruction are retained.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from sketch_anything.registry.builder import build_object_registry
from sketch_anything.libero_utils.env import get_object_bbox_3d
from path1_execution.config import ObjectInfo, Path1Config

logger = logging.getLogger(__name__)

# PATH1-LOCAL BODY OVERRIDES
# build_object_registry sometimes resolves a stove knob/button instead of the cooking
# surface.  Map those wrong bodies to the semantically correct placement surface.
# Applied only for PLACEMENT tasks ("put X on/in stove") — NOT for interaction tasks
# ("turn on stove") where the knob/button body IS the correct target.
# This does NOT touch sketch_anything/ — it is a path1-only post-processing step.
PLACEMENT_BODY_OVERRIDES: dict[str, str] = {
    "flat_stove_1_button": "flat_stove_1_burner",
    "flat_stove_1_main":   "flat_stove_1_burner",
    "flat_stove_1_base":   "flat_stove_1_burner",
}

# Task verbs that indicate INTERACTION (not placement): skip PLACEMENT_BODY_OVERRIDES for these.
INTERACTION_TASK_VERBS = ("turn on", "turn off", "press", "rotate", "open", "close", "push")

# MISSING OBJECT INJECTIONS
# build_object_registry may not identify certain objects (e.g. cabinets, surfaces) even
# when they are mentioned in the task.  These fallbacks inject those objects directly from
# the MuJoCo simulation when the label substring appears in the task but is missing from
# the registry.  Format: (task_keyword, label_to_inject, mujoco_body).
MISSING_OBJECT_INJECTIONS: list[tuple[str, str, str]] = [
    ("cabinet", "cabinet", "wooden_cabinet_1_cabinet_top"),
    ("rack",    "rack",    "wine_rack_1_main"),
]


def extract_objects(env, sim, task_instruction: str, config: Path1Config) -> Dict[str, ObjectInfo]:
    """Build an enriched object registry from MuJoCo ground-truth state.

    Calls build_object_registry (which projects 3D poses → 2D bboxes for
    each camera), then enriches each entry with 3D centroid, bbox corners,
    and body orientation matrix.

    A task-relevance filter retains only objects whose label appears as a
    case-insensitive substring of task_instruction.

    Args:
        env: LIBERO OffScreenRenderEnv instance.
        sim: MuJoCo simulation object (env.sim).
        task_instruction: Natural language task description.
        config: Path1Config with camera_names, image_width, image_height.

    Returns:
        Dict mapping label -> ObjectInfo.
    """
    # per_camera_registry: camera_name -> {object_id: {id, label, mujoco_body, bbox, center}}
    per_camera_registry = build_object_registry(
        env,
        task_instruction,
        config.camera_names,
        image_width=config.image_width,
        image_height=config.image_height,
        use_llm=False,
    )

    # Collect all unique objects across cameras.
    # Key: object_id, Value: dict with label, mujoco_body, bbox per camera
    objects_collected: Dict[str, dict] = {}
    for cam_name, view_registry in per_camera_registry.items():
        for obj_id, entry in view_registry.items():
            if obj_id not in objects_collected:
                objects_collected[obj_id] = {
                    "label": entry["label"],
                    "mujoco_body": entry["mujoco_body"],
                    "bbox_2d_per_camera": {},
                }
            objects_collected[obj_id]["bbox_2d_per_camera"][cam_name] = entry["bbox"]

    task_lower = task_instruction.lower()
    result: Dict[str, ObjectInfo] = {}

    for obj_id, info in objects_collected.items():
        label = info["label"]
        mujoco_body = info["mujoco_body"]

        # Task-relevance filter
        if label.lower() not in task_lower:
            logger.debug(f"Filtering out '{label}' — not mentioned in task instruction")
            continue

        # Apply path1-local body overrides before 3D extraction.
        # Skip for interaction tasks (turn on, push, open) where the original body is correct.
        is_interaction_task = any(v in task_lower for v in INTERACTION_TASK_VERBS)
        if not is_interaction_task and mujoco_body in PLACEMENT_BODY_OVERRIDES:
            overridden = PLACEMENT_BODY_OVERRIDES[mujoco_body]
            if overridden in sim.model.body_names:
                logger.info(
                    f"Body override: '{label}' ({mujoco_body}) → {overridden}"
                )
                mujoco_body = overridden
            else:
                logger.warning(
                    f"Override body '{overridden}' not found in sim; keeping '{mujoco_body}'"
                )

        # Get 3D bounding box corners for this body
        try:
            corners = get_object_bbox_3d(sim, mujoco_body)  # (8, 3)
        except Exception as e:
            logger.warning(f"Could not get 3D bbox for '{label}' ({mujoco_body}): {e}")
            continue

        centroid_3d = corners.mean(axis=0)  # (3,)

        # Orientation matrix from sim body xmat
        try:
            body_id = sim.model.body_name2id(mujoco_body)
            orientation = sim.data.body_xmat[body_id].reshape(3, 3).copy()
        except Exception as e:
            logger.warning(f"Could not get orientation for '{label}' ({mujoco_body}): {e}")
            orientation = np.eye(3)

        obj_info = ObjectInfo(
            label=label,
            centroid_3d=centroid_3d,
            bbox_3d_corners=corners,
            orientation=orientation,
            bbox_2d_per_camera=info["bbox_2d_per_camera"],
        )
        result[label] = obj_info
        logger.info(
            f"Extracted object '{label}' at centroid {centroid_3d.round(3).tolist()}"
        )

    # Inject objects that are task-relevant but not found by build_object_registry.
    for task_kw, inject_label, inject_body in MISSING_OBJECT_INJECTIONS:
        if task_kw in task_lower and inject_label not in result:
            if inject_body not in sim.model.body_names:
                logger.debug(f"Injection body '{inject_body}' not in sim — skipping '{inject_label}'")
                continue
            try:
                corners = get_object_bbox_3d(sim, inject_body)
                centroid_3d = corners.mean(axis=0)
                body_id = sim.model.body_name2id(inject_body)
                orientation = sim.data.body_xmat[body_id].reshape(3, 3).copy()
                result[inject_label] = ObjectInfo(
                    label=inject_label,
                    centroid_3d=centroid_3d,
                    bbox_3d_corners=corners,
                    orientation=orientation,
                    bbox_2d_per_camera={},
                )
                logger.info(f"Injected missing object '{inject_label}' ({inject_body}) at {centroid_3d.round(3)}")
            except Exception as e:
                logger.warning(f"Could not inject '{inject_label}' ({inject_body}): {e}")

    logger.info(f"Object extraction complete: {len(result)} task-relevant objects found")
    return result
