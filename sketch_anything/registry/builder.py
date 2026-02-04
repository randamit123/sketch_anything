"""Object Registry Builder for LIBERO environments.

Constructs a per-view registry of detected objects with bounding boxes by
extracting 3D poses from MuJoCo and projecting to 2D via camera matrices.

Object name resolution uses a three-tier strategy:
    1. **LLM resolver** (primary) -- sends the task instruction and the full
       list of MuJoCo body names to a lightweight LLM and asks it to return
       the matching bodies as JSON.  This handles novel objects without any
       hard-coded mapping.
    2. **Static mapping** (fallback) -- a hand-curated dict covering common
       LIBERO objects.  Used when no LLM is available or when the LLM call
       fails.
    3. **Substring / dynamic matching** -- last resort fuzzy matching against
       available body names.
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Dict, List, Optional, Set

import numpy as np

from sketch_anything.libero_utils.camera import (
    compute_2d_bbox,
    get_camera_matrices,
)
from sketch_anything.libero_utils.env import get_object_bbox_3d, list_available_bodies
from sketch_anything.registry.extractor import extract_object_names

logger = logging.getLogger(__name__)

# Type alias for a single view's object registry.
ViewObjectRegistry = Dict[str, dict]

# ---------------------------------------------------------------------------
# Static name mapping for common LIBERO objects.
# Maps natural language name -> list of possible MuJoCo body names to try.
# ---------------------------------------------------------------------------
LIBERO_OBJECT_MAPPING: Dict[str, List[str]] = {
    # Containers
    "bowl": ["akita_black_bowl_1", "bowl", "akita_black_bowl"],
    "blue bowl": ["akita_black_bowl_1", "bowl_blue"],
    "plate": ["plate_1", "plate"],
    "basket": ["basket"],
    "tray": ["tray"],
    # Blocks
    "red block": ["red_block", "cube_red"],
    "blue block": ["blue_block", "cube_blue"],
    "green block": ["green_block", "cube_green"],
    "cube": ["cube", "wooden_cube"],
    # Kitchen items
    "mug": ["mug", "coffee_mug"],
    "cup": ["cup"],
    "butter": ["butter"],
    "cream cheese": ["cream_cheese_1", "cream_cheese"],
    "milk": ["milk"],
    "orange juice": ["orange_juice"],
    "wine bottle": ["wine_bottle_1", "wine_bottle"],
    # Furniture
    "rack": ["wine_rack_1", "wine_rack"],
    "wine rack": ["wine_rack_1", "wine_rack"],
    "drawer": ["drawer", "top_drawer", "middle_drawer", "bottom_drawer"],
    "top drawer": ["top_drawer"],
    "middle drawer": ["middle_drawer"],
    "bottom drawer": ["bottom_drawer"],
    "cabinet": ["wooden_cabinet_1", "cabinet_door", "cabinet"],
    "door": ["door", "microwave_door"],
    # Appliances
    # NOTE: "stove" mapping depends on the task verb — see
    # TASK_VERB_BODY_OVERRIDES below. The default mapping here uses the MAIN
    # surface body (for "put on" tasks). "turn on" tasks override to the knob
    # via TASK_EXTRA_OBJECTS.
    "microwave": ["microwave"],
    "stove": ["flat_stove_1", "flat_stove_1_main", "stove", "flat_stove"],
    "stove knob": ["flat_stove_1_button", "flat_stove_1_knob", "stove_knob", "knob", "button"],
    # Regions / landmarks
    "front of stove": ["flat_stove_1", "flat_stove_1_main"],
    "front_of_stove": ["flat_stove_1", "flat_stove_1_main"],
}

# Extra objects to include based on task verb + object combinations.
# Format: (verb_substring, object_name) -> list of extra natural names to add.
# This ensures "turn on the stove" also registers the stove knob, etc.
TASK_EXTRA_OBJECTS: Dict[tuple, List[str]] = {
    ("turn on", "stove"): ["stove knob"],
    ("turn off", "stove"): ["stove knob"],
    ("open", "microwave"): ["microwave door"],
    ("close", "microwave"): ["microwave door"],
    ("open", "cabinet"): ["cabinet door"],
    ("close", "cabinet"): ["cabinet door"],
}

# Task-verb-aware body overrides for the static resolver.
# When the task verb matches, the object's candidate list is replaced with
# these bodies INSTEAD of the default LIBERO_OBJECT_MAPPING entry.
# This ensures "put on stove" → stove surface while "turn on stove" → knob.
TASK_VERB_BODY_OVERRIDES: Dict[tuple, List[str]] = {
    # "turn on the stove" → the knob is the primary interaction target
    ("turn on", "stove"): ["flat_stove_1_button", "flat_stove_1_knob", "stove_knob"],
    ("turn off", "stove"): ["flat_stove_1_button", "flat_stove_1_knob", "stove_knob"],
}

# Body names to try for the robot gripper.
# Includes actual body names and grip site names from various
# LIBERO / robosuite versions.
GRIPPER_BODY_NAMES = [
    # Actual body names (common across robosuite versions)
    "gripper0_right_finger",
    "gripper0_left_finger",
    "gripper0_eef",
    "robot0_right_hand",
    "robot0_link7",
    "right_hand",
    # Grip site names (may also exist as bodies in some setups)
    "robot0_grip_site",
    "gripper0_grip_site",
]

# Substrings that identify gripper-related bodies when static names miss.
_GRIPPER_BODY_PATTERNS = ["grip", "hand", "finger", "eef"]


def _find_gripper_body(available_bodies: Set[str], sim=None) -> Optional[str]:
    """Find the best gripper body in the environment.

    Search order:
        1. Static ``GRIPPER_BODY_NAMES`` list.
        2. Substring match with ``_GRIPPER_BODY_PATTERNS``.
        3. MuJoCo site lookup -- find a grip *site* and walk up to its
           parent body (sites are not bodies, but the parent always is).

    Args:
        available_bodies: Set of body name strings from the model.
        sim: MuJoCo sim object (optional, enables site-based lookup).

    Returns:
        A valid MuJoCo body name, or ``None``.
    """
    # 1. Static list
    for name in GRIPPER_BODY_NAMES:
        if name in available_bodies:
            logger.info(f"Gripper resolved via static list: '{name}'")
            return name

    # 2. Pattern matching against available bodies
    for body in sorted(available_bodies):
        body_lower = body.lower()
        for pattern in _GRIPPER_BODY_PATTERNS:
            if pattern in body_lower:
                logger.info(f"Gripper resolved via pattern '{pattern}': '{body}'")
                return body

    # 3. Site-based lookup: find site, then get its parent body
    if sim is not None:
        site_candidates = ["robot0_grip_site", "gripper0_grip_site", "grip_site"]
        for site_name in site_candidates:
            try:
                site_id = sim.model.site_name2id(site_name)
                parent_body_id = sim.model.site_bodyid[site_id]
                parent_body_name = sim.model.body_id2name(parent_body_id)
                if parent_body_name:
                    logger.info(
                        f"Gripper resolved via site '{site_name}' "
                        f"-> parent body '{parent_body_name}'"
                    )
                    return parent_body_name
            except (KeyError, ValueError):
                continue

    return None


# ---------------------------------------------------------------------------
# LLM-based dynamic object resolver
# ---------------------------------------------------------------------------

_LLM_RESOLVE_PROMPT = """\
You are helping a robot manipulation system map natural language object names \
to MuJoCo simulator body names.

Task instruction: "{task_instruction}"

The objects mentioned in the task are: {object_names}

The simulator has the following body names available:
{body_list}

IMPORTANT: The task verb determines WHICH body to pick for each object:

- "put on", "place on", "put in" → the object is a DESTINATION SURFACE. \
Pick the LARGEST / MAIN body (e.g. "flat_stove_1_main" for the stove cooking \
surface, NOT "flat_stove_1_button" which is a tiny knob). The robot needs to \
place something ON TOP of this object, so it must be the surface/container body.

- "turn on", "turn off", "twist", "rotate" → the object is being MANIPULATED. \
Pick the SPECIFIC INTERACTIVE PART (e.g. a knob, button, handle, or switch). \
Also include the main body as a separate entry.

- "open", "close", "pull" → pick the MOVABLE PART (e.g. a door, drawer handle).

- "pick up", "push", "slide" → pick the MAIN body of the object being moved.

For EACH object name, pick the single best matching body name from the list \
above based on these rules.

Return ONLY a JSON object mapping each object name to its best matching body \
name. Example for "put the bowl on the stove":

{{"bowl": "akita_black_bowl_1", "stove": "flat_stove_1_main"}}

Example for "turn on the stove":

{{"stove": "flat_stove_1_main", "stove knob": "flat_stove_1_button"}}

Include ONLY object names that appear in the task or are relevant sub-parts. \
Do NOT include robot/gripper bodies. Prefer body names containing "main", \
"base", or "body" for destination surfaces. Return valid JSON only, no other text.
"""


def _resolve_via_llm(
    task_instruction: str,
    extracted_names: List[str],
    available_bodies: Set[str],
    model_path: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Use a lightweight LLM to match object names to MuJoCo bodies.

    Tries to import and call the same Qwen model that the pipeline already
    loads.  If the model is not available (e.g. no GPU), returns ``None``
    so the caller can fall back to the static resolver.

    Args:
        task_instruction: Task description string.
        extracted_names: Object names extracted from the instruction.
        available_bodies: Set of MuJoCo body names in the model.

    Returns:
        Dict mapping natural_name -> body_name, or None on failure.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.info("LLM resolver: transformers not available, skipping")
        return None

    # Only attempt if CUDA is available (lightweight CPU inference is too slow
    # for a 1-3B model to be worth the wait).
    if not torch.cuda.is_available():
        logger.info("LLM resolver: no CUDA device, skipping")
        return None

    model_name = model_path if model_path else "Qwen/Qwen2.5-1.5B-Instruct"
    logger.info(f"LLM resolver: loading {model_name} for object matching...")

    try:
        # Try fast tokenizer first, fall back to slow if the tokenizer
        # class isn't available in this transformers version.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except (ValueError, KeyError) as tok_err:
            logger.info(f"Fast tokenizer failed ({tok_err}), trying slow tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Filter bodies to only non-robot ones to keep prompt small
        filtered = sorted(
            b for b in available_bodies
            if not b.startswith("robot0_link")
            and b != "world"
            and not b.startswith("base_")
        )
        body_list = "\n".join(f"  - {b}" for b in filtered)

        prompt = _LLM_RESOLVE_PROMPT.format(
            task_instruction=task_instruction,
            object_names=", ".join(extracted_names),
            body_list=body_list,
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1, do_sample=True
            )

        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        logger.info(f"LLM resolver raw response: {response}")

        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", response)
        if not json_match:
            logger.warning("LLM resolver: no JSON found in response")
            return None

        mapping = _json.loads(json_match.group())

        # Validate that all values are actually available bodies
        validated: Dict[str, str] = {}
        for name, body in mapping.items():
            if body in available_bodies:
                validated[name] = body
            else:
                # Try substring match as the LLM might have truncated
                for avail in available_bodies:
                    if body in avail or avail in body:
                        validated[name] = avail
                        break
                else:
                    logger.warning(
                        f"LLM resolver: body '{body}' for '{name}' "
                        f"not found in environment, skipping"
                    )

        logger.info(f"LLM resolver result: {validated}")

        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

        return validated if validated else None

    except Exception as e:
        logger.warning(f"LLM resolver failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main resolution logic
# ---------------------------------------------------------------------------

def resolve_object_names(
    task_instruction: str,
    available_bodies: Set[str],
    objects_dict: Optional[dict] = None,
    fixtures_dict: Optional[dict] = None,
    sim=None,
    use_llm: bool = True,
    llm_model_path: Optional[str] = None,
) -> Dict[str, str]:
    """Map natural language object names to MuJoCo body names.

    Resolution strategy:
        1. Extract object names from the task instruction.
        2. If ``use_llm`` is True and a GPU is available, call a lightweight
           LLM to match names to bodies (primary resolver).
        3. Fall back to static mapping + substring matching for any names
           the LLM did not resolve (or if the LLM is unavailable).
        4. Always add the gripper via ``_find_gripper_body``.

    Args:
        task_instruction: Task description string.
        available_bodies: Set of all MuJoCo body names in the environment.
        objects_dict: env.env.objects_dict (optional, for dynamic fallback).
        fixtures_dict: env.env.fixtures_dict (optional, for dynamic fallback).
        sim: MuJoCo sim object (optional, for site-based gripper lookup).
        use_llm: Whether to attempt LLM-based resolution (default True).
        llm_model_path: Local path to LLM model weights (optional).

    Returns:
        Dict mapping natural_name -> MuJoCo body_name.
    """
    extracted_names = extract_object_names(task_instruction)
    logger.info(f"Extracted object names from instruction: {extracted_names}")

    # Add task-specific extra objects (e.g. "stove knob" for "turn on the stove")
    task_lower = task_instruction.lower()
    extras_added: List[str] = []
    for (verb, obj), extra_names in TASK_EXTRA_OBJECTS.items():
        if verb in task_lower and obj in task_lower:
            for extra in extra_names:
                if extra not in extracted_names:
                    extracted_names.append(extra)
                    extras_added.append(extra)
    if extras_added:
        logger.info(f"  Added task-specific extras: {extras_added}")

    resolved: Dict[str, str] = {}

    # --- Tier 1: LLM resolver (if available) ---
    if use_llm:
        llm_result = _resolve_via_llm(
            task_instruction, extracted_names, available_bodies,
            model_path=llm_model_path,
        )
        if llm_result:
            resolved.update(llm_result)
            logger.info(f"  LLM resolved {len(llm_result)} objects")

    # --- Tier 2 + 3: Static + substring for anything still unresolved ---
    for name in extracted_names:
        name_key = name.replace(" ", "_")
        # Skip if the LLM already resolved this name (or a close variant)
        if name in resolved or name_key in resolved:
            continue
        body = _resolve_single(
            name, available_bodies, objects_dict, fixtures_dict,
            task_instruction=task_instruction,
        )
        if body is not None:
            resolved[name] = body
            logger.info(f"  '{name}' -> '{body}' (static/substring)")
        else:
            logger.warning(f"  Could not resolve '{name}' to any MuJoCo body")

    # Always include gripper
    gripper_body = _find_gripper_body(available_bodies, sim=sim)
    if gripper_body is not None:
        resolved["gripper"] = gripper_body
    else:
        logger.warning(
            "Could not find gripper body in environment. "
            "Tried static names, pattern matching, and site lookup."
        )
        body_sample = sorted(available_bodies)[:30]
        logger.warning(f"  Available bodies (first 30): {body_sample}")

    return resolved


def _resolve_single(
    name: str,
    available_bodies: Set[str],
    objects_dict: Optional[dict],
    fixtures_dict: Optional[dict],
    task_instruction: str = "",
) -> Optional[str]:
    """Resolve a single object name to a MuJoCo body name."""
    name_lower = name.lower().strip()
    task_lower = task_instruction.lower()

    # 0. Check task-verb body overrides first (e.g. "turn on" + "stove" → knob)
    for (verb, obj), candidates in TASK_VERB_BODY_OVERRIDES.items():
        if verb in task_lower and obj == name_lower:
            for candidate in candidates:
                if candidate in available_bodies:
                    logger.info(f"  '{name}' -> '{candidate}' (verb override: '{verb}')")
                    return candidate
                for body in available_bodies:
                    if candidate in body or body in candidate:
                        logger.info(f"  '{name}' -> '{body}' (verb override partial: '{verb}')")
                        return body

    # 1. Try static mapping
    if name_lower in LIBERO_OBJECT_MAPPING:
        for candidate in LIBERO_OBJECT_MAPPING[name_lower]:
            if candidate in available_bodies:
                return candidate
            # Partial match
            for body in available_bodies:
                if candidate in body or body in candidate:
                    return body

    # 2. Try direct match against available bodies
    name_underscore = name_lower.replace(" ", "_")
    if name_underscore in available_bodies:
        return name_underscore

    # 3. Try substring match against available bodies
    for body in available_bodies:
        body_lower = body.lower()
        if name_underscore in body_lower or body_lower in name_underscore:
            return body

    # 4. Dynamic fallback: search objects_dict and fixtures_dict
    for d in [objects_dict, fixtures_dict]:
        if d is None:
            continue
        for obj_key, obj_val in d.items():
            key_lower = obj_key.lower()
            if name_lower in key_lower or key_lower in name_lower:
                # Use root_body if available (robosuite object)
                if hasattr(obj_val, "root_body"):
                    return obj_val.root_body
                return obj_key

    return None


def build_object_registry(
    env,
    task_instruction: str,
    camera_names: List[str],
    image_width: int = 256,
    image_height: int = 256,
    use_llm: bool = True,
    llm_model_path: Optional[str] = None,
) -> Dict[str, ViewObjectRegistry]:
    """Build object registries with bounding boxes for each camera view.

    Args:
        env: LIBERO ControlEnv / OffScreenRenderEnv instance.
        task_instruction: Natural language task description.
        camera_names: List of camera names, e.g. ["agentview", "robot0_eye_in_hand"].
        image_width: Width of rendered images.
        image_height: Height of rendered images.
        use_llm: Whether to use LLM-based object resolution (default True).
        llm_model_path: Local path to LLM model weights (optional).

    Returns:
        Dict mapping camera_name -> ViewObjectRegistry.
        Each ViewObjectRegistry maps object_id -> {id, label, bbox, center}.
    """
    sim = env.sim

    available_bodies = list_available_bodies(sim)
    logger.info(f"Environment has {len(available_bodies)} bodies")

    # Get object/fixture dicts if available (LIBERO ControlEnv wraps env.env)
    objects_dict = getattr(env.env, "objects_dict", None) if hasattr(env, "env") else None
    fixtures_dict = getattr(env.env, "fixtures_dict", None) if hasattr(env, "env") else None

    object_mapping = resolve_object_names(
        task_instruction, available_bodies, objects_dict, fixtures_dict,
        sim=sim, use_llm=use_llm, llm_model_path=llm_model_path,
    )
    logger.info(f"Resolved objects: {object_mapping}")

    registries: Dict[str, ViewObjectRegistry] = {}

    # Camera names where the gripper should be excluded from the registry.
    # The eye-in-hand camera IS mounted on the gripper, so the gripper bbox
    # covers the bottom half of the frame and produces misleading arrows.
    _EXCLUDE_GRIPPER_CAMERAS = {"robot0_eye_in_hand"}

    for camera_name in camera_names:
        K, R, t = get_camera_matrices(sim, camera_name, image_width, image_height)
        view_registry: ViewObjectRegistry = {}

        for natural_name, body_name in object_mapping.items():
            # Skip gripper for eye-in-hand cameras
            if natural_name == "gripper" and camera_name in _EXCLUDE_GRIPPER_CAMERAS:
                logger.info(
                    f"Excluding gripper from '{camera_name}' registry "
                    f"(camera is mounted on gripper)"
                )
                continue

            try:
                corners_3d = get_object_bbox_3d(sim, body_name)
                bbox = compute_2d_bbox(corners_3d, K, R, t, image_width, image_height)
                center = [
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ]
                obj_id = natural_name.replace(" ", "_")

                view_registry[obj_id] = {
                    "id": obj_id,
                    "label": natural_name,
                    "mujoco_body": body_name,
                    "bbox": bbox,
                    "center": center,
                }
            except Exception as e:
                logger.warning(f"Failed to process '{natural_name}' ({body_name}): {e}")

        registries[camera_name] = view_registry

    return registries
