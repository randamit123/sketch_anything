"""Primitive validation against schema and semantic constraints.

Validates VLM output for correctness before rendering. Produces hard errors
(must fix) and soft warnings (advisory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sketch_anything.schemas.primitives import (
    AbsolutePosition,
    CirclePrimitive,
    GripperPrimitive,
    ObjectRelativePosition,
    SketchPrimitives,
)

VALID_ANCHORS = {
    "center", "top", "bottom", "left", "right",
    "top_left", "top_right", "bottom_left", "bottom_right",
}

VALID_PURPOSES = {
    "grasp_point", "release_point", "contact",
    "rotation_pivot", "target_location",
}

VALID_ACTIONS = {"open", "close"}


@dataclass
class ValidationResult:
    """Result of primitive validation."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_primitives(
    primitives: SketchPrimitives,
    object_registry: Dict[str, dict],
) -> ValidationResult:
    """Validate primitives against the object registry and schema constraints.

    Args:
        primitives: Parsed sketch primitives from VLM output.
        object_registry: Mapping of object_id -> {id, label, bbox, center}
            for a single camera view.

    Returns:
        ValidationResult with errors (hard failures) and warnings (advisory).
    """
    result = ValidationResult()
    valid_ids = set(object_registry.keys())

    steps_seen: set[int] = set()

    for i, prim in enumerate(primitives.primitives):
        prefix = f"primitives[{i}]"
        steps_seen.add(prim.step)

        # Validate step number (Pydantic enforces >= 1, but double-check)
        if prim.step < 1:
            result.errors.append(
                f"Step number {prim.step} must be >= 1"
            )

        # Type-specific validation
        if isinstance(prim, CirclePrimitive):
            _validate_position(prim.center, valid_ids, f"{prefix}.center", result)
            if not (0.01 <= prim.radius <= 0.15):
                result.errors.append(
                    f"Circle radius {prim.radius} must be in range [0.01, 0.15]"
                )
            if prim.purpose not in VALID_PURPOSES:
                result.errors.append(
                    f"Purpose '{prim.purpose}' is invalid. "
                    f"Must be one of: {', '.join(sorted(VALID_PURPOSES))}"
                )

        elif isinstance(prim, GripperPrimitive):
            _validate_position(prim.position, valid_ids, f"{prefix}.position", result)
            if prim.action not in VALID_ACTIONS:
                result.errors.append(
                    f"Gripper action '{prim.action}' must be 'open' or 'close'"
                )

        else:
            # ArrowPrimitive
            _validate_position(prim.start, valid_ids, f"{prefix}.start", result)
            _validate_position(prim.end, valid_ids, f"{prefix}.end", result)
            for j, wp in enumerate(prim.waypoints):
                _validate_position(wp, valid_ids, f"{prefix}.waypoints[{j}]", result)

    # ---- Warnings ----

    # Step sequence gaps
    if steps_seen:
        expected = set(range(1, max(steps_seen) + 1))
        missing = expected - steps_seen
        if missing:
            result.warnings.append(
                f"Step sequence has gaps: missing steps {sorted(missing)}"
            )

    # Missing grasp before transport
    _check_grasp_before_transport(primitives, result)

    # Check that the VLM actually uses the object registry
    _check_registry_usage(primitives, valid_ids, result)

    # Finalize
    if result.errors:
        result.is_valid = False

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_position(
    position,
    valid_ids: set[str],
    location: str,
    result: ValidationResult,
) -> None:
    """Validate a single Position value."""
    if isinstance(position, AbsolutePosition):
        x, y = position.coords
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            result.errors.append(
                f"Absolute coordinate ({x}, {y}) at {location} "
                f"is outside valid range [0, 1]"
            )
    elif isinstance(position, ObjectRelativePosition):
        if position.object_id not in valid_ids:
            result.errors.append(
                f"Unknown object_id '{position.object_id}'. "
                f"Valid IDs are: {', '.join(sorted(valid_ids))}"
            )
        if position.anchor not in VALID_ANCHORS:
            result.errors.append(
                f"Anchor '{position.anchor}' is invalid. "
                f"Must be one of: {', '.join(sorted(VALID_ANCHORS))}"
            )
        # Warning: large offset
        dx, dy = position.offset
        if abs(dx) > 0.3 or abs(dy) > 0.3:
            result.warnings.append(
                f"Large offset ({dx}, {dy}) at {location} exceeds 0.3"
            )


def _check_registry_usage(
    primitives: SketchPrimitives,
    valid_ids: set[str],
    result: ValidationResult,
) -> None:
    """Warn if no primitives reference objects from the registry.

    The VLM is instructed to prefer object-relative positions when
    referencing detected objects.  If it outputs *only* absolute coordinates
    it likely ignored the registry, which usually means bad spatial output.
    """
    if not primitives.primitives or not valid_ids:
        return

    # Collect all positions from all primitives
    positions = _collect_all_positions(primitives)

    obj_relative_count = sum(
        1 for p in positions if isinstance(p, ObjectRelativePosition)
    )
    referenced_ids = {
        p.object_id for p in positions if isinstance(p, ObjectRelativePosition)
    }

    # Non-gripper objects we expect to see referenced
    non_gripper_ids = valid_ids - {"gripper"}

    if obj_relative_count == 0 and non_gripper_ids:
        result.warnings.append(
            "No primitives use object-relative positions. "
            "The VLM may have ignored the object registry. "
            f"Available objects: {', '.join(sorted(non_gripper_ids))}"
        )
    elif non_gripper_ids and not (referenced_ids & non_gripper_ids):
        result.warnings.append(
            "No non-gripper objects are referenced in any primitive. "
            f"Expected references to: {', '.join(sorted(non_gripper_ids))}"
        )


def _collect_all_positions(primitives: SketchPrimitives) -> list:
    """Extract every Position from all primitives."""
    from sketch_anything.schemas.primitives import ArrowPrimitive

    positions = []
    for prim in primitives.primitives:
        if isinstance(prim, CirclePrimitive):
            positions.append(prim.center)
        elif isinstance(prim, GripperPrimitive):
            positions.append(prim.position)
        elif isinstance(prim, ArrowPrimitive):
            positions.append(prim.start)
            positions.append(prim.end)
            positions.extend(prim.waypoints)
    return positions


def _check_grasp_before_transport(
    primitives: SketchPrimitives,
    result: ValidationResult,
) -> None:
    """Warn if an arrow follows no grasp_point circle at an earlier step."""
    grasp_steps: set[int] = set()
    for prim in primitives.primitives:
        if isinstance(prim, CirclePrimitive) and prim.purpose == "grasp_point":
            grasp_steps.add(prim.step)

    for prim in primitives.primitives:
        # An arrow at step > 1 that isn't preceded by any grasp circle
        if (
            hasattr(prim, "start")
            and prim.step > 1
            and not any(gs < prim.step for gs in grasp_steps)
        ):
            result.warnings.append(
                f"Arrow at step {prim.step} has no preceding grasp_point circle"
            )
            break  # Only warn once
