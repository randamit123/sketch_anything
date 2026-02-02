"""Position resolution: converts Position specs to pixel coordinates.

Handles both absolute and object-relative positions per CLAUDE.md Section 2.4.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sketch_anything.schemas.primitives import (
    AbsolutePosition,
    ObjectRelativePosition,
)


# ---------------------------------------------------------------------------
# Anchor computation
# ---------------------------------------------------------------------------

def compute_anchor(
    bbox: List[float],
    anchor: str,
) -> Tuple[float, float]:
    """Compute the normalized (x, y) for an anchor on a bounding box.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in normalized coordinates.
        anchor: One of the 9 anchor names.

    Returns:
        (x, y) normalized coordinates of the anchor point.
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    anchor_map = {
        "center":       (cx, cy),
        "top":          (cx, y_min),
        "bottom":       (cx, y_max),
        "left":         (x_min, cy),
        "right":        (x_max, cy),
        "top_left":     (x_min, y_min),
        "top_right":    (x_max, y_min),
        "bottom_left":  (x_min, y_max),
        "bottom_right": (x_max, y_max),
    }

    if anchor not in anchor_map:
        raise ValueError(
            f"Unknown anchor '{anchor}'. "
            f"Must be one of: {', '.join(sorted(anchor_map))}"
        )

    return anchor_map[anchor]


# ---------------------------------------------------------------------------
# Position resolution
# ---------------------------------------------------------------------------

def resolve_position(
    position,
    object_registry: Dict[str, dict],
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """Resolve a Position to pixel coordinates.

    Args:
        position: An AbsolutePosition or ObjectRelativePosition.
        object_registry: Mapping of object_id -> {id, label, bbox, center}.
        image_width: Width of the target image in pixels.
        image_height: Height of the target image in pixels.

    Returns:
        (x_pixel, y_pixel) integer pixel coordinates.
    """
    if isinstance(position, AbsolutePosition):
        x_norm, y_norm = position.coords
    elif isinstance(position, ObjectRelativePosition):
        obj = object_registry[position.object_id]
        bbox = obj["bbox"]
        anchor_coords = compute_anchor(bbox, position.anchor)
        offset = position.offset if position.offset else (0.0, 0.0)
        x_norm = anchor_coords[0] + offset[0]
        y_norm = anchor_coords[1] + offset[1]
    else:
        raise TypeError(f"Unknown position type: {type(position)}")

    x_pixel = round(x_norm * image_width)
    y_pixel = round(y_norm * image_height)

    return (x_pixel, y_pixel)
