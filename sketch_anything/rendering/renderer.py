"""Deterministic renderer: draws sketch primitives onto scene images.

Rendering order per step (ascending): Circles -> Arrows -> Grippers.
All input/output images are RGB uint8. OpenCV BGR conversion is internal.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from sketch_anything.rendering.config import (
    RenderConfig,
    get_step_color,
    rgb_to_bgr,
)
from sketch_anything.rendering.resolver import resolve_position
from sketch_anything.schemas.primitives import (
    ArrowPrimitive,
    CirclePrimitive,
    GripperPrimitive,
    SketchPrimitives,
)


def render_primitives(
    image: np.ndarray,
    primitives: SketchPrimitives,
    object_registry: Dict[str, dict],
    config: RenderConfig | None = None,
) -> np.ndarray:
    """Render validated primitives onto a scene image.

    Args:
        image: RGB image array, shape (H, W, 3), dtype uint8.
        primitives: Validated primitive specification.
        object_registry: Object registry for position resolution.
        config: Rendering parameters. Uses defaults if None.

    Returns:
        RGB image array with primitives rendered, same shape as input.
    """
    if config is None:
        config = RenderConfig()

    # Work in BGR for OpenCV
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    line_type = cv2.LINE_AA if config.use_antialiasing else cv2.LINE_8

    # --- Object bounding boxes + labels ---
    if config.draw_object_bboxes and object_registry:
        _draw_object_bboxes(canvas, object_registry, w, h, config, line_type)

    # --- Primitives ---
    all_steps = sorted({p.step for p in primitives.primitives})

    for step in all_steps:
        color_rgb = get_step_color(step)
        color_bgr = rgb_to_bgr(color_rgb)

        step_prims = [p for p in primitives.primitives if p.step == step]

        # Render order: circles, arrows, grippers
        circles = [p for p in step_prims if isinstance(p, CirclePrimitive)]
        arrows = [p for p in step_prims if isinstance(p, ArrowPrimitive)]
        grippers = [p for p in step_prims if isinstance(p, GripperPrimitive)]

        for c in circles:
            _render_circle(canvas, c, object_registry, w, h, color_bgr, config, line_type)
            if config.draw_labels:
                _label_circle(canvas, c, object_registry, w, h, color_bgr, config, line_type)

        for a in arrows:
            _render_arrow(canvas, a, object_registry, w, h, color_bgr, config, line_type)
            if config.draw_labels:
                _label_arrow(canvas, a, object_registry, w, h, color_bgr, config, line_type)

        for g in grippers:
            _render_gripper(canvas, g, object_registry, w, h, color_bgr, config, line_type)
            if config.draw_labels:
                _label_gripper(canvas, g, object_registry, w, h, color_bgr, config, line_type)

    # --- Legend ---
    if config.draw_legend and primitives.primitives:
        _draw_legend(canvas, primitives, config)

    # Convert back to RGB
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Per-primitive renderers
# ---------------------------------------------------------------------------

def _render_arrow(
    canvas: np.ndarray,
    arrow: ArrowPrimitive,
    registry: Dict[str, dict],
    w: int,
    h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Draw an arrow with optional waypoints."""
    start = resolve_position(arrow.start, registry, w, h)
    end = resolve_position(arrow.end, registry, w, h)
    waypoints = [resolve_position(wp, registry, w, h) for wp in arrow.waypoints]

    # Build point sequence: start -> waypoints -> end
    points = [start] + waypoints + [end]

    # Draw line segments
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], color, config.arrow_thickness, line_type)

    # Draw waypoint dots
    for wp_px in waypoints:
        cv2.circle(canvas, wp_px, config.waypoint_radius, color, -1, line_type)

    # Draw arrowhead at end
    if len(points) >= 2:
        _draw_arrowhead(canvas, points[-2], points[-1], color, config, line_type)


def _draw_arrowhead(
    canvas: np.ndarray,
    from_pt: Tuple[int, int],
    to_pt: Tuple[int, int],
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Draw an arrowhead at to_pt pointing away from from_pt."""
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    seg_len = math.sqrt(dx * dx + dy * dy)

    if seg_len < 1e-6:
        return

    tip_len = max(config.arrow_tip_length * seg_len, 8.0)
    angle = math.atan2(dy, dx)
    spread = math.pi / 6  # 30 degrees

    p1 = (
        int(to_pt[0] - tip_len * math.cos(angle - spread)),
        int(to_pt[1] - tip_len * math.sin(angle - spread)),
    )
    p2 = (
        int(to_pt[0] - tip_len * math.cos(angle + spread)),
        int(to_pt[1] - tip_len * math.sin(angle + spread)),
    )

    pts = np.array([to_pt, p1, p2], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color, line_type)


def _render_circle(
    canvas: np.ndarray,
    circle: CirclePrimitive,
    registry: Dict[str, dict],
    w: int,
    h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Draw a stroked circle with center dot."""
    center = resolve_position(circle.center, registry, w, h)
    radius_px = round(circle.radius * max(w, h))

    # Stroked circle
    cv2.circle(canvas, center, radius_px, color, config.circle_stroke_width, line_type)
    # Center dot
    cv2.circle(canvas, center, config.circle_center_radius, color, -1, line_type)


def _render_gripper(
    canvas: np.ndarray,
    gripper: GripperPrimitive,
    registry: Dict[str, dict],
    w: int,
    h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Draw a diamond marker for gripper state change.

    Close = filled diamond, Open = outlined diamond.
    """
    pos = resolve_position(gripper.position, registry, w, h)
    half = config.gripper_marker_size // 2

    # Diamond points (rotated square)
    pts = np.array([
        [pos[0], pos[1] - half],  # top
        [pos[0] + half, pos[1]],  # right
        [pos[0], pos[1] + half],  # bottom
        [pos[0] - half, pos[1]],  # left
    ], dtype=np.int32)

    if gripper.action == "close":
        cv2.fillPoly(canvas, [pts], color, line_type)
    else:  # open
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=line_type)


# ---------------------------------------------------------------------------
# Object bounding boxes
# ---------------------------------------------------------------------------

def _draw_object_bboxes(
    canvas: np.ndarray,
    registry: Dict[str, dict],
    w: int,
    h: int,
    config: RenderConfig,
    line_type: int,
) -> None:
    """Draw bounding boxes and labels for all objects in the registry."""
    bbox_bgr = rgb_to_bgr(config.bbox_color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj_id, obj_data in registry.items():
        bbox = obj_data["bbox"]
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_bgr, config.bbox_thickness, line_type)

        label = obj_data.get("label", obj_id)
        (tw, th), _ = cv2.getTextSize(label, font, config.label_font_scale, config.label_thickness)

        # Background for readability
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), bbox_bgr, -1)
        cv2.putText(
            canvas, label, (x1 + 1, y1 - 3),
            font, config.label_font_scale, (0, 0, 0),
            config.label_thickness, line_type,
        )


# ---------------------------------------------------------------------------
# Primitive labels
# ---------------------------------------------------------------------------

def _put_label(
    canvas: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
    offset: Tuple[int, int] = (6, -6),
) -> None:
    """Draw a small text label near a position with a dark background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = pos[0] + offset[0]
    y = pos[1] + offset[1]

    (tw, th), _ = cv2.getTextSize(text, font, config.label_font_scale, config.label_thickness)

    # Clamp to canvas bounds
    h_img, w_img = canvas.shape[:2]
    x = max(0, min(x, w_img - tw - 2))
    y = max(th + 4, min(y, h_img - 2))

    cv2.rectangle(canvas, (x - 1, y - th - 2), (x + tw + 1, y + 2), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, config.label_font_scale, color, config.label_thickness, line_type)


def _label_circle(
    canvas: np.ndarray,
    circle: CirclePrimitive,
    registry: Dict[str, dict],
    w: int, h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Label a circle with step number and purpose."""
    center = resolve_position(circle.center, registry, w, h)
    text = f"S{circle.step}:{circle.purpose}"
    _put_label(canvas, text, center, color, config, line_type)


def _label_arrow(
    canvas: np.ndarray,
    arrow: ArrowPrimitive,
    registry: Dict[str, dict],
    w: int, h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Label an arrow with step number at its midpoint."""
    start = resolve_position(arrow.start, registry, w, h)
    end = resolve_position(arrow.end, registry, w, h)
    mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    text = f"S{arrow.step}:move"
    _put_label(canvas, text, mid, color, config, line_type, offset=(4, -10))


def _label_gripper(
    canvas: np.ndarray,
    gripper: GripperPrimitive,
    registry: Dict[str, dict],
    w: int, h: int,
    color: Tuple[int, int, int],
    config: RenderConfig,
    line_type: int,
) -> None:
    """Label a gripper with step number and action."""
    pos = resolve_position(gripper.position, registry, w, h)
    text = f"S{gripper.step}:{gripper.action}"
    _put_label(canvas, text, pos, color, config, line_type, offset=(10, 4))


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _draw_legend(
    canvas: np.ndarray,
    primitives: SketchPrimitives,
    config: RenderConfig,
) -> None:
    """Draw a step-color legend in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h_img, w_img = canvas.shape[:2]

    all_steps = sorted({p.step for p in primitives.primitives})
    if not all_steps:
        return

    # Build legend entries: step number -> list of primitive types
    step_info: Dict[int, List[str]] = {}
    for p in primitives.primitives:
        step_info.setdefault(p.step, [])
        if isinstance(p, ArrowPrimitive):
            step_info[p.step].append("arrow")
        elif isinstance(p, CirclePrimitive):
            step_info[p.step].append(p.purpose)
        elif isinstance(p, GripperPrimitive):
            step_info[p.step].append(f"grip:{p.action}")

    line_h = 14
    padding = 4
    legend_h = len(all_steps) * line_h + padding * 2 + line_h  # +header
    legend_w = 0

    # Measure widths
    entries = []
    for step in all_steps:
        color_rgb = get_step_color(step)
        color_bgr = rgb_to_bgr(color_rgb)
        desc = ", ".join(sorted(set(step_info.get(step, []))))
        text = f"Step {step}: {desc}"
        (tw, _), _ = cv2.getTextSize(text, font, config.label_font_scale, config.label_thickness)
        legend_w = max(legend_w, tw)
        entries.append((step, text, color_bgr))

    legend_w += padding * 2 + 12  # color swatch + padding

    # Draw background
    x0 = w_img - legend_w - 4
    y0 = 4
    cv2.rectangle(canvas, (x0, y0), (x0 + legend_w, y0 + legend_h), (0, 0, 0), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + legend_w, y0 + legend_h), (128, 128, 128), 1)

    # Header
    cv2.putText(
        canvas, "Legend", (x0 + padding, y0 + line_h),
        font, config.label_font_scale, (255, 255, 255), config.label_thickness,
    )

    # Entries
    for i, (step, text, color_bgr) in enumerate(entries):
        y = y0 + (i + 1) * line_h + padding + line_h
        # Color swatch
        cv2.rectangle(canvas, (x0 + padding, y - 8), (x0 + padding + 8, y), color_bgr, -1)
        # Text
        cv2.putText(
            canvas, text, (x0 + padding + 12, y),
            font, config.label_font_scale, (255, 255, 255), config.label_thickness,
        )
