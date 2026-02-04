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
        RGB image array with primitives rendered. If render_scale > 1,
        the output will be larger than the input.
    """
    if config is None:
        config = RenderConfig()

    # Work in BGR for OpenCV
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    # Upscale image if render_scale > 1
    scale = config.render_scale
    if scale > 1:
        new_w, new_h = w * scale, h * scale
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = new_h, new_w

    line_type = cv2.LINE_AA if config.use_antialiasing else cv2.LINE_8

    # Create a scaled config for rendering (scale pixel-unit parameters)
    from dataclasses import replace
    scaled_config = replace(
        config,
        arrow_thickness=config.arrow_thickness * scale,
        waypoint_radius=config.waypoint_radius * scale,
        circle_stroke_width=config.circle_stroke_width * scale,
        circle_center_radius=config.circle_center_radius * scale,
        gripper_marker_size=config.gripper_marker_size * scale,
        bbox_thickness=config.bbox_thickness * scale,
        label_font_scale=config.label_font_scale * scale,
        label_thickness=max(1, config.label_thickness * scale),
    )

    # --- Object bounding boxes + labels ---
    if config.draw_object_bboxes and object_registry:
        _draw_object_bboxes(canvas, object_registry, w, h, scaled_config, line_type)

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
            _render_circle(canvas, c, object_registry, w, h, color_bgr, scaled_config, line_type)
            if scaled_config.draw_labels:
                _label_circle(canvas, c, object_registry, w, h, color_bgr, scaled_config, line_type)

        for a in arrows:
            _render_arrow(canvas, a, object_registry, w, h, color_bgr, scaled_config, line_type)
            if scaled_config.draw_labels:
                _label_arrow(canvas, a, object_registry, w, h, color_bgr, scaled_config, line_type)

        for g in grippers:
            _render_gripper(canvas, g, object_registry, w, h, color_bgr, scaled_config, line_type)
            if scaled_config.draw_labels:
                _label_gripper(canvas, g, object_registry, w, h, color_bgr, scaled_config, line_type)

    # --- Legend ---
    if scaled_config.draw_legend and primitives.primitives:
        _draw_legend(canvas, primitives, scaled_config)

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
    """Draw an arrow with optional waypoints.

    When waypoints are present, a smooth quadratic/cubic Bézier spline is
    drawn through them instead of straight line segments.  The arrowhead
    direction is derived from the final tangent of the curve so it points
    correctly even on curved paths.
    """
    start = resolve_position(arrow.start, registry, w, h)
    end = resolve_position(arrow.end, registry, w, h)
    waypoints = [resolve_position(wp, registry, w, h) for wp in arrow.waypoints]

    # Build point sequence: start -> waypoints -> end
    control_pts = [start] + waypoints + [end]

    if len(control_pts) <= 2:
        # No waypoints — simple straight line
        cv2.line(canvas, start, end, color, config.arrow_thickness, line_type)
    else:
        # Smooth Bézier curve through control points
        curve_pts = _bezier_curve(control_pts, num_samples=64)
        # Draw as a polyline for smooth rendering
        pts_array = np.array(curve_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_array], isClosed=False, color=color,
                      thickness=config.arrow_thickness, lineType=line_type)

    # Draw waypoint dots
    for wp_px in waypoints:
        cv2.circle(canvas, wp_px, config.waypoint_radius, color, -1, line_type)

    # Draw arrowhead at end — use the curve tangent for direction
    if len(control_pts) <= 2:
        from_pt = control_pts[-2]
    else:
        # Use the second-to-last point on the Bézier curve for tangent
        curve_pts = _bezier_curve(control_pts, num_samples=64)
        from_pt = curve_pts[-4] if len(curve_pts) >= 4 else curve_pts[-2]

    _draw_arrowhead(canvas, from_pt, end, color, config, line_type)


def _bezier_curve(
    control_points: List[Tuple[int, int]],
    num_samples: int = 64,
) -> List[Tuple[int, int]]:
    """Compute a composite Bézier curve passing through all control points.

    For 3 points (start, waypoint, end), a quadratic Bézier is used.
    For 4+ points, a Catmull-Rom spline is computed through all points,
    which guarantees the curve passes through every control point (unlike
    a raw cubic Bézier where interior points are just attractors).
    """
    n = len(control_points)
    if n < 2:
        return list(control_points)
    if n == 2:
        return list(control_points)

    if n == 3:
        # Quadratic Bézier: curve passes through start, near waypoint, and end.
        # To make it pass *through* the waypoint we compute a virtual control
        # point: C = 2*P1 - 0.5*(P0 + P2)  (standard trick).
        p0 = np.array(control_points[0], dtype=float)
        p1 = np.array(control_points[1], dtype=float)
        p2 = np.array(control_points[2], dtype=float)
        # Virtual control point so the curve passes through p1 at t=0.5
        c1 = 2.0 * p1 - 0.5 * (p0 + p2)

        result = []
        for i in range(num_samples + 1):
            t = i / num_samples
            # Quadratic Bézier with virtual control point
            pt = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c1 + t ** 2 * p2
            result.append((int(round(pt[0])), int(round(pt[1]))))
        return result

    # 4+ points: Catmull-Rom spline (passes through all points)
    return _catmull_rom_chain(control_points, num_samples)


def _catmull_rom_chain(
    points: List[Tuple[int, int]],
    total_samples: int = 64,
    alpha: float = 0.5,
) -> List[Tuple[int, int]]:
    """Catmull-Rom spline through a sequence of points.

    The ``alpha`` parameter controls the spline type:
        0.0 = uniform, 0.5 = centripetal (default, no cusps), 1.0 = chordal.
    """
    pts = [np.array(p, dtype=float) for p in points]
    n = len(pts)

    # Pad with ghost points at the ends (reflection)
    pts = [2 * pts[0] - pts[1]] + pts + [2 * pts[-1] - pts[-2]]
    segments = n - 1
    samples_per_seg = max(total_samples // segments, 8)

    result: List[Tuple[int, int]] = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            # Catmull-Rom basis matrix
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            result.append((int(round(pt[0])), int(round(pt[1]))))

    # Always include the final point
    result.append((int(round(pts[-2][0])), int(round(pts[-2][1]))))
    return result


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

def build_legend_data(primitives: SketchPrimitives) -> dict:
    """Build a JSON-serializable legend describing the primitives.

    Returns a dict with step-by-step descriptions and color info, suitable
    for saving alongside the annotated image so the legend does not need
    to be burned into the image itself.

    Example output::

        {
            "steps": [
                {
                    "step": 1,
                    "color_rgb": [0, 255, 0],
                    "color_hex": "#00FF00",
                    "primitives": ["grasp_point (circle)", "move (arrow)"]
                },
                ...
            ]
        }
    """
    all_steps = sorted({p.step for p in primitives.primitives})
    steps = []

    for step in all_steps:
        color_rgb = get_step_color(step)
        color_hex = "#{:02X}{:02X}{:02X}".format(*color_rgb)

        step_prims = [p for p in primitives.primitives if p.step == step]
        descriptions: List[str] = []
        for p in step_prims:
            if isinstance(p, ArrowPrimitive):
                descriptions.append("move (arrow)")
            elif isinstance(p, CirclePrimitive):
                descriptions.append(f"{p.purpose} (circle)")
            elif isinstance(p, GripperPrimitive):
                descriptions.append(f"gripper {p.action} (gripper)")

        steps.append({
            "step": step,
            "color_rgb": list(color_rgb),
            "color_hex": color_hex,
            "primitives": descriptions,
        })

    return {"steps": steps}


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
