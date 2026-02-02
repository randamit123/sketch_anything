"""Rendering configuration and color palette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Step-based color palette (RGB).
# Matches CLAUDE.md Section 5.1.
STEP_COLORS: dict[int, Tuple[int, int, int]] = {
    1: (0, 255, 0),       # Green
    2: (0, 200, 255),     # Cyan
    3: (0, 100, 255),     # Blue
    4: (255, 0, 255),     # Magenta
    5: (255, 50, 50),     # Red
}
DEFAULT_COLOR: Tuple[int, int, int] = (180, 0, 255)  # Purple for step 6+


def get_step_color(step: int) -> Tuple[int, int, int]:
    """Return the RGB color for a given step number."""
    return STEP_COLORS.get(step, DEFAULT_COLOR)


def rgb_to_bgr(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert RGB tuple to BGR for OpenCV."""
    return (color[2], color[1], color[0])


@dataclass
class RenderConfig:
    """Configuration for the deterministic primitive renderer."""

    arrow_thickness: int = 3
    arrow_tip_length: float = 0.15
    waypoint_radius: int = 4
    circle_stroke_width: int = 3
    circle_center_radius: int = 4
    gripper_marker_size: int = 14
    use_antialiasing: bool = True

    # Label / annotation options
    draw_labels: bool = False
    draw_object_bboxes: bool = True
    draw_legend: bool = False
    label_font_scale: float = 0.4
    label_thickness: int = 1
    bbox_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow RGB
    bbox_thickness: int = 1
