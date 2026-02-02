"""Tests for the deterministic renderer.

Tests position resolution, individual primitive rendering, and the
complete pick-and-place rendering from CLAUDE.md Section 12.
"""

import numpy as np
import pytest

from sketch_anything.rendering.config import RenderConfig, get_step_color
from sketch_anything.rendering.renderer import render_primitives
from sketch_anything.rendering.resolver import compute_anchor, resolve_position
from sketch_anything.schemas.primitives import (
    AbsolutePosition,
    ObjectRelativePosition,
    SketchPrimitives,
)

REGISTRY = {
    "gripper": {
        "id": "gripper",
        "label": "robot gripper",
        "bbox": [0.42, 0.18, 0.48, 0.24],
        "center": [0.45, 0.21],
    },
    "red_block": {
        "id": "red_block",
        "label": "red block",
        "bbox": [0.28, 0.42, 0.36, 0.50],
        "center": [0.32, 0.46],
    },
    "blue_bowl": {
        "id": "blue_bowl",
        "label": "blue bowl",
        "bbox": [0.54, 0.38, 0.64, 0.45],
        "center": [0.59, 0.415],
    },
}


class TestComputeAnchor:
    def test_center(self):
        bbox = [0.2, 0.3, 0.4, 0.5]
        x, y = compute_anchor(bbox, "center")
        assert abs(x - 0.3) < 1e-6
        assert abs(y - 0.4) < 1e-6

    def test_top(self):
        bbox = [0.2, 0.3, 0.4, 0.5]
        x, y = compute_anchor(bbox, "top")
        assert abs(x - 0.3) < 1e-6
        assert abs(y - 0.3) < 1e-6

    def test_bottom_right(self):
        bbox = [0.2, 0.3, 0.4, 0.5]
        x, y = compute_anchor(bbox, "bottom_right")
        assert abs(x - 0.4) < 1e-6
        assert abs(y - 0.5) < 1e-6

    def test_invalid_anchor(self):
        with pytest.raises(ValueError, match="Unknown anchor"):
            compute_anchor([0, 0, 1, 1], "middle")


class TestResolvePosition:
    def test_absolute(self):
        pos = AbsolutePosition(type="absolute", coords=(0.5, 0.75))
        px = resolve_position(pos, {}, 256, 256)
        assert px == (128, 192)

    def test_object_relative_center(self):
        pos = ObjectRelativePosition(
            type="object_relative",
            object_id="red_block",
            anchor="center",
        )
        px = resolve_position(pos, REGISTRY, 256, 256)
        expected_x = round(0.32 * 256)
        expected_y = round(0.46 * 256)
        assert px == (expected_x, expected_y)

    def test_object_relative_with_offset(self):
        pos = ObjectRelativePosition(
            type="object_relative",
            object_id="red_block",
            anchor="top",
            offset=(0.0, -0.03),
        )
        px = resolve_position(pos, REGISTRY, 256, 256)
        # top anchor: x = (0.28+0.36)/2=0.32, y = 0.42
        # with offset: y = 0.42 - 0.03 = 0.39
        expected_x = round(0.32 * 256)
        expected_y = round(0.39 * 256)
        assert px == (expected_x, expected_y)


class TestGetStepColor:
    def test_step_1_green(self):
        assert get_step_color(1) == (0, 255, 0)

    def test_step_4_magenta(self):
        assert get_step_color(4) == (255, 0, 255)

    def test_step_6_plus_purple(self):
        assert get_step_color(6) == (180, 0, 255)
        assert get_step_color(10) == (180, 0, 255)


class TestRenderPrimitives:
    def _make_image(self, w=256, h=256):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_output_shape_matches_input(self):
        image = self._make_image()
        sp = SketchPrimitives(primitives=[])
        result = render_primitives(image, sp, REGISTRY)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_empty_primitives_no_sketch_drawn(self):
        """With no primitives, only bounding boxes may be drawn (if enabled)."""
        image = self._make_image()
        sp = SketchPrimitives(primitives=[])
        config = RenderConfig(draw_object_bboxes=False, draw_legend=False)
        result = render_primitives(image, sp, REGISTRY, config=config)
        np.testing.assert_array_equal(result, image)

    def test_circle_modifies_pixels(self):
        image = self._make_image()
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "circle",
                    "center": {"type": "absolute", "coords": [0.5, 0.5]},
                    "radius": 0.04,
                    "purpose": "grasp_point",
                    "step": 1,
                }
            ]
        })
        result = render_primitives(image, sp, REGISTRY)
        # Some pixels should be non-zero (green for step 1)
        assert result.sum() > 0

    def test_arrow_modifies_pixels(self):
        image = self._make_image()
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "arrow",
                    "start": {"type": "absolute", "coords": [0.1, 0.1]},
                    "end": {"type": "absolute", "coords": [0.9, 0.9]},
                    "step": 1,
                }
            ]
        })
        result = render_primitives(image, sp, REGISTRY)
        assert result.sum() > 0

    def test_gripper_markers(self):
        image = self._make_image()
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "gripper",
                    "position": {"type": "absolute", "coords": [0.3, 0.3]},
                    "action": "close",
                    "step": 2,
                },
                {
                    "type": "gripper",
                    "position": {"type": "absolute", "coords": [0.7, 0.7]},
                    "action": "open",
                    "step": 4,
                },
            ]
        })
        result = render_primitives(image, sp, REGISTRY)
        assert result.sum() > 0

    def test_complete_pick_and_place_rendering(self):
        """Render the full CLAUDE.md Section 12 example."""
        image = self._make_image()
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "circle",
                    "center": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "radius": 0.04,
                    "purpose": "grasp_point",
                    "step": 1,
                },
                {
                    "type": "arrow",
                    "start": {"type": "object_relative", "object_id": "gripper", "anchor": "center"},
                    "end": {"type": "object_relative", "object_id": "red_block", "anchor": "top", "offset": [0.0, -0.03]},
                    "waypoints": [],
                    "step": 1,
                },
                {
                    "type": "gripper",
                    "position": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "action": "close",
                    "step": 2,
                },
                {
                    "type": "arrow",
                    "start": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "end": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center", "offset": [0.0, -0.05]},
                    "waypoints": [{"type": "absolute", "coords": [0.45, 0.25]}],
                    "step": 3,
                },
                {
                    "type": "circle",
                    "center": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center"},
                    "radius": 0.05,
                    "purpose": "release_point",
                    "step": 4,
                },
                {
                    "type": "gripper",
                    "position": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center", "offset": [0.0, -0.05]},
                    "action": "open",
                    "step": 4,
                },
            ]
        })
        result = render_primitives(image, sp, REGISTRY)
        assert result.shape == (256, 256, 3)
        assert result.sum() > 0
        # Multiple colors should be present (at least green for step 1 and another)
        # Check green channel has non-trivial values
        assert result[:, :, 1].max() > 0
