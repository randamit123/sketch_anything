"""Tests for primitive validation.

Covers CLAUDE.md test cases 4-6 and warning generation.
"""

import pytest

from sketch_anything.schemas.primitives import SketchPrimitives
from sketch_anything.validation.validator import validate_primitives

# Shared registry used across tests.
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


class TestValidPrimitivesAccepted:
    """CLAUDE.md Test 4: valid primitives pass validation."""

    def test_valid_circle(self):
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "circle",
                    "center": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "radius": 0.04,
                    "purpose": "grasp_point",
                    "step": 1,
                }
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid
        assert result.errors == []

    def test_valid_complete_sequence(self):
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
                    "type": "gripper",
                    "position": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "action": "close",
                    "step": 2,
                },
                {
                    "type": "arrow",
                    "start": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "end": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center"},
                    "step": 3,
                },
                {
                    "type": "gripper",
                    "position": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center"},
                    "action": "open",
                    "step": 4,
                },
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid


class TestInvalidObjectIdRejected:
    """CLAUDE.md Test 5: unknown object_id produces error."""

    def test_nonexistent_object(self):
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "circle",
                    "center": {"type": "object_relative", "object_id": "nonexistent_object", "anchor": "center"},
                    "radius": 0.04,
                    "purpose": "grasp_point",
                    "step": 1,
                }
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert not result.is_valid
        assert any("Unknown object_id" in e for e in result.errors)


class TestOutOfBoundsRejected:
    """CLAUDE.md Test 6: out-of-bounds absolute coords produce error.

    Note: Pydantic catches this at parse time, but validate_primitives
    also re-checks for defense in depth.
    """

    def test_coords_out_of_range(self):
        """Pydantic should reject coords > 1.0 at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SketchPrimitives(**{
                "primitives": [
                    {
                        "type": "arrow",
                        "start": {"type": "absolute", "coords": [1.5, 0.5]},
                        "end": {"type": "absolute", "coords": [0.5, 0.5]},
                        "step": 1,
                    }
                ]
            })


class TestWarnings:
    def test_step_sequence_gap(self):
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
                    "type": "gripper",
                    "position": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "action": "close",
                    "step": 5,
                },
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid  # Gaps are warnings, not errors
        assert any("gap" in w.lower() for w in result.warnings)

    def test_large_offset_warning(self):
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "circle",
                    "center": {
                        "type": "object_relative",
                        "object_id": "red_block",
                        "anchor": "center",
                        "offset": [0.5, 0.0],
                    },
                    "radius": 0.04,
                    "purpose": "grasp_point",
                    "step": 1,
                }
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid
        assert any("offset" in w.lower() for w in result.warnings)

    def test_missing_grasp_before_transport(self):
        """Arrow at step 2 with no grasp circle before it."""
        sp = SketchPrimitives(**{
            "primitives": [
                {
                    "type": "arrow",
                    "start": {"type": "object_relative", "object_id": "red_block", "anchor": "center"},
                    "end": {"type": "object_relative", "object_id": "blue_bowl", "anchor": "center"},
                    "step": 2,
                }
            ]
        })
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid
        assert any("grasp" in w.lower() for w in result.warnings)

    def test_empty_primitives_valid(self):
        sp = SketchPrimitives(primitives=[])
        result = validate_primitives(sp, REGISTRY)
        assert result.is_valid
        assert result.errors == []
