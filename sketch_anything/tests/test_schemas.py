"""Tests for Pydantic schema models.

Covers valid parsing, invalid value rejection, and the complete
pick-and-place example from CLAUDE.md Section 12.
"""

import pytest
from pydantic import ValidationError

from sketch_anything.schemas.primitives import (
    AbsolutePosition,
    ArrowPrimitive,
    CirclePrimitive,
    GripperPrimitive,
    ObjectRelativePosition,
    SketchPrimitives,
)


class TestAbsolutePosition:
    def test_valid(self):
        pos = AbsolutePosition(type="absolute", coords=(0.5, 0.3))
        assert pos.coords == (0.5, 0.3)

    def test_boundary_values(self):
        pos = AbsolutePosition(type="absolute", coords=(0.0, 1.0))
        assert pos.coords == (0.0, 1.0)

    def test_out_of_range_x(self):
        with pytest.raises(ValidationError, match="valid range"):
            AbsolutePosition(type="absolute", coords=(1.5, 0.5))

    def test_out_of_range_y(self):
        with pytest.raises(ValidationError, match="valid range"):
            AbsolutePosition(type="absolute", coords=(0.5, -0.1))

    def test_negative_coords(self):
        with pytest.raises(ValidationError, match="valid range"):
            AbsolutePosition(type="absolute", coords=(-0.1, 0.5))


class TestObjectRelativePosition:
    def test_valid_with_defaults(self):
        pos = ObjectRelativePosition(
            type="object_relative",
            object_id="red_block",
            anchor="center",
        )
        assert pos.offset == (0.0, 0.0)

    def test_valid_with_offset(self):
        pos = ObjectRelativePosition(
            type="object_relative",
            object_id="gripper",
            anchor="top",
            offset=(0.0, -0.03),
        )
        assert pos.offset == (0.0, -0.03)

    def test_invalid_anchor(self):
        with pytest.raises(ValidationError):
            ObjectRelativePosition(
                type="object_relative",
                object_id="x",
                anchor="invalid_anchor",
            )


class TestArrowPrimitive:
    def test_minimal(self):
        arrow = ArrowPrimitive(
            type="arrow",
            start=AbsolutePosition(type="absolute", coords=(0.1, 0.1)),
            end=AbsolutePosition(type="absolute", coords=(0.9, 0.9)),
            step=1,
        )
        assert arrow.waypoints == []

    def test_with_waypoints(self):
        arrow = ArrowPrimitive(
            type="arrow",
            start=AbsolutePosition(type="absolute", coords=(0.1, 0.1)),
            end=AbsolutePosition(type="absolute", coords=(0.9, 0.9)),
            waypoints=[AbsolutePosition(type="absolute", coords=(0.5, 0.2))],
            step=3,
        )
        assert len(arrow.waypoints) == 1

    def test_invalid_step_zero(self):
        with pytest.raises(ValidationError):
            ArrowPrimitive(
                type="arrow",
                start=AbsolutePosition(type="absolute", coords=(0.1, 0.1)),
                end=AbsolutePosition(type="absolute", coords=(0.9, 0.9)),
                step=0,
            )


class TestCirclePrimitive:
    def test_valid(self):
        circle = CirclePrimitive(
            type="circle",
            center=ObjectRelativePosition(
                type="object_relative", object_id="red_block", anchor="center"
            ),
            radius=0.04,
            purpose="grasp_point",
            step=1,
        )
        assert circle.radius == 0.04

    def test_radius_too_small(self):
        with pytest.raises(ValidationError):
            CirclePrimitive(
                type="circle",
                center=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
                radius=0.005,
                purpose="grasp_point",
                step=1,
            )

    def test_radius_too_large(self):
        with pytest.raises(ValidationError):
            CirclePrimitive(
                type="circle",
                center=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
                radius=0.2,
                purpose="grasp_point",
                step=1,
            )

    def test_invalid_purpose(self):
        with pytest.raises(ValidationError):
            CirclePrimitive(
                type="circle",
                center=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
                radius=0.04,
                purpose="invalid_purpose",
                step=1,
            )


class TestGripperPrimitive:
    def test_close(self):
        g = GripperPrimitive(
            type="gripper",
            position=ObjectRelativePosition(
                type="object_relative", object_id="red_block", anchor="center"
            ),
            action="close",
            step=2,
        )
        assert g.action == "close"

    def test_open(self):
        g = GripperPrimitive(
            type="gripper",
            position=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
            action="open",
            step=4,
        )
        assert g.action == "open"

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            GripperPrimitive(
                type="gripper",
                position=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
                action="squeeze",
                step=1,
            )


class TestSketchPrimitives:
    def test_empty_list(self):
        sp = SketchPrimitives(primitives=[])
        assert len(sp.primitives) == 0

    def test_complete_pick_and_place(self):
        """Parse the full example from CLAUDE.md Section 12."""
        data = {
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
        }
        sp = SketchPrimitives(**data)
        assert len(sp.primitives) == 6
        assert sp.primitives[0].type == "circle"
        assert sp.primitives[2].type == "gripper"
        assert sp.primitives[3].type == "arrow"

    def test_json_round_trip(self):
        """Verify model_dump / re-parse round trip."""
        sp = SketchPrimitives(primitives=[
            CirclePrimitive(
                type="circle",
                center=AbsolutePosition(type="absolute", coords=(0.5, 0.5)),
                radius=0.04,
                purpose="contact",
                step=1,
            )
        ])
        data = sp.model_dump()
        sp2 = SketchPrimitives(**data)
        assert len(sp2.primitives) == 1
        assert sp2.primitives[0].radius == 0.04
