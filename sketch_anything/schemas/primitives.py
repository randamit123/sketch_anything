"""Pydantic models for sketch primitive schemas.

Defines the Position discriminated union, three primitive types (Arrow, Circle,
Gripper), and the top-level SketchPrimitives container. These models enforce
schema validation at construction time and can be used with Outlines for
constrained VLM decoding.

Coordinate system: normalized [0.0, 1.0], origin top-left, x right, y down.
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Position types
# ---------------------------------------------------------------------------

class AbsolutePosition(BaseModel):
    """A position specified by normalized (x, y) coordinates."""

    type: Literal["absolute"]
    coords: Tuple[float, float]

    @field_validator("coords")
    @classmethod
    def validate_coords(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if not (0.0 <= v[0] <= 1.0 and 0.0 <= v[1] <= 1.0):
            raise ValueError(
                f"Absolute coordinate ({v[0]}, {v[1]}) is outside valid range [0, 1]"
            )
        return v


class ObjectRelativePosition(BaseModel):
    """A position relative to a detected object's bounding box anchor."""

    type: Literal["object_relative"]
    object_id: str
    anchor: Literal[
        "center",
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]
    offset: Tuple[float, float] = (0.0, 0.0)


Position = Union[AbsolutePosition, ObjectRelativePosition]


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

class ArrowPrimitive(BaseModel):
    """Directed end-effector motion from start to end, optionally via waypoints."""

    type: Literal["arrow"]
    start: Position
    end: Position
    waypoints: List[Position] = Field(default_factory=list)
    step: int = Field(ge=1)


class CirclePrimitive(BaseModel):
    """Marks a point of interest (grasp, release, contact, etc.)."""

    type: Literal["circle"]
    center: Position
    radius: float = Field(ge=0.01, le=0.15)
    purpose: Literal[
        "grasp_point",
        "release_point",
        "contact",
        "rotation_pivot",
        "target_location",
    ]
    step: int = Field(ge=1)


class GripperPrimitive(BaseModel):
    """Indicates a gripper state change at a specific location."""

    type: Literal["gripper"]
    position: Position
    action: Literal["open", "close"]
    step: int = Field(ge=1)


Primitive = Union[ArrowPrimitive, CirclePrimitive, GripperPrimitive]


# ---------------------------------------------------------------------------
# Top-level output
# ---------------------------------------------------------------------------

class SketchPrimitives(BaseModel):
    """Container for all sketch primitives produced by the VLM."""

    primitives: List[Primitive]
