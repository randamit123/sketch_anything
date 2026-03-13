"""Integration tests for the mock pipeline.

Tests end-to-end flow with mock VLM. No LIBERO or GPU required.
"""

import pathlib

import numpy as np
import pytest

from sketch_anything.config import Config
from sketch_anything.rendering.renderer import render_primitives
from sketch_anything.schemas.primitives import SketchPrimitives
from sketch_anything.validation.validator import validate_primitives
from sketch_anything.vlm.config import VLMConfig
from sketch_anything.vlm.generator import VLMPrimitiveGenerator, _generate_mock_primitives
from sketch_anything.vlm.prompt import format_prompt, format_object_registry

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

TASK = "pick up the red block and place it in the blue bowl"


class TestMockGenerator:
    def test_generates_valid_primitives(self):
        primitives = _generate_mock_primitives(REGISTRY, TASK)
        assert isinstance(primitives, SketchPrimitives)
        assert len(primitives.primitives) > 0

    def test_passes_validation(self):
        primitives = _generate_mock_primitives(REGISTRY, TASK)
        result = validate_primitives(primitives, REGISTRY)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_has_expected_structure(self):
        primitives = _generate_mock_primitives(REGISTRY, TASK)
        types = [p.type for p in primitives.primitives]
        assert "circle" in types
        assert "arrow" in types
        assert "gripper" in types

    def test_step_ordering(self):
        primitives = _generate_mock_primitives(REGISTRY, TASK)
        steps = [p.step for p in primitives.primitives]
        assert 1 in steps
        assert max(steps) >= 2

    def test_single_object_registry(self):
        """Test with only one non-gripper object."""
        registry = {
            "gripper": REGISTRY["gripper"],
            "red_block": REGISTRY["red_block"],
        }
        primitives = _generate_mock_primitives(registry, "pick up the red block")
        result = validate_primitives(primitives, registry)
        assert result.is_valid

    def test_empty_registry(self):
        primitives = _generate_mock_primitives({}, TASK)
        assert len(primitives.primitives) == 0


class TestVLMPrimitiveGeneratorMock:
    def test_mock_mode(self):
        config = VLMConfig(use_mock=True)
        generator = VLMPrimitiveGenerator(config)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        primitives = generator.generate(image, REGISTRY, TASK)
        assert isinstance(primitives, SketchPrimitives)
        assert len(primitives.primitives) > 0


class TestPromptFormatting:
    def test_format_registry(self):
        text = format_object_registry(REGISTRY)
        assert "red_block" in text
        assert "blue_bowl" in text
        assert "gripper" in text

    def test_format_prompt(self):
        prompt = format_prompt(REGISTRY, TASK)
        assert TASK in prompt
        assert "red_block" in prompt
        assert "object_relative" in prompt
        assert "arrow" in prompt


class TestEndToEndMock:
    def test_full_pipeline_mock(self):
        """Mock VLM -> validate -> render on a synthetic image."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)

        # Generate
        primitives = _generate_mock_primitives(REGISTRY, TASK)

        # Validate
        result = validate_primitives(primitives, REGISTRY)
        assert result.is_valid, f"Errors: {result.errors}"

        # Render (default render_scale=2 doubles dimensions)
        annotated = render_primitives(image, primitives, REGISTRY)
        assert annotated.shape == (512, 512, 3)
        assert annotated.sum() > 0  # Something was drawn


_CONFIG_PATH = pathlib.Path(__file__).parents[3] / "config" / "default.yaml"


class TestConfig:
    def test_default_config(self):
        config = Config.default()
        assert config.libero.camera_names == ["agentview", "robot0_eye_in_hand", "frontview"]
        assert config.vlm.device == "cuda"

    def test_yaml_config(self):
        if not _CONFIG_PATH.exists():
            pytest.skip(f"config/default.yaml not found at {_CONFIG_PATH}")
        config = Config.from_yaml(str(_CONFIG_PATH))
        assert config.vlm.device == "cuda"
        assert config.libero.image_width == 256


class TestJsonExtraction:
    def test_extract_json_plain(self):
        from sketch_anything.vlm.generator import _extract_json
        result = _extract_json('{"primitives": []}')
        assert result == {"primitives": []}

    def test_extract_json_from_markdown_fence(self):
        from sketch_anything.vlm.generator import _extract_json
        text = '```json\n{"primitives": []}\n```'
        result = _extract_json(text)
        assert result == {"primitives": []}

    def test_extract_json_with_surrounding_text(self):
        from sketch_anything.vlm.generator import _extract_json
        text = 'Here is my answer:\n{"primitives": []}\nDone.'
        result = _extract_json(text)
        assert result == {"primitives": []}

    def test_extract_json_raises_on_garbage(self):
        from sketch_anything.vlm.generator import _extract_json
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_json("no json here at all")


class TestNormalizePosition:
    def test_already_flat_absolute(self):
        from sketch_anything.vlm.generator import _normalize_position
        pos = {"type": "absolute", "coords": [0.5, 0.3]}
        assert _normalize_position(pos) == pos

    def test_nested_absolute_dict(self):
        from sketch_anything.vlm.generator import _normalize_position
        result = _normalize_position({"absolute": {"coords": [0.5, 0.3]}})
        assert result == {"type": "absolute", "coords": [0.5, 0.3]}

    def test_nested_absolute_list(self):
        from sketch_anything.vlm.generator import _normalize_position
        result = _normalize_position({"absolute": [0.5, 0.3]})
        assert result == {"type": "absolute", "coords": [0.5, 0.3]}

    def test_nested_object_relative(self):
        from sketch_anything.vlm.generator import _normalize_position
        result = _normalize_position({
            "object_relative": {"object_id": "red_block", "anchor": "center"}
        })
        assert result == {
            "type": "object_relative",
            "object_id": "red_block",
            "anchor": "center",
        }

    def test_already_flat_object_relative(self):
        from sketch_anything.vlm.generator import _normalize_position
        pos = {"type": "object_relative", "object_id": "x", "anchor": "top"}
        assert _normalize_position(pos) == pos
