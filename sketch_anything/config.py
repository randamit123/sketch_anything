"""Top-level configuration for the Sketch Anything pipeline.

Composes VLM, rendering, and LIBERO settings into a single Config object.
Supports loading from YAML or using sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sketch_anything.rendering.config import RenderConfig
from sketch_anything.vlm.config import VLMConfig


@dataclass
class LiberoConfig:
    """LIBERO environment settings."""

    camera_names: List[str] = field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )
    image_width: int = 256
    image_height: int = 256


@dataclass
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    save_intermediates: bool = True
    output_dir: str = "outputs"


@dataclass
class Config:
    """Top-level pipeline configuration."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    rendering: RenderConfig = field(default_factory=RenderConfig)
    libero: LiberoConfig = field(default_factory=LiberoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def default(cls) -> Config:
        """Return configuration with sensible defaults."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated Config instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        vlm_data = data.get("vlm", {})
        rendering_data = data.get("rendering", {})
        libero_data = data.get("libero", {})
        logging_data = data.get("logging", {})

        # Handle color dict in rendering (stored as lists in YAML)
        rendering_data.pop("colors", None)  # Colors are hardcoded in palette

        return cls(
            vlm=VLMConfig(**vlm_data),
            rendering=RenderConfig(**rendering_data),
            libero=LiberoConfig(**libero_data),
            logging=LoggingConfig(**logging_data),
        )
