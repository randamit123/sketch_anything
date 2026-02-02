"""VLM configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLMConfig:
    """Configuration for VLM primitive generation.

    Set ``use_mock=True`` for development/testing without a GPU.
    On the Linux production machine, use the default settings with
    ``use_constrained_decoding=True`` for guaranteed valid JSON.
    """

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.1
    max_retries: int = 3
    use_constrained_decoding: bool = True
    device: str = "cuda"
    use_mock: bool = False
