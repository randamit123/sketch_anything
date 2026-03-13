"""Top-level pipeline entry point for Path 1."""
from path1_execution.config import Path1Config, OrchestratorResult
from path1_execution.agents.orchestrator import Orchestrator


def run_pipeline(
    hdf5_path: str,
    task_instruction: str = None,
    config: Path1Config = None,
    demo_index: int = 0,
) -> OrchestratorResult:
    """Run the Path 1 pipeline end-to-end.

    Args:
        hdf5_path: Path to LIBERO demo HDF5 file.
        task_instruction: Natural language task description. If None, extracted from HDF5.
        config: Pipeline configuration. If None, uses defaults.
        demo_index: Which demo episode to load initial state from.

    Returns:
        OrchestratorResult with success flag and final results.
    """
    if config is None:
        config = Path1Config()

    orchestrator = Orchestrator()
    return orchestrator.run(hdf5_path, task_instruction, config, demo_index=demo_index)
