"""CLI entry point for Path 1 pipeline.

Usage:
    python -m path1_execution.tools.run_path1 \\
        --hdf5 /path/to/demo.hdf5 \\
        --output ./outputs/path1 \\
        --demo-index 0 \\
        [--mock | --no-mock] \\
        --max-retries 5 \\
        --gpu 0
"""
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Path 1: Trajectory planning + execution")
    p.add_argument("--hdf5", required=True, help="Path to LIBERO demo HDF5 file")
    p.add_argument("--output", default="./outputs/path1", help="Output directory")
    p.add_argument("--demo-index", type=int, default=0, help="Demo index from HDF5")
    p.add_argument("--max-retries", type=int, default=5, help="Maximum retry attempts")
    p.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        default=True,
        help="Use mock planner (default)",
    )
    p.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Use o3 LLM planner instead of mock",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (sets CUDA_VISIBLE_DEVICES)",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["agentview", "frontview", "robot0_eye_in_hand"],
        help="Camera names to use",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from path1_execution.config import Path1Config
    from path1_execution.pipeline import run_pipeline

    config = Path1Config(
        use_mock_planner=args.mock,
        max_retries=args.max_retries,
        output_dir=args.output,
        camera_names=args.cameras,
    )

    logger.info("Starting Path 1 pipeline")
    logger.info("  HDF5:         %s", args.hdf5)
    logger.info("  Mock planner: %s", config.use_mock_planner)
    logger.info("  Max retries:  %d", config.max_retries)
    logger.info("  Output dir:   %s", config.output_dir)

    result = run_pipeline(args.hdf5, config=config, demo_index=args.demo_index)

    if result.success:
        logger.info("SUCCESS after %d attempt(s)", result.attempts)
        sys.exit(0)
    else:
        logger.error("FAILED after %d attempt(s)", result.attempts)
        sys.exit(1)


if __name__ == "__main__":
    main()
