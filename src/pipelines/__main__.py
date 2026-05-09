"""Allow `python -m src.pipelines.run_full --config <path>`."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.pipelines.run_full import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IndabaX Loan Default Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
