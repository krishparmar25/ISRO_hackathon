from __future__ import annotations

import argparse

from lunarice_pds4.pipeline import run_pipeline, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="LunarIce-Net PDS4-native processing CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate", help="Check PRADAN folder paths and PDS4 labels")
    validate_parser.add_argument("--config", default="config.yaml")

    run_parser = sub.add_parser("run", help="Run the DFSAR + OHRC processing pipeline")
    run_parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    if args.command == "validate":
        validate_config(args.config)
    elif args.command == "run":
        run_pipeline(args.config)


if __name__ == "__main__":
    main()

