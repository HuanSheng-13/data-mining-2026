"""Run from repository root: python -m extension2_finance_agent --help."""
import argparse
import json
from pathlib import Path

from extension2_finance_agent.core.config import Config, local_path, load_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "freeze", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="extension2_finance_agent/configs/final.json")
        if name in ("run", "freeze", "evaluate"):
            command.add_argument("--output", type=Path, required=True)
        if name in ("run", "evaluate"):
            command.add_argument("--split", choices=["train", "test"], default="train")
            command.add_argument("--limit", type=int)
        if name == "run":
            command.add_argument("--question-id", action="append")
            command.add_argument("--frozen", type=Path)
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--check-only", action="store_true")
            command.add_argument("--judge-url")
            command.add_argument("--model")
    args = parser.parse_args()
    try:
        config = Config.load(args.config)
        if hasattr(args, "output") and local_path(args.output).is_relative_to(config.data):
            raise ValueError("Output must be outside the original data package")
        if args.command == "prepare":
            manifest = load_factory(config.prepare_factory)(config)
            print(json.dumps({"reports": len(manifest["reports"]), "chunks": manifest["chunks"]}))
        elif args.command == "freeze":
            from extension2_finance_agent.experiments.runner import freeze
            print(json.dumps(freeze(config, local_path(args.output))))
        elif args.command == "run":
            from extension2_finance_agent.experiments.runner import run
            result = run(config, args.split, local_path(args.output), args.question_id, args.limit,
                         local_path(args.frozen) if args.frozen else None)
            return 0 if result["status"] == "completed" else 1
        else:
            from extension2_finance_agent.experiments.evaluate import evaluate
            return evaluate(config, args.split, local_path(args.predictions), local_path(args.output),
                            args.check_only, args.judge_url, args.model, args.limit)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
