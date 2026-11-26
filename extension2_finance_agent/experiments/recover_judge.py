"""Independent CLI for retrying failed judgments on saved predictions."""
import argparse
from extension2_finance_agent.core.config import Config, local_path
from extension2_finance_agent.experiments.evaluate import recover, recover_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="extension2_finance_agent/configs/final.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Config.load(args.config)
    output = local_path(args.output)
    if output.is_relative_to(config.data):
        raise ValueError("Do not write into the original data package")
    result = recover(config, local_path(args.predictions), local_path(args.source), output)
    print({k: result[k] for k in ("evaluation_complete", "api_failed", "support_percentage")})


if __name__ == "__main__":
    main()
