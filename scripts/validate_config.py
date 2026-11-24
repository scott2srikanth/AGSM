import argparse
from pathlib import Path

from agsm.config import load_config, merge, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    for path in args.paths or sorted(Path("configs").rglob("*.yaml")):
        config = load_config(path)
        validate_config(config)
        for overrides in config.get("sweep", {}).values():
            validate_config(merge(config, overrides))
        print(f"Valid: {path}")


if __name__ == "__main__":
    main()
