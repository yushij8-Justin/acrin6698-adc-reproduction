"""Retrieve or replay a single prespecified ACRIN-6698 quantitative cohort."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

from fetch.image_io import COUNTS, ROOT, InputError, bindings_ok, require
from fetch.public_images import fetch

RUNNERS = {
    "original30": ROOT / "original30" / "run.py",
    "holdout18": ROOT / "holdout18" / "reproduce.py",
}


def check_environment():
    pin = (ROOT / ".python-version").read_text().strip()
    require(".".join(str(item) for item in sys.version_info[:3]) == pin, f"USE_PYTHON_{pin}")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        name, version = line.split("==")
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InputError(f"INSTALL_PINNED_{name}_{version}") from exc
        require(installed == version, f"USE_PINNED_{name}_{version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "analyze", "all"])
    parser.add_argument("cohort", choices=tuple(COUNTS))
    parser.add_argument("--cache", type=Path, required=True, help="Local directory containing imaging ZIPs")
    parser.add_argument("--output", type=Path, help="New results directory for analyze or all")
    parser.add_argument("--provider", choices=("auto", "nbia", "idc"), default="auto")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.timeout < 5 or args.timeout > 180:
        parser.error("--timeout must be 5–180 seconds")
    if args.action != "fetch" and (not args.output or args.output.exists()):
        parser.error("analysis requires a new --output directory")
    try:
        bindings_ok()
        if args.action != "fetch":
            check_environment()
        if args.action in {"fetch", "all"}:
            print(fetch(args.cohort, args.cache, args.provider, args.workers, args.timeout), flush=True)
        if args.action in {"analyze", "all"}:
            subprocess.run(
                [sys.executable, "-B", str(RUNNERS[args.cohort]), "--cache", str(args.cache.resolve()),
                 "--output", str(args.output.resolve())],
                cwd=ROOT, check=True,
            )
    except InputError as error:
        print(f"Reproduction could not proceed: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError:
        print("Cohort computation or result verification failed.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
