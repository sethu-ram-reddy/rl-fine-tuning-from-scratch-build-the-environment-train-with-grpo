"""Run the original course protocol through the project CLI."""

import sys

from rlvr.cli import main as run_experiment


def main(argv=()):
    run_experiment(["--protocol", "course", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
