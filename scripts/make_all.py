"""Rerun every step of the analysis in order; outputs go to results/ and figures/.

By default it starts from the committed data/selected. With --network it first downloads the
samples again and rebuilds data/selected (about 1 GB), and reruns the normalization check.
The whole chain takes about 30 minutes, most of it the significance toys.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
NETWORK_STEPS = ["select_events.py", "check_normalization.py"]
STEPS = [
    "phase1_numbers.py",
    "plot_m4l.py",
    "mass_model.py",
    "mass_toys.py",
    "mass_simulation.py",
    "mass_data.py",
    "significance.py",
    "significance_data.py",
    "classifier.py",
    "classifier_data.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", action="store_true", help="redownload the samples")
    args = parser.parse_args()
    steps = NETWORK_STEPS + STEPS if args.network else STEPS
    for step in steps:
        start = time.time()
        print(f"{step} ...", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / step)], check=True)
        print(f"{step} done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
