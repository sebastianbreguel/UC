import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], description: str) -> None:
    """Run a pipeline step and exit on failure."""
    print(f"\n--- {description} ---")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eye-tracking data collection and visualization pipeline")
    parser.add_argument("--duration", type=int, required=True, help="Recording duration in seconds")
    parser.add_argument("--name", type=str, required=True, help="Participant identifier")
    parser.add_argument("--width", type=int, default=1920, help="Screen width in pixels")
    parser.add_argument("--height", type=int, default=1080, help="Screen height in pixels")

    args = parser.parse_args()

    # Create output directories
    base = Path("data") / args.name
    for subdir in ["gaze_posts", "times", "screenshots", "heatmaps", "scanpath"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Directories for {args.name} created successfully.")

    gaze_file = base / "gaze"

    run_step(
        [sys.executable, "scripts/generate.py", str(args.duration), args.name],
        "Collecting eye-tracking data",
    )

    run_step(
        [
            sys.executable,
            "scripts/gazeProcess.py",
            f"{gaze_file}.csv",
            f"{gaze_file}_clean.csv",
            str(args.width),
            str(args.height),
        ],
        "Processing gaze data",
    )

    run_step(
        [sys.executable, "scripts/match.py", args.name],
        "Matching data with post metadata",
    )

    run_step(
        [sys.executable, "scripts/visualizations.py", args.name],
        "Generating visualizations",
    )

    print(f"\nPipeline completed for {args.name}")


if __name__ == "__main__":
    main()
