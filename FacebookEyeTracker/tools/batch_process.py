"""
Batch processing utility for running pipeline steps on multiple participants.

Supports running individual pipeline steps or full pipeline for multiple participants.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], description: str, verbose: bool = False) -> bool:
    """Run a command and return success status."""
    if verbose:
        print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
        )
        if not verbose and result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        if not verbose:
            print(f"Exit code: {e.returncode}")
            if e.stdout:
                print(f"Output: {e.stdout}")
            if e.stderr:
                print(f"Error: {e.stderr}")
        return False


def process_participant(
    participant: str,
    steps: list[str],
    duration: int | None = None,
    width: int = 1920,
    height: int = 1080,
    verbose: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Process a single participant through specified pipeline steps."""
    results: dict[str, Any] = {
        "participant": participant,
        "steps": {},
    }

    print(f"\n{'=' * 60}")
    print(f"Processing participant: {participant}")
    print(f"Steps: {', '.join(steps)}")
    print(f"{'=' * 60}")

    data_dir = Path("data") / participant
    gaze_file = data_dir / "gaze.csv"
    gaze_clean_file = data_dir / "gaze_clean.csv"

    for step in steps:
        print(f"\n--- Step: {step} ---")

        if step == "generate":
            if duration is None:
                print("ERROR: --duration required for 'generate' step")
                results["steps"][step] = False
                continue

            cmd = ["python", "scripts/generate.py", str(duration), participant]
            success = run_command(cmd, "Data generation", verbose)
            results["steps"][step] = success

        elif step == "process":
            if not gaze_file.exists():
                print(f"ERROR: Raw gaze file not found: {gaze_file}")
                results["steps"][step] = False
                continue

            cmd = [
                "python",
                "scripts/gazeProcess.py",
                str(gaze_file),
                str(gaze_clean_file),
                str(width),
                str(height),
            ]
            success = run_command(cmd, "Gaze processing", verbose)
            results["steps"][step] = success

        elif step == "match":
            if not gaze_clean_file.exists():
                print(f"ERROR: Processed gaze file not found: {gaze_clean_file}")
                print("Run 'process' step first")
                results["steps"][step] = False
                continue

            cmd = ["python", "scripts/match.py", participant]
            success = run_command(cmd, "Data matching", verbose)
            results["steps"][step] = success

        elif step == "visualize":
            gaze_posts_dir = data_dir / "gaze_posts"
            if not gaze_posts_dir.exists():
                print(f"ERROR: Gaze posts directory not found: {gaze_posts_dir}")
                print("Run 'match' step first")
                results["steps"][step] = False
                continue

            cmd = ["python", "scripts/visualizations.py", participant]
            success = run_command(cmd, "Visualization generation", verbose)
            results["steps"][step] = success

        elif step == "screenshot":
            if duration is None:
                print("ERROR: --duration required for 'screenshot' step")
                results["steps"][step] = False
                continue

            cmd = ["python", "scripts/screenshot.py", participant, str(duration)]
            success = run_command(cmd, "Screenshot capture", verbose)
            results["steps"][step] = success

        if not results["steps"][step]:
            print(f"⚠️  Step '{step}' failed for {participant}")
            if not continue_on_error:
                print("Stopping processing for this participant")
                break
        else:
            print(f"✅ Step '{step}' completed successfully")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch process multiple participants through the eye-tracking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  generate   - Collect eye-tracking data (requires --duration)
  process    - Clean and process raw gaze data
  match      - Match gaze data with posts
  visualize  - Generate heatmaps and scanpaths
  screenshot - Capture screenshots (requires --duration)

Examples:
  # Run visualizations for multiple participants
  python batch_process.py --participants alice bob charlie --steps visualize

  # Run full pipeline (process -> match -> visualize)
  python batch_process.py --participants alice bob --steps process match visualize

  # Collect data for new participants (60 second duration)
  python batch_process.py --participants new_user --steps generate screenshot --duration 60

  # Run all steps with custom resolution
  python batch_process.py --participants alice --steps process match visualize --width 2560 --height 1440
        """,
    )

    parser.add_argument(
        "--participants",
        "-p",
        nargs="+",
        required=True,
        help="Participant name(s) to process",
    )

    parser.add_argument(
        "--steps",
        "-s",
        nargs="+",
        choices=["generate", "process", "match", "visualize", "screenshot"],
        required=True,
        help="Pipeline steps to run",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        help="Recording duration in seconds (required for 'generate' and 'screenshot' steps)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Screen width in pixels (default: 1920)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Screen height in pixels (default: 1080)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output from each command",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if a step fails",
    )

    args = parser.parse_args()

    # Validate duration requirement
    if ("generate" in args.steps or "screenshot" in args.steps) and args.duration is None:
        parser.error("--duration is required when using 'generate' or 'screenshot' steps")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    print(f"Participants: {', '.join(args.participants)}")
    print(f"Steps: {', '.join(args.steps)}")
    if args.duration:
        print(f"Duration: {args.duration} seconds")
    print(f"Resolution: {args.width}x{args.height}")
    print("=" * 60)

    all_results = []
    total_success = 0
    total_failed = 0

    for participant in args.participants:
        results = process_participant(
            participant,
            args.steps,
            args.duration,
            args.width,
            args.height,
            args.verbose,
            args.continue_on_error,
        )
        all_results.append(results)

        # Count successes and failures
        for _step, success in results["steps"].items():
            if success:
                total_success += 1
            else:
                total_failed += 1

    # Final summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)

    for result in all_results:
        participant = result["participant"]
        steps = result["steps"]
        success_count = sum(1 for s in steps.values() if s)
        total_count = len(steps)

        status = "✅" if success_count == total_count else "⚠️"
        print(f"{status} {participant}: {success_count}/{total_count} steps succeeded")

    print("\n" + "-" * 60)
    print(f"Total steps succeeded: {total_success}")
    print(f"Total steps failed: {total_failed}")
    print("=" * 60)

    # Exit with error code if any failures
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
