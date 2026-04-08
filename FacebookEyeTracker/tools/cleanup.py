"""
Cleanup utility for managing experimental data.

Provides options to clean screenshots, processed data, visualizations, or all data
for specified participants.
"""

import argparse
import shutil
from pathlib import Path


def cleanup_screenshots(participant_name: str, data_dir: str = "data") -> int:
    """Delete all screenshots for a participant."""
    screenshot_dir = Path(data_dir) / participant_name / "screenshots"
    if not screenshot_dir.exists():
        print(f"No screenshot directory found for {participant_name}")
        return 0

    count = 0
    for file_path in screenshot_dir.glob("screenshot*.png"):
        file_path.unlink()
        print(f"Deleted: {file_path}")
        count += 1

    return count


def cleanup_visualizations(participant_name: str, data_dir: str = "data") -> int:
    """Delete all generated visualizations (heatmaps and scanpaths)."""
    participant_dir = Path(data_dir) / participant_name
    count = 0

    for viz_type in ["heatmaps", "scanpath"]:
        viz_dir = participant_dir / viz_type
        if viz_dir.exists():
            for file_path in viz_dir.glob("*.png"):
                file_path.unlink()
                print(f"Deleted: {file_path}")
                count += 1

    return count


def cleanup_processed_data(participant_name: str, data_dir: str = "data") -> int:
    """Delete processed data (gaze_clean.csv, gaze_posts/, times/)."""
    participant_dir = Path(data_dir) / participant_name
    count = 0

    # Delete processed gaze file
    gaze_clean = participant_dir / "gaze_clean.csv"
    if gaze_clean.exists():
        gaze_clean.unlink()
        print(f"Deleted: {gaze_clean}")
        count += 1

    # Delete gaze_posts directory
    gaze_posts_dir = participant_dir / "gaze_posts"
    if gaze_posts_dir.exists():
        file_count = len(list(gaze_posts_dir.glob("*")))
        shutil.rmtree(gaze_posts_dir)
        print(f"Deleted directory: {gaze_posts_dir} ({file_count} files)")
        count += file_count

    # Delete times directory
    times_dir = participant_dir / "times"
    if times_dir.exists():
        file_count = len(list(times_dir.glob("*")))
        shutil.rmtree(times_dir)
        print(f"Deleted directory: {times_dir} ({file_count} files)")
        count += file_count

    return count


def cleanup_all(participant_name: str, data_dir: str = "data") -> int:
    """Delete entire participant directory."""
    participant_dir = Path(data_dir) / participant_name

    if not participant_dir.exists():
        print(f"No data found for participant: {participant_name}")
        return 0

    file_count = sum(1 for _ in participant_dir.rglob("*") if _.is_file())
    shutil.rmtree(participant_dir)
    print(f"Deleted entire directory: {participant_dir} ({file_count} files)")

    return file_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cleanup utility for eye-tracking experiment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete screenshots for one participant
  python cleanup.py --participants alice --screenshots

  # Delete visualizations for multiple participants
  python cleanup.py --participants alice bob charlie --visualizations

  # Delete all data for a participant
  python cleanup.py --participants alice --all

  # Clean processed data but keep raw data
  python cleanup.py --participants alice --processed
        """,
    )

    parser.add_argument(
        "--participants",
        "-p",
        nargs="+",
        required=True,
        help="Participant name(s) to clean data for",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )

    # Cleanup options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--screenshots",
        action="store_true",
        help="Delete only screenshots",
    )
    group.add_argument(
        "--visualizations",
        action="store_true",
        help="Delete only visualizations (heatmaps and scanpaths)",
    )
    group.add_argument(
        "--processed",
        action="store_true",
        help="Delete processed data (keeps raw gaze.csv and screenshots)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Delete entire participant directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted\n")

    total_deleted = 0

    for participant in args.participants:
        print(f"\n{'=' * 60}")
        print(f"Processing participant: {participant}")
        print(f"{'=' * 60}")

        if args.dry_run:
            # Just show what exists
            participant_dir = Path(args.data_dir) / participant
            if participant_dir.exists():
                file_count = sum(1 for _ in participant_dir.rglob("*") if _.is_file())
                print(f"Would process {file_count} files in {participant_dir}")
            else:
                print(f"No data found for {participant}")
            continue

        # Actual cleanup
        if args.screenshots:
            count = cleanup_screenshots(participant, args.data_dir)
        elif args.visualizations:
            count = cleanup_visualizations(participant, args.data_dir)
        elif args.processed:
            count = cleanup_processed_data(participant, args.data_dir)
        elif args.all:
            count = cleanup_all(participant, args.data_dir)

        print(f"\nDeleted {count} file(s) for {participant}")
        total_deleted += count

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("DRY RUN COMPLETE - No files were deleted")
    else:
        print(f"CLEANUP COMPLETE - Total files deleted: {total_deleted}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
