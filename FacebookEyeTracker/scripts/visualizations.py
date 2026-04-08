import argparse
import subprocess
import sys
from pathlib import Path


def extract_post_id(filename: str) -> int:
    """Extract the post ID from a gaze CSV filename like 'name_gaze_42.csv'."""
    return int(filename.split("_")[-1].replace(".csv", ""))


def create_visualizations(post_ids: list[int], name: str, root: Path, width: int = 1920, height: int = 1080) -> None:
    """Generate heatmap and scanpath visualizations for each post."""
    for post_id in post_ids:
        input_csv = str(root / f"gaze_posts/{name}_gaze_{post_id}.csv")
        screenshot_path = str(root / f"screenshots/{name}_screenshot_{post_id}.png")
        heatmap_file = str(root / f"heatmaps/{name}_heatmap_{post_id}.png")
        scanpath_file = str(root / f"scanpath/{name}_scanpath_{post_id}.png")

        subprocess.run(
            [
                sys.executable,
                "scripts/visualizations/gazeHeatplot.py",
                input_csv,
                str(width),
                str(height),
                "-b",
                screenshot_path,
                "-o",
                heatmap_file,
            ],
            check=False,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/visualizations/scanpathPlot.py",
                "-g",
                input_csv,
                "-i",
                screenshot_path,
                "-o",
                scanpath_file,
            ],
            check=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heatmap and scanpath visualizations")
    parser.add_argument("name", type=str, help="Participant name")
    args = parser.parse_args()

    root = Path("data") / args.name
    gaze_posts_dir = root / "gaze_posts"

    post_ids = [extract_post_id(f.name) for f in gaze_posts_dir.glob("*.csv")]
    create_visualizations(post_ids, args.name, root)


if __name__ == "__main__":
    main()
