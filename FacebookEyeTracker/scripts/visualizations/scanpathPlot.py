import argparse
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def main(args: Any) -> None:
    # Load data
    data = pd.read_csv(args.gaze_csv)
    image = plt.imread(args.image_path)

    parts = args.image_path.split("/")
    name_post_id_part = parts[-1]  # Assumes format is '.../screenshots/{name}_screenshot_{post_id}.png'
    name_part, _ = name_post_id_part.split("_screenshot_")
    post_id_part = _.split(".")[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    # Initialize plot
    ax.set_xlim([0, 1920])
    ax.set_ylim([1080, 0])

    # Setup variables for plotting
    radius = 450
    last_x = data["x"].iloc[0]
    last_y = data["y"].iloc[0]
    accumulated_time = 0
    start_time = data["time_seconds"].iloc[0]
    plot_x = []
    plot_y = []
    times = []
    last_time = max(data["time_seconds"])

    # Loop through data
    last_take = False
    for i in range(1, len(data)):
        if 0 < data["x"].iloc[i] < 1920 and 0 < data["y"].iloc[i] < 1020:
            pass
        else:
            continue
        dist = euclidean_distance(last_x, last_y, data["x"].iloc[i], data["y"].iloc[i])
        if last_time - data["time_seconds"].iloc[i] < 0.8 and not last_take:
            plot_x.append(last_x)
            plot_y.append(last_y)
            times.append(accumulated_time if accumulated_time > 0 else 1)
            last_x = data["x"].iloc[i]
            last_y = data["y"].iloc[i]
            start_time = data["time_seconds"].iloc[i]
            accumulated_time = 0
            last_take = True
            continue

        if dist <= radius:
            accumulated_time += data["time_seconds"].iloc[i] - start_time
        else:
            plot_x.append(last_x)
            plot_y.append(last_y)
            times.append(accumulated_time if accumulated_time > 0 else 1)
            last_x = data["x"].iloc[i]
            last_y = data["y"].iloc[i]
            start_time = data["time_seconds"].iloc[i]
            accumulated_time = 0

    plot_x.append(last_x)
    plot_y.append(last_y)
    times.append(accumulated_time if accumulated_time > 0 else 1)

    # Normalize times for circle sizes
    max_time = max(times) if max(times) > 0 else 1
    sizes = [math.log(time) / math.log(max_time) * 500 for time in times]

    # Plot points and lines
    ax.plot(plot_x, plot_y, marker="", linestyle="-", color="red")
    ax.scatter(plot_x, plot_y, s=sizes, c="red", alpha=0.5, edgecolors="black")

    # Annotate points
    for i, (px, py, _time) in enumerate(zip(plot_x, plot_y, times, strict=False)):
        ax.annotate(
            i,
            (px, py),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            va="center",
            color="black",
        )

    ax.set_title("Gaze Duration and Path Visualization")

    # Remove numbers (tick labels) from the axes
    ax.set_xticks([])
    ax.set_yticks([])

    if not os.path.isfile("scans.csv"):
        with open("scans.csv", "w") as f:
            f.write("userName,postID,length_plot_x\n")
    with open("scans.csv", "a") as f:
        f.write(f"{name_part},{post_id_part},{len(plot_x)}\n")
    print(name_part, post_id_part)

    plt.savefig(args.output_scanpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gaze duration and path visualization.")
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        required=True,
        help="Path to the background image.",
    )
    parser.add_argument(
        "-g",
        "--gaze_csv",
        type=str,
        required=True,
        help="Path to the gaze data CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output_scanpath",
        type=str,
        required=True,
        help="Output path for the generated scanpath image.",
    )

    args = parser.parse_args()
    main(args)
