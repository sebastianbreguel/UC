import argparse
import csv
import math

import numpy as np
from utils import linear_interpolate, try_float


def process_gaze_data(input_file: str, output_file: str, width: int, height: int) -> None:
    rows = []

    """
    Read the data obtained by the generate.py
    clean the data, avergae left and right and int values
    """
    with open(input_file) as infile, open(output_file, mode="w", newline="") as outfile:
        reader = csv.DictReader(infile)
        # fieldnames = ['time_seconds',"current_time", 'x', 'y']
        fieldnames = ["time_seconds", "x", "y"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        partial = []

        for row in reader:
            left_x = try_float(row["left_x"])
            left_y = try_float(row["left_y"])
            right_x = try_float(row["right_x"])
            right_y = try_float(row["right_y"])

            if math.isnan(left_x) or math.isnan(left_y) or math.isnan(right_x) or math.isnan(right_y):
                continue

            partial.append(
                [
                    left_x + right_x,
                    left_y + right_y,
                    abs(left_x - right_x),
                    abs(left_y - right_y),
                ]
            )

        partial = np.array(partial)

        # Calculate mean and standard deviation for abs_diff_x and abs_diff_y

        std_mean_x = np.std(partial[:, 0]) / 2
        std_mean_y = np.std(partial[:, 1]) / 2
        mean_abs_diff_x = np.mean(partial[:, 2])
        std_abs_diff_x = np.std(partial[:, 2])
        mean_abs_diff_y = np.mean(partial[:, 3])
        std_abs_diff_y = np.std(partial[:, 3])
        print(std_abs_diff_y, "hoal")

        print(f"std of std_mean_x: {std_mean_x}")
        print(f"std of std_mean_y: {std_mean_y}")

        print(f"Mean of abs_diff_x: {mean_abs_diff_x}")
        print(f"Standard Deviation of abs_diff_x: {std_abs_diff_x}")
        print(f"Mean of abs_diff_y: {mean_abs_diff_y}")
        print(f"Standard Deviation of abs_diff_y: {std_abs_diff_y}")

        # Rewind the reader to process the file again
        infile.seek(0)

        reader = csv.DictReader(infile)

        for row in reader:
            left_x = try_float(row["left_x"])
            left_y = try_float(row["left_y"])
            right_x = try_float(row["right_x"])
            right_y = try_float(row["right_y"])

            if math.isnan(left_x) and not math.isnan(right_x):
                left_x = np.random.normal(right_x - mean_abs_diff_x, std_abs_diff_x)
            if math.isnan(left_y) and not math.isnan(right_y):
                left_y = np.random.normal(right_y, std_abs_diff_y)
            if math.isnan(right_x) and not math.isnan(left_x):
                right_x = np.random.normal(left_x + mean_abs_diff_x, std_abs_diff_x)
            if math.isnan(right_y) and not math.isnan(left_y):
                right_y = np.random.normal(left_y, std_abs_diff_y)

            avg_x = int((left_x + right_x) / 2 * width) if not math.isnan(left_x) else right_y
            avg_y = int((left_y + right_y) / 2 * height) if not math.isnan(left_y) else right_y

            rows.append(
                {
                    # 'current_time' : row['current_time'],
                    "x": avg_x,
                    "y": avg_y,
                    "time_seconds": row["time_seconds"],
                }
            )

    process_nans(rows, output_file, std_abs_diff_x, std_abs_diff_y)


def process_nans(rows: list[dict[str, str | int | float]], output_file: str, std_abs_diff_x: float, std_abs_diff_y: float) -> None:
    problems = []
    is_in_nans = False
    sub = []

    for index in range(len(rows)):
        row = rows[index]
        x = try_float(row["x"])
        y = try_float(row["y"])

        if math.isnan(x) and not is_in_nans:
            sub.append(index - 1)
            is_in_nans = True

        elif is_in_nans and not math.isnan(x):
            sub.append(index)
            is_in_nans = False
            problems.append(sub)
            sub = []

    if math.isnan(x):
        rows = rows[: sub[0]]

    if problems and problems[0][0] == -1:
        start = problems.pop(0)

    for i in range(len(problems)):
        before = problems[i][0]
        after = problems[i][1]
        reader_before = rows[before]
        reader_after = rows[after]
        distance = after - before

        x = linear_interpolate(try_float(reader_before["x"]), try_float(reader_after["x"]), distance)
        y = linear_interpolate(try_float(reader_before["y"]), try_float(reader_after["y"]), distance)

        for j in range(1, distance):
            row = rows[before + j]
            row["x"] = int(x[j - 1])
            row["y"] = int(y[j - 1])

    with open(output_file, mode="w", newline="") as outfile:
        # fieldnames = ['time_seconds','current_time', 'x', 'y']
        fieldnames = ["x", "y", "time_seconds"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(rows)):
            if index < start[1]:
                continue
            row = rows[index]
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gaze data")
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("output_file", type=str, help="Path to the output file")
    parser.add_argument("width", type=int, help="Screen width")
    parser.add_argument("height", type=int, help="Screen height")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    width = args.width
    height = args.height

    process_gaze_data(input_file, output_file, width, height)
