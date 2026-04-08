import argparse
import csv
import math

from utils import linear_interpolate, try_float


def process_gaze_data(input_file: str, output_file: str, width: int, height: int) -> None:
    rows = []

    """
    Read the data obtained by the generate.py
    clean the data, avergae left and right and int values
    """
    with open(input_file) as infile, open(output_file, mode="w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["time_seconds", "current_time", "x", "y"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            left_x = try_float(row["left_x"])
            left_y = try_float(row["left_y"])
            right_x = try_float(row["right_x"])
            right_y = try_float(row["right_y"])

            if math.isnan(left_x) and not math.isnan(right_x):
                left_x = right_x
            if math.isnan(left_y) and not math.isnan(right_y):
                left_y = right_y
            if math.isnan(right_x) and not math.isnan(left_x):
                right_x = left_x
            if math.isnan(right_y) and not math.isnan(left_y):
                right_y = left_y

            avg_x = int((left_x + right_x) / 2 * width) if not math.isnan(left_x) else float("nan")
            avg_y = int((left_y + right_y) / 2 * height) if not math.isnan(left_y) else float("nan")

            rows.append(
                {
                    "current_time": row["current_time"],
                    "x": avg_x,
                    "y": avg_y,
                    "time_seconds": row["time_seconds"],
                }
            )

    process_nans(rows, output_file)


def process_nans(rows: list[dict[str, str | int | float]], output_file: str) -> None:
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

    if problems and problems[0][0] == -1:
        start = problems.pop(0)
        min_time = rows[start[-1]]["time_seconds"]

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
        fieldnames = ["x", "y", "time_seconds", "current_time"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(rows)):
            if index < start[1]:
                continue
            row = rows[index]
            row["time_seconds"] = float(row["time_seconds"]) - float(min_time)
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
