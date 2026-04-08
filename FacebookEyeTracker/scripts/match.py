import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd
import requests


def download_and_filter_json(user_name: str, root: str) -> None:
    path = f"{root}/times/" + user_name + "_posts_times.json"
    url = "http://localhost:3001/download/answersPostsAndSurvey"

    if os.path.isfile(path):
        return

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Verifica si hubo algún error en la solicitud

        # Decodifica el contenido JSON
        data = response.json()

        # Procesa el contenido JSON filtrando por nombre de usuario
        filtered_data = [entry for entry in data if entry.get("userName").lower() == (user_name).lower()]
        # Guarda el contenido filtrado en un archivo JSON
        with open(path, "w", encoding="utf-8") as file:
            json.dump(filtered_data, file, ensure_ascii=False, indent=4)

        print(f"Filtered file saved as {user_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


def load_gaze_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["current_time"] = pd.to_datetime(df["current_time"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    return df


def load_json_data(file_path: str) -> list[Any]:
    with open(file_path) as file:
        json_data = cast(list[Any], json.load(file))
    return json_data


def process_gaze_data(df: pd.DataFrame, json_data: list[Any]) -> pd.DataFrame:
    # Step 1: initial date and first time
    initial_date = datetime.strptime(json_data[0]["initialDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
    last_time_seconds = df[df["current_time"] - initial_date < timedelta(seconds=0)]
    if last_time_seconds.empty:
        less = df["current_time"].loc[0]
        last_time_seconds = -abs(less - initial_date).total_seconds()
    else:
        last_time_seconds = last_time_seconds["time_seconds"].iloc[-1]

    # Step 2: Remove all rows where `time_seconds` is less than 0
    df["time_seconds"] = df["time_seconds"] - last_time_seconds
    df = df[df["time_seconds"] >= 0].reset_index(drop=True)
    df["postID"] = None

    # Step 3: Assign postID based on PostStartTime and PostEndTime
    for obj in json_data:
        post_start_time = obj["PostStartTime"]
        post_end_time = obj["PostEndTime"]
        post_id = obj["postID"]

        df.loc[
            (df["time_seconds"] >= post_start_time) & (df["time_seconds"] <= post_end_time),
            "postID",
        ] = post_id

    # Step 4: Remove all rows where `postID` is None
    df = df[df["postID"].notna()].reset_index(drop=True)
    print("dataframe filtered", df)
    return df


def process_screenshots(screenshots_folder: str, json_data: list[Any]) -> pd.DataFrame:
    # Step 1: Get the list of screenshot files
    screenshot_files = os.listdir(screenshots_folder)
    screenshot_files = [file for file in screenshot_files if file.endswith(".png")]
    screenshot_assignments = []

    # Step 2: Assign postID based on screenshot timestamp
    for file in screenshot_files:
        timestamp_str = file.replace("screenshot_", "").replace(".png", "").replace("_", ":")
        screenshot_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        assigned_post_id = None
        for obj in json_data:
            initial_date = datetime.strptime(obj["initialDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
            post_start_time = initial_date + timedelta(seconds=obj["PostStartTime"])
            post_end_time = initial_date + timedelta(seconds=obj["PostEndTime"])

            if post_start_time <= screenshot_time <= post_end_time:
                assigned_post_id = obj["postID"]
                break

        screenshot_assignments.append(
            {
                "filename": file,
                "screenshot_time": screenshot_time,
                "postID": assigned_post_id,
            }
        )

    # Step 3: Create a DataFrame with the assignments
    screenshot_df = pd.DataFrame(screenshot_assignments)
    screenshot_df = screenshot_df[screenshot_df["postID"].notna()].reset_index(drop=True)
    screenshot_df = screenshot_df.drop_duplicates(subset="postID", keep="first")
    screenshot_df.sort_values(by="screenshot_time", inplace=True)
    return screenshot_df


def assign_screenshot_filenames(df: pd.DataFrame, screenshot_df: pd.DataFrame) -> pd.DataFrame:
    screenshot_df["postID"] = screenshot_df["postID"].astype(int)
    post_id_to_filename = screenshot_df.set_index("postID")["filename"].to_dict()
    df["screenshot_filename"] = df["postID"].map(post_id_to_filename)
    return df


def save_split_files(df: pd.DataFrame, output_folder: str, name: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    unique_post_ids = df["postID"].unique()

    for post_id in unique_post_ids:
        df_filtered = df[df["postID"] == post_id]
        filename = f"{name}_gaze_{post_id}.csv"
        df_filtered["x"] = df_filtered["x"].astype(int)
        df_filtered["y"] = df_filtered["y"].astype(int)
        df_filtered.to_csv(os.path.join(output_folder, filename), index=False)

    print(f"Archivos CSV creados en la carpeta {output_folder}")


def collect_screenshots(unique_post_ids: Any, name: str, root: str) -> None:
    for post_id in unique_post_ids:
        df_file = pd.read_csv(root + f"gaze_posts/{name}_gaze_{post_id}.csv")
        image_screenshot = root + f"screenshots/{df_file['screenshot_filename'].iloc[0]}"

        new_screenshot_filename = f"{name}_screenshot_{post_id}.png"
        new_screenshot_path = root + f"screenshots/{new_screenshot_filename}"
        print(os.path.exists(image_screenshot), image_screenshot)
        if os.path.exists(image_screenshot):
            os.rename(image_screenshot, new_screenshot_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help="total seconds to collect data")
    args = vars(parser.parse_args())
    name = args["name"]

    root = f"data/{name}/"
    input_file = root + "gaze_clean.csv"
    json_file = root + f"times/{name}_posts_times.json"
    screenshot_folder = root + "screenshots/"

    download_and_filter_json(name, root)

    df_initial = load_gaze_data(input_file)
    json_data = load_json_data(json_file)
    df_processed = process_gaze_data(df_initial, json_data)
    screenshot_df = process_screenshots(screenshot_folder, json_data)
    df = assign_screenshot_filenames(df_processed, screenshot_df)
    save_split_files(df, root + "gaze_posts/", name)
    unique_post_ids = df["postID"].unique()
    collect_screenshots(unique_post_ids, name=name, root=root)


if __name__ == "__main__":
    main()
