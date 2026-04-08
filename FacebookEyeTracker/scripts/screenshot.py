import os
import sys
import time

import pyautogui
from utils import get_current_time_iso8601


def take_screenshots(name: str, duration: int, slots: int = 3) -> None:
    time.sleep(5)
    print(duration, slots, int(duration / slots))
    screenshot(name, 0)
    for _ in range(1, int(duration / slots) + 1):
        time.sleep(slots)
        screenshot(name, _)


def screenshot(name: str, screen_number: int) -> None:
    target_folder = f"data/{name}/screenshots"

    current_time = get_current_time_iso8601(option=2)
    screenshot_path = os.path.join(target_folder, f"screenshot_{current_time}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}", screen_number)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        duration = int(sys.argv[2])
        take_screenshots(name, duration)
    else:
        print("Please provide a folder name as an argument.")
