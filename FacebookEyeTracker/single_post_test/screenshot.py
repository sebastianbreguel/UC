import os
import sys
import time

import pyautogui


def screenshot(name: str, duration: int) -> None:
    time.sleep(5)
    target_folder = "images"

    screenshot_path = os.path.join(target_folder, f"{name}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)

    print(f"Screenshot saved to {screenshot_path}")
    time.sleep(duration - 5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        duration = int(sys.argv[2])
        screenshot(name, duration)
    else:
        print("Please provide a folder name as an argument.")
