import winsound
from datetime import datetime, timezone


def make_beep() -> None:
    frequency = 1000  # Set Frequency To 2500 Hertz
    duration_beep = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration_beep)  # type: ignore
    return None


def get_current_time_iso8601() -> str:
    # Get the current time in UTC
    now = datetime.now(timezone.utc)
    print(now)
    # Format the time in ISO 8601 with milliseconds
    formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return formatted_time


def try_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def linear_interpolate(start: float, end: float, steps: int) -> list[float]:
    return [(start + (end - start) * i / steps) for i in range(1, steps)]
