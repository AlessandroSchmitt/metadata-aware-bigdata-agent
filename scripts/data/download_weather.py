import json
import shutil
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "data_sources.json"
SOURCE_KEY = "weather_central_park_2024"


def human_size(num_bytes):
    value = float(num_bytes)

    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TiB"


def main():
    print("=" * 70)
    print("NOAA WEATHER DOWNLOAD")
    print("=" * 70)

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        sources = json.load(handle)

    source = sources[SOURCE_KEY]

    url = source["url"]
    destination = ROOT / source["local_path"]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Dataset:      {source['dataset']}")
    print(f"Station:      {source['station_name']}")
    print(f"Station ID:   {source['station_id']}")
    print(f"Year:         {source['year']}")
    print(f"Destination:  {destination.relative_to(ROOT)}")

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"Status:       already exists "
            f"({human_size(destination.stat().st_size)})"
        )
        print("Skipping download.")
        return

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "metadata-aware-bigdata-agent/1.0"
        },
    )

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            print(
                f"HTTP status:  "
                f"{getattr(response, 'status', None)}"
            )

            length = response.headers.get(
                "Content-Length"
            )

            if length:
                print(
                    f"Remote size:  "
                    f"{human_size(int(length))}"
                )

            with temp_path.open("wb") as output:
                shutil.copyfileobj(
                    response,
                    output,
                    length=1024 * 1024,
                )

        temp_path.replace(destination)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    elapsed = time.perf_counter() - start

    print(
        f"Local size:   "
        f"{human_size(destination.stat().st_size)}"
    )
    print(f"Download:     {elapsed:.2f} s")
    print("Status:       OK")

    print()
    print("=" * 70)
    print("FIRST TWO PHYSICAL LINES")
    print("=" * 70)

    with destination.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        for _ in range(2):
            line = handle.readline()
            print(line[:2000].rstrip())

    print()
    print("=" * 70)
    print("WEATHER DOWNLOAD: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
