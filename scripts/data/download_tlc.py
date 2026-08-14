import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "data_sources.json"


def human_size(num_bytes):
    value = float(num_bytes)

    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TiB"


def download_file(name, info):
    url = info["url"]
    destination = ROOT / info["local_path"]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Dataset:      {info['dataset']}")
    print(f"Format:       {info['format']}")
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
        }
    )

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            status = getattr(
                response,
                "status",
                None
            )

            content_length = response.headers.get(
                "Content-Length"
            )

            print(f"HTTP status:  {status}")

            if content_length:
                print(
                    f"Remote size:  "
                    f"{human_size(int(content_length))}"
                )

            with temp_path.open("wb") as output:
                shutil.copyfileobj(
                    response,
                    output,
                    length=1024 * 1024
                )

        temp_path.replace(destination)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    elapsed = time.perf_counter() - start
    size = destination.stat().st_size

    print(f"Local size:   {human_size(size)}")
    print(f"Download:     {elapsed:.2f} s")
    print("Status:       OK")


def main():
    print("=" * 70)
    print("NYC TLC DATA DOWNLOAD")
    print("=" * 70)

    if not CONFIG_PATH.exists():
        print(
            f"ERROR: configuration file not found: "
            f"{CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as handle:
        sources = json.load(handle)

    for name, info in sources.items():
        download_file(
            name,
            info
        )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print()

    for name, info in sources.items():
        path = ROOT / info["local_path"]

        print(
            f"{name:25s} "
            f"{human_size(path.stat().st_size)}"
        )


if __name__ == "__main__":
    main()
