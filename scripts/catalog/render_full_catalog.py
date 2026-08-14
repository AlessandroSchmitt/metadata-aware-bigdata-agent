from pathlib import Path

from metadata_agent.catalog import MetadataCatalog


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT / "data/catalog/metadata_catalog.sqlite"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts/benchmarks/full_catalog_context.txt"
)


def main():

    print("=" * 70)
    print("FULL METADATA CATALOG RENDERER")
    print("=" * 70)

    catalog = MetadataCatalog(
        DB_PATH
    )

    context = (
        catalog.render_full_catalog()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        context,
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("CONTEXT STATISTICS")
    print("=" * 70)

    print(
        f"Characters: {len(context):,}"
    )

    print(
        f"UTF-8 bytes: "
        f"{len(context.encode('utf-8')):,}"
    )

    print(
        f"Lines:      "
        f"{len(context.splitlines()):,}"
    )

    print(
        f"Words:      "
        f"{len(context.split()):,}"
    )

    print()
    print("=" * 70)
    print("REQUIRED KNOWLEDGE CHECK")
    print("=" * 70)

    required_fragments = [
        "DATASET yellow_taxi",
        "DATASET green_taxi",
        "DATASET taxi_zones",
        "DATASET weather_hourly",

        "tpep_pickup_datetime:"
        "timestamp_ntz",

        "lpep_pickup_datetime:"
        "timestamp_ntz",

        "semantic=pickup_datetime",

        "yellow_taxi.PULocationID "
        "= taxi_zones.LocationID",

        "green_taxi.PULocationID "
        "= taxi_zones.LocationID",

        "date_trunc('hour', "
        "yellow_taxi.tpep_pickup_datetime) "
        "= weather_hourly.weather_hour",

        "date_trunc('hour', "
        "green_taxi.lpep_pickup_datetime) "
        "= weather_hourly.weather_hour",

        "weather_hourly.has_precipitation "
        "= TRUE",

        "Includes trace precipitation",
    ]

    all_pass = True

    for fragment in required_fragments:

        present = (
            fragment in context
        )

        print(
            f"{'PASS' if present else 'FAIL'} "
            f"{fragment}"
        )

        if not present:
            all_pass = False

    print()
    print("=" * 70)
    print("CONTEXT PREVIEW")
    print("=" * 70)

    preview_lines = (
        context.splitlines()[:80]
    )

    print(
        "\n".join(preview_lines)
    )

    print()
    print("...")

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        OUTPUT_PATH.relative_to(ROOT)
    )

    if not all_pass:
        raise RuntimeError(
            "Required metadata missing "
            "from rendered context."
        )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "FULL CATALOG RENDERER: PASS"
    )


if __name__ == "__main__":
    main()
