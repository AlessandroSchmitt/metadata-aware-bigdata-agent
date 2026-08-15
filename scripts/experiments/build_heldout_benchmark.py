import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    ROOT / "config/benchmark_queries.json"
)

OUTPUT_PATH = (
    ROOT / "config/benchmark_heldout_queries.json"
)


# Two unseen linguistic formulations for each frozen
# development intent.
#
# Gold SQL, expected output columns and required metadata
# are copied verbatim from the canonical benchmark.
PARAPHRASES = {
    "Q01": [
        (
            "Across all Yellow Taxi rides in January 2024, "
            "what was the mean trip distance? Return exactly "
            "one column named average_trip_distance."
        ),
        (
            "Find the average distance travelled per Yellow "
            "Taxi trip for January 2024. The result must "
            "contain exactly one column called "
            "average_trip_distance."
        ),
    ],

    "Q02": [
        (
            "Count the Green Taxi trips in January 2024 whose "
            "trip distance exceeded 10 miles. Return exactly "
            "one column named trip_count."
        ),
        (
            "How many January 2024 Green Taxi rides were "
            "longer than 10 miles? Output exactly one column "
            "called trip_count."
        ),
    ],

    "Q03": [
        (
            "For January 2024 Yellow Taxi rides, identify the "
            "pickup taxi zone with the greatest number of "
            "trips. Return only the top result with exactly "
            "two columns named zone_name and trip_count."
        ),
        (
            "Where did the largest number of Yellow Taxi "
            "pickups occur by taxi zone in January 2024? "
            "Return just the leading zone using exactly the "
            "columns zone_name and trip_count."
        ),
    ],

    "Q04": [
        (
            "In January 2024, which borough had the largest "
            "number of Green Taxi trip dropoffs? Return only "
            "the top borough with exactly two columns named "
            "borough and trip_count."
        ),
        (
            "Find the borough that was the most common Green "
            "Taxi dropoff destination during January 2024. "
            "Return one top result with exactly the columns "
            "borough and trip_count."
        ),
    ],

    "Q05": [
        (
            "What was the mean trip distance for each taxi "
            "service, Yellow and Green, in January 2024? "
            "Return one row per service with exactly two "
            "columns named taxi_type and "
            "average_trip_distance."
        ),
        (
            "Compare Yellow Taxi with Green Taxi by their "
            "average trip distance for January 2024. Produce "
            "one row for each service and exactly the columns "
            "taxi_type and average_trip_distance."
        ),
    ],

    "Q06": [
        (
            "For Yellow Taxi rides that began in rainy hours "
            "during January 2024, what was the average trip "
            "distance? Return exactly one column named "
            "average_trip_distance."
        ),
        (
            "Calculate the mean distance of January 2024 "
            "Yellow Taxi trips whose pickup occurred during "
            "an hour with precipitation. Output exactly one "
            "column called average_trip_distance."
        ),
    ],

    "Q07": [
        (
            "Count the Green Taxi trips in January 2024 whose "
            "pickup took place during a dry hour. Return "
            "exactly one column named trip_count."
        ),
        (
            "How many Green Taxi rides started in hours "
            "classified as dry during January 2024? Output "
            "exactly one column called trip_count."
        ),
    ],

    "Q08": [
        (
            "Among Yellow Taxi trips picked up during rainy "
            "hours in January 2024, which pickup borough had "
            "the greatest trip count? Return only the top "
            "borough with exactly two columns named borough "
            "and trip_count."
        ),
        (
            "Find the pickup borough with the most January "
            "2024 Yellow Taxi rides occurring during hours "
            "with precipitation. Return just the leading "
            "borough using exactly the columns borough and "
            "trip_count."
        ),
    ],

    "Q09": [
        (
            "What was the mean air temperature across rainy "
            "hours in January 2024? Return exactly one column "
            "named average_temperature_f."
        ),
        (
            "Calculate the average temperature for January "
            "2024 hours in which precipitation occurred. "
            "Output exactly one column called "
            "average_temperature_f."
        ),
    ],

    "Q10": [
        (
            "For Yellow Taxi pickups occurring during rainy "
            "hours in January 2024, which hour of the day had "
            "the highest pickup count? Return only the top "
            "hour with exactly two columns named hour_of_day "
            "and trip_count."
        ),
        (
            "During January 2024 rainy periods, at what clock "
            "hour did the largest number of Yellow Taxi "
            "pickups occur? Return just the leading hour "
            "using exactly the columns hour_of_day and "
            "trip_count."
        ),
    ],
}


source = json.loads(
    SOURCE_PATH.read_text(
        encoding="utf-8"
    )
)

source_queries = {
    query["id"]: query
    for query in source["queries"]
}


if set(PARAPHRASES) != set(source_queries):
    raise RuntimeError(
        "Paraphrase intent IDs do not exactly match "
        "the canonical benchmark IDs."
    )


heldout = []


for source_id in sorted(source_queries):
    canonical = source_queries[source_id]

    variants = PARAPHRASES[source_id]

    if len(variants) != 2:
        raise RuntimeError(
            f"{source_id} must have exactly "
            "two held-out paraphrases."
        )

    for number, question in enumerate(
        variants,
        start=1,
    ):
        heldout.append(
            {
                "id": f"{source_id}_H{number}",
                "source_query_id": source_id,
                "variant": number,
                "category": canonical["category"],
                "question": question,
                "expected_columns": (
                    canonical["expected_columns"]
                ),
                "gold_sql": (
                    canonical["gold_sql"]
                ),
                "required_metadata": (
                    canonical["required_metadata"]
                ),
            }
        )


manifest = {
    "benchmark": {
        "name": (
            "NYC Metadata-Aware Text-to-Spark-SQL "
            "Held-Out Benchmark"
        ),
        "version": "1.0",
        "period": "January 2024",
        "source_benchmark": (
            "NYC Metadata-Aware Text-to-Spark-SQL "
            "Benchmark v1.0"
        ),
        "development_freeze": (
            "development-freeze-v1"
        ),
        "description": (
            "Held-out linguistic generalization set "
            "containing two previously unevaluated "
            "paraphrases for each of the ten frozen "
            "development intents."
        ),
    },
    "queries": heldout,
}


OUTPUT_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 70)
print("HELD-OUT BENCHMARK BUILD")
print("=" * 70)

print(
    f"Canonical intents: "
    f"{len(source_queries)}"
)

print(
    f"Held-out questions: "
    f"{len(heldout)}"
)

print(
    "Variants per intent: 2"
)

print()
print(
    OUTPUT_PATH.relative_to(ROOT)
)

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "HELD-OUT BENCHMARK BUILD: PASS"
)
