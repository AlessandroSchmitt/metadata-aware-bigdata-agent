import time
from pathlib import Path

from qdrant_client import QdrantClient

from metadata_agent.embeddings import (
    OllamaEmbedder,
)


ROOT = Path(__file__).resolve().parents[2]

QDRANT_PATH = (
    ROOT / ".qdrant/metadata_catalog"
)

COLLECTION = "metadata_catalog"

QUESTION = (
    "What was the average trip distance "
    "for Yellow Taxi trips during rainy "
    "hours in January 2024? "
    "Return exactly one column named "
    "average_trip_distance."
)

GOLD_REQUIRED = {
    "dataset:yellow_taxi",
    "dataset:weather_hourly",
    "column:yellow_taxi.trip_distance",
    (
        "column:yellow_taxi."
        "tpep_pickup_datetime"
    ),
    "column:weather_hourly.weather_hour",
    (
        "column:weather_hourly."
        "has_precipitation"
    ),
    (
        "relationship:"
        "yellow_pickup_weather_hour"
    ),
    "rule:rainy_hour",
}


def point_key(point):
    payload = point.payload

    return (
        f"{payload['entity_type']}:"
        f"{payload['entity_key']}"
    )


print("=" * 70)
print("REAL METADATA SEMANTIC RETRIEVAL")
print("=" * 70)

print()
print(
    f"Question: {QUESTION}"
)


embedder = OllamaEmbedder(
    model="embeddinggemma"
)

embedding_result = embedder.embed(
    QUESTION,
    keep_alive="10m",
)

query_vector = (
    embedding_result[
        "embeddings"
    ][0]
)


print()
print("=" * 70)
print("QUERY EMBEDDING")
print("=" * 70)

print(
    f"Dimension:       "
    f"{len(query_vector)}"
)

print(
    f"Prompt tokens:   "
    f"{embedding_result['prompt_eval_count']}"
)

print(
    f"Wall time:       "
    f"{embedding_result['wall_time_seconds']:.3f} s"
)


client = QdrantClient(
    path=str(QDRANT_PATH)
)


search_start = time.perf_counter()

response = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=20,
    with_payload=True,
)

search_time = (
    time.perf_counter()
    - search_start
)

points = response.points


print()
print("=" * 70)
print("TOP 20 RETRIEVAL RESULTS")
print("=" * 70)

for rank, point in enumerate(
    points,
    start=1,
):
    payload = point.payload

    print(
        f"{rank:2d}. "
        f"score={point.score:.6f} "
        f"{payload['entity_type']:12s} "
        f"{payload['entity_key']}"
    )


print()
print(
    f"Qdrant search time: "
    f"{search_time:.6f} s"
)


print()
print("=" * 70)
print("GOLD METADATA")
print("=" * 70)

for key in sorted(GOLD_REQUIRED):
    print(
        f"  {key}"
    )


print()
print("=" * 70)
print("RECALL AT K")
print("=" * 70)

for k in [
    5,
    8,
    10,
    12,
    15,
    20,
]:
    retrieved = {
        point_key(point)
        for point in points[:k]
    }

    hits = (
        GOLD_REQUIRED
        & retrieved
    )

    recall = (
        len(hits)
        / len(GOLD_REQUIRED)
    )

    print(
        f"Recall@{k:2d}: "
        f"{len(hits)}/"
        f"{len(GOLD_REQUIRED)} "
        f"= {recall:.3f}"
    )


retrieved_20 = {
    point_key(point)
    for point in points
}

missing = (
    GOLD_REQUIRED
    - retrieved_20
)


print()
print("=" * 70)
print("MISSING GOLD ITEMS AT K=20")
print("=" * 70)

if missing:
    for key in sorted(missing):
        print(
            f"  {key}"
        )
else:
    print(
        "None — Recall@20 = 1.0"
    )


client.close()


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "REAL METADATA RETRIEVAL TEST: PASS"
)
