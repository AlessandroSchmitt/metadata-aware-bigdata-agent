import json
import math
import time
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


MODEL = "embeddinggemma"


def embed(texts):
    payload = {
        "model": MODEL,
        "input": texts,
        "keep_alive": "0",
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/embed",
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json"
        },
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    wall = (
        time.perf_counter()
        - start
    )

    return result, wall


print("=" * 70)
print("VECTOR RETRIEVAL STACK SANITY TEST")
print("=" * 70)


documents = [
    (
        "yellow taxi trip distance "
        "and pickup time"
    ),
    (
        "hourly weather precipitation "
        "rain and rainy conditions"
    ),
    (
        "taxi zone borough and "
        "location identifier"
    ),
]

query = (
    "average taxi distance "
    "during rainy weather"
)


print()
print("=" * 70)
print("GENERATING DOCUMENT EMBEDDINGS")
print("=" * 70)

doc_result, doc_wall = embed(
    documents
)

doc_vectors = doc_result[
    "embeddings"
]

print(
    f"Vectors generated: "
    f"{len(doc_vectors)}"
)

print(
    f"Vector dimension:  "
    f"{len(doc_vectors[0])}"
)

print(
    f"Wall time:         "
    f"{doc_wall:.2f} s"
)

print(
    f"Prompt tokens:     "
    f"{doc_result.get('prompt_eval_count')}"
)


print()
print("=" * 70)
print("VECTOR NORMS")
print("=" * 70)

for index, vector in enumerate(
    doc_vectors
):
    norm = math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )

    print(
        f"Document {index}: "
        f"{norm:.6f}"
    )


print()
print("=" * 70)
print("GENERATING QUERY EMBEDDING")
print("=" * 70)

query_result, query_wall = embed(
    query
)

query_vector = (
    query_result["embeddings"][0]
)

print(
    f"Dimension:          "
    f"{len(query_vector)}"
)

print(
    f"Wall time:          "
    f"{query_wall:.2f} s"
)


if (
    len(query_vector)
    != len(doc_vectors[0])
):
    raise RuntimeError(
        "Query/document vector "
        "dimension mismatch."
    )


print()
print("=" * 70)
print("QDRANT LOCAL MODE")
print("=" * 70)

client = QdrantClient(
    location=":memory:"
)

collection = (
    "metadata_sanity"
)

client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(
        size=len(doc_vectors[0]),
        distance=Distance.COSINE,
    ),
)

client.upsert(
    collection_name=collection,
    wait=True,
    points=[
        PointStruct(
            id=index,
            vector=vector,
            payload={
                "text": documents[index],
            },
        )
        for index, vector in enumerate(
            doc_vectors
        )
    ],
)

result = client.query_points(
    collection_name=collection,
    query=query_vector,
    limit=3,
    with_payload=True,
)

points = result.points

for rank, point in enumerate(
    points,
    start=1,
):
    print(
        f"{rank}. "
        f"score={point.score:.6f} "
        f"text={point.payload['text']}"
    )


if len(points) != 3:
    raise RuntimeError(
        "Unexpected number of "
        "Qdrant results."
    )


print()
print("=" * 70)
print("EXPECTED SEMANTIC CHECK")
print("=" * 70)

top_text = (
    points[0].payload["text"]
)

print(
    f"Top result: {top_text}"
)

if (
    "distance" not in top_text.lower()
    and "weather" not in top_text.lower()
):
    raise RuntimeError(
        "Semantic retrieval sanity "
        "check failed."
    )

print(
    "Semantic retrieval: PASS"
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "VECTOR RETRIEVAL STACK: PASS"
)
