import json
import shutil
import time
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from metadata_agent.catalog import MetadataCatalog
from metadata_agent.embeddings import OllamaEmbedder
from metadata_agent.metadata_documents import (
    MetadataDocumentBuilder,
)


ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = (
    ROOT
    / "data/catalog/metadata_catalog.sqlite"
)

QDRANT_PATH = (
    ROOT
    / ".qdrant/metadata_catalog"
)

REPORT_PATH = (
    ROOT
    / "artifacts/benchmarks/"
      "metadata_vector_index.json"
)

COLLECTION = "metadata_catalog"
MODEL = "embeddinggemma"
BATCH_SIZE = 20


print("=" * 70)
print("BUILD METADATA VECTOR INDEX")
print("=" * 70)


catalog = MetadataCatalog(
    CATALOG_PATH
)

documents = MetadataDocumentBuilder(
    catalog
).build()


print()
print("=" * 70)
print("METADATA DOCUMENTS")
print("=" * 70)

print(
    f"Total documents: {len(documents)}"
)

counts = Counter(
    document.entity_type
    for document in documents
)

for entity_type in sorted(counts):
    print(
        f"{entity_type:15s} "
        f"{counts[entity_type]:3d}"
    )


if not documents:
    raise RuntimeError(
        "No metadata documents were generated."
    )


# ---------------------------------------------------------
# Generate embeddings in batches
# ---------------------------------------------------------

embedder = OllamaEmbedder(
    model=MODEL
)

all_vectors = []

total_prompt_tokens = 0
total_embedding_wall = 0.0
total_embedding_api_time = 0.0


print()
print("=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)

for start_index in range(
    0,
    len(documents),
    BATCH_SIZE,
):
    batch = documents[
        start_index:
        start_index + BATCH_SIZE
    ]

    texts = [
        document.text
        for document in batch
    ]

    result = embedder.embed(
        texts,
        keep_alive="10m",
    )

    all_vectors.extend(
        result["embeddings"]
    )

    total_prompt_tokens += (
        result["prompt_eval_count"]
    )

    total_embedding_wall += (
        result["wall_time_seconds"]
    )

    total_embedding_api_time += (
        result["total_duration_seconds"]
    )

    end_index = (
        start_index + len(batch)
    )

    print(
        f"Embedded documents "
        f"{start_index + 1:3d}-"
        f"{end_index:3d} "
        f"| tokens="
        f"{result['prompt_eval_count']:5d} "
        f"| wall="
        f"{result['wall_time_seconds']:.2f}s"
    )


if len(all_vectors) != len(documents):
    raise RuntimeError(
        "Embedding/document count mismatch."
    )

vector_size = len(
    all_vectors[0]
)

if not all(
    len(vector) == vector_size
    for vector in all_vectors
):
    raise RuntimeError(
        "Inconsistent vector dimensions."
    )


print()
print(
    f"Vector dimension:       "
    f"{vector_size}"
)

print(
    f"Total embedding tokens: "
    f"{total_prompt_tokens}"
)

print(
    f"Embedding wall sum:     "
    f"{total_embedding_wall:.2f} s"
)


# ---------------------------------------------------------
# Build persistent Qdrant index
# ---------------------------------------------------------

print()
print("=" * 70)
print("BUILDING QDRANT INDEX")
print("=" * 70)

if QDRANT_PATH.exists():
    shutil.rmtree(
        QDRANT_PATH
    )

QDRANT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

client = QdrantClient(
    path=str(QDRANT_PATH)
)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE,
    ),
)


points = []

for point_id, (
    document,
    vector,
) in enumerate(
    zip(
        documents,
        all_vectors,
    ),
    start=1,
):
    payload = {
        "entity_type": (
            document.entity_type
        ),
        "entity_key": (
            document.entity_key
        ),
        "dataset": (
            document.dataset
        ),
        "text": document.text,
    }

    points.append(
        PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
    )


index_start = time.perf_counter()

client.upsert(
    collection_name=COLLECTION,
    points=points,
    wait=True,
)

index_time = (
    time.perf_counter()
    - index_start
)

stored_count = client.count(
    collection_name=COLLECTION,
    exact=True,
).count


print(
    f"Stored points:          "
    f"{stored_count}"
)

print(
    f"Index write time:       "
    f"{index_time:.3f} s"
)

print(
    f"Storage path:           "
    f"{QDRANT_PATH.relative_to(ROOT)}"
)


if stored_count != len(documents):
    raise RuntimeError(
        "Qdrant point count mismatch."
    )


client.close()


# ---------------------------------------------------------
# Re-open index: persistence validation
# ---------------------------------------------------------

print()
print("=" * 70)
print("PERSISTENCE VALIDATION")
print("=" * 70)

reopened = QdrantClient(
    path=str(QDRANT_PATH)
)

reopened_count = reopened.count(
    collection_name=COLLECTION,
    exact=True,
).count

print(
    f"Points after reopen:    "
    f"{reopened_count}"
)

reopened.close()


if reopened_count != len(documents):
    raise RuntimeError(
        "Persistent index validation failed."
    )


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

report = {
    "collection": COLLECTION,
    "embedding_model": MODEL,
    "vector_dimension": vector_size,
    "document_count": len(documents),
    "documents_by_type": dict(
        sorted(counts.items())
    ),
    "embedding": {
        "batch_size": BATCH_SIZE,
        "prompt_eval_count": (
            total_prompt_tokens
        ),
        "wall_time_seconds_sum": (
            total_embedding_wall
        ),
        "api_duration_seconds_sum": (
            total_embedding_api_time
        ),
    },
    "qdrant": {
        "stored_points": stored_count,
        "persistent_points": (
            reopened_count
        ),
        "write_time_seconds": (
            index_time
        ),
        "storage_path": str(
            QDRANT_PATH.relative_to(ROOT)
        ),
    },
}


REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    REPORT_PATH.relative_to(ROOT)
)


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "METADATA VECTOR INDEX: PASS"
)
