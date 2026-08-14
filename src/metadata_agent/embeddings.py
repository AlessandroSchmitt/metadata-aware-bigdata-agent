import json
import time
import urllib.request


class OllamaEmbedder:
    def __init__(
        self,
        model="embeddinggemma",
        base_url="http://127.0.0.1:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(
        self,
        texts,
        keep_alive="10m",
    ):
        if isinstance(texts, str):
            texts = [texts]

        payload = {
            "model": self.model,
            "input": texts,
            "keep_alive": keep_alive,
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
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

        wall_time = (
            time.perf_counter() - start
        )

        embeddings = result.get(
            "embeddings",
            []
        )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama returned an unexpected "
                "number of embeddings."
            )

        return {
            "embeddings": embeddings,
            "wall_time_seconds": wall_time,
            "prompt_eval_count": result.get(
                "prompt_eval_count",
                0,
            ),
            "load_duration_seconds": (
                result.get(
                    "load_duration",
                    0,
                )
                / 1_000_000_000
            ),
            "total_duration_seconds": (
                result.get(
                    "total_duration",
                    0,
                )
                / 1_000_000_000
            ),
        }
