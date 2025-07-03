import os
from llama_index.embeddings.base import BaseEmbedding
from pydantic import Field

MORPHEUS_API_KEY = os.environ.get("MORPHEUS_API_KEY")
MORPHEUS_BASE_URL = os.environ.get("MORPHEUS_BASE_URL", "https://api.mor.org/api/v1")
MORPHEUS_EMBEDDING_MODEL = os.environ.get("MORPHEUS_EMBEDDING_MODEL", "LMR-OpenAI-GPT-4o")  # Or the correct embedding model ID if Morpheus exposes a dedicated one

class MorpheusEmbedding(BaseEmbedding):
    api_url: str = Field(default=MORPHEUS_BASE_URL)
    api_key: str = Field(default=MORPHEUS_API_KEY)
    model: str = Field(default=MORPHEUS_EMBEDDING_MODEL)

    def _embed(self, texts: list[str], model: str = None) -> list[list[float]]:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        chosen_model = model or self.model
        data = {"model": chosen_model, "input": texts}
        response = requests.post(f"{self.api_url}/embeddings", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["data"]
