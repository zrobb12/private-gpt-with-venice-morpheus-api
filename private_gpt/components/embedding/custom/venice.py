import os
from llama_index.embeddings.base import BaseEmbedding
from pydantic import Field

VENICE_API_KEY = os.environ.get("VENICE_API_KEY")
VENICE_BASE_URL = os.environ.get("VENICE_BASE_URL", "https://api.venice.ai/api/v1")
VENICE_EMBEDDING_MODEL = os.environ.get("VENICE_EMBEDDING_MODEL", "default-model")  # Replace with actual model if needed

class VeniceEmbedding(BaseEmbedding):
    api_url: str = Field(default=VENICE_BASE_URL)
    api_key: str = Field(default=VENICE_API_KEY)
    model: str = Field(default=VENICE_EMBEDDING_MODEL)

    def _embed(self, texts: list[str], model: str = None) -> list[list[float]]:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        chosen_model = model or self.model
        data = {"model": chosen_model, "input": texts}
        response = requests.post(f"{self.api_url}/embeddings", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["data"]
