import os
from llama_index.core.llms.base import LLM
from pydantic import Field

MORPHEUS_API_KEY = os.environ.get("MORPHEUS_API_KEY")
MORPHEUS_BASE_URL = os.environ.get("MORPHEUS_BASE_URL", "https://api.mor.org/api/v1")

class MorpheusLLM(LLM):
    api_url: str = Field(default=MORPHEUS_BASE_URL)
    api_key: str = Field(default=MORPHEUS_API_KEY)
    model: str = Field(default="gpt-4o")

    def chat(self, prompt: str, **kwargs) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "prompt": prompt}
        response = requests.post(f"{self.api_url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
