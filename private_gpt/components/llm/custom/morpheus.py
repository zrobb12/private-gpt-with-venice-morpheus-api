import os
from llama_index.core.llms.base import LLM
from pydantic import Field

MORPHEUS_API_KEY = os.environ.get("MORPHEUS_API_KEY")
MORPHEUS_BASE_URL = os.environ.get("MORPHEUS_BASE_URL", "https://api.mor.org/api/v1")
MORPHEUS_MODEL = os.environ.get("MORPHEUS_MODEL", "LMR-OpenAI-GPT-4o")  # Default

# Morpheus-supported models (for user reference):
# - LMR-OpenAI-GPT-4o (default)
# - LMR-ClaudeAI-Sonnet
# - Llama 2.0
# - Llama 7.1-dev
# - LMR-Hyperbolic-SD
# - nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
# - meta-llama/Llama-3.2-3B-Instruct
# - NousResearch/Hermes-2-Theta-Llama-3-8B
# - cognitivecomputations/dolphin-2.9.2-qwen2-72b
# - Hermes 3 Llama 3.1
# - qwen:4b
# - LMR-Prodia-MochiTest
# - tinyllama-1.1b
# - LMR-ProdiaSD
# - LMR-ProdiaSDXL
# - nfa-llama2
# - LMR2-Hermes-3-Llama-3.1-8B
# - Llama 7.1-dev
# - LMR2-Hermes-3-Llama-3.1-8B
# - itrl-meta-llama-3-3-70b-instruct-awq-int4-1tp-1pp
# - dolphin-2.9.2-qwen2-72b
# - qwen-2.5-qwq-32b
# - qwen-2.5-vl
# - venice-uncensored
# - Whisper-1
# - SmolLM2-360M-Heroku
# - venice-uncensored-web
# - qwen3-235b
# - qwen3-235b-web
# - llama-3.3-70b-web
# - mistral-31-24b-web

class MorpheusLLM(LLM):
    api_url: str = Field(default=MORPHEUS_BASE_URL)
    api_key: str = Field(default=MORPHEUS_API_KEY)
    model: str = Field(default=MORPHEUS_MODEL)

    def chat(self, prompt: str, model: str = None, **kwargs) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        chosen_model = model or self.model
        data = {"model": chosen_model, "prompt": prompt}
        response = requests.post(f"{self.api_url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
