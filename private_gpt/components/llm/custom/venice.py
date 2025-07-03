import os
from llama_index.core.llms.base import LLM
from pydantic import Field

VENICE_API_KEY = os.environ.get("VENICE_API_KEY")
VENICE_BASE_URL = os.environ.get("VENICE_BASE_URL", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "venice-uncensored")  # Default

# Venice-supported models (for reference/documentation):
# - venice-uncensored      (Venice Uncensored 1.1)
# - qwen-2.5-qwq-32b       (Venice Reasoning)
# - qwen3-4b               (Venice Small)
# - mistral-31-24b         (Venice Medium)
# - qwen3-235b             (Venice Large)
# - llama-3.2-3b           (Llama 3.2 3B)
# - llama-3.3-70b          (Llama 3.3 70B)
# - llama-3.1-405b         (Llama 3.1 405B)
# - dolphin-2.9.2-qwen2-72b (Dolphin 72B)
# - qwen-2.5-vl            (Qwen 2.5 VL 72B)
# - qwen-2.5-coder-32b     (Qwen 2.5 Coder 32B)
# - deepseek-r1-671b       (DeepSeek R1 671B)
# - deepseek-coder-v2-lite (DeepSeek Coder V2 Lite)

class VeniceLLM(LLM):
    api_url: str = Field(default=VENICE_BASE_URL)
    api_key: str = Field(default=VENICE_API_KEY)
    model: str = Field(default=VENICE_MODEL)

    def chat(self, prompt: str, model: str = None, **kwargs) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}"}
        chosen_model = model or self.model
        data = {"model": chosen_model, "prompt": prompt}
        response = requests.post(f"{self.api_url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
