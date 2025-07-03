import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.paths import models_cache_path, models_path
from private_gpt.settings.settings import Settings

# Custom Embeddings
from private_gpt.components.embedding.custom.venice import VeniceEmbedding
from private_gpt.components.embedding.custom.morpheus import MorpheusEmbedding

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the Embedding model in mode=%s", embedding_mode)
        match settings.embedding.mode:
            case "llamacpp":
                try:
                    from llama_index.embeddings.llama_cpp import LlamaCPPEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "LlamaCPP embedding dependencies not found, install with `poetry install --extras llms-llama-cpp`"
                    ) from e

                self.embedding_model = LlamaCPPEmbedding(
                    model_path=str(models_path / settings.llamacpp.embedding_hf_model_file),
                    cache_dir=str(models_cache_path),
                    max_seq_length=settings.embedding.max_seq_length,
                    context_window=settings.embedding.context_window,
                )

            case "openai":
                try:
                    from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAI embedding dependencies not found, install with `poetry install --extras llms-openai`"
                    ) from e

                openai_settings = settings.openai
                self.embedding_model = OpenAIEmbedding(
                    api_base=openai_settings.api_base,
                    api_key=openai_settings.api_key,
                    model=openai_settings.embedding_model,
                )

            case "openailike":
                try:
                    from llama_index.embeddings.openai_like import OpenAILikeEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAILike embedding dependencies not found, install with `poetry install --extras llms-openai-like`"
                    ) from e
                openai_settings = settings.openai
                self.embedding_model = OpenAILikeEmbedding(
                    api_base=openai_settings.api_base,
                    api_key=openai_settings.api_key,
                    model=openai_settings.embedding_model,
                )

            case "ollama":
                try:
                    from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Ollama embedding dependencies not found, install with `poetry install --extras llms-ollama`"
                    ) from e

                ollama_settings = settings.ollama
                model_name = (
                    ollama_settings.embedding_model + ":latest"
                    if ":" not in ollama_settings.embedding_model
                    else ollama_settings.embedding_model
                )
                self.embedding_model = OllamaEmbedding(
                    model=model_name,
                    base_url=ollama_settings.api_base,
                )

            case "azopenai":
                try:
                    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Azure OpenAI embedding dependencies not found, install with `poetry install --extras llms-azopenai`"
                    ) from e

                azopenai_settings = settings.azopenai
                self.embedding_model = AzureOpenAIEmbedding(
                    model=azopenai_settings.embedding_model,
                    deployment_name=azopenai_settings.embedding_deployment_name,
                    api_key=azopenai_settings.api_key,
                    azure_endpoint=azopenai_settings.azure_endpoint,
                    api_version=azopenai_settings.api_version,
                )

            case "gemini":
                try:
                    from llama_index.embeddings.gemini import GeminiEmbedding  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Google Gemini embedding dependencies not found, install with `poetry install --extras llms-gemini`"
                    ) from e
                gemini_settings = settings.gemini
                self.embedding_model = GeminiEmbedding(
                    model_name=gemini_settings.embedding_model, api_key=gemini_settings.api_key
                )

            case "venice":
                self.embedding_model = VeniceEmbedding()  # Uses env for embedding model
            case "morpheus":
                self.embedding_model = MorpheusEmbedding()  # Uses env for embedding model

            case "mock":
                from llama_index.core.embeddings import MockEmbedding
                self.embedding_model = MockEmbedding()
