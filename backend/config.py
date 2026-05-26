from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://agent:agentpass@localhost:5432/ecommerce"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # LangFuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_tokens_per_request: int = 2000
    session_ttl_seconds: int = 1800
    memory_window_size: int = 6

    # LLM routing
    short_prompt_threshold: int = 150
    model_fast: str = "gpt-3.5-turbo"
    model_smart: str = "gpt-4o"

    # RAG
    top_k_rag: int = 5

    # Semantic cache
    semantic_cache_threshold: float = 0.87   # cosine similarity — tuned from observed scores
    semantic_cache_enabled: bool = True

    # Auto few-shot learning
    few_shot_top_k: int = 3                  # examples to inject per query
    few_shot_store_path: str = "data/few_shot_examples.json"

    # Self-healing SQL
    sql_max_retries: int = 2

    # Streaming TTS
    tts_streaming_enabled: bool = True
    tts_chunk_words: int = 25               # words per TTS audio chunk

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        protected_namespaces = ("settings_",)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
