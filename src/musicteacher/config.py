"""Central configuration, loaded from environment variables and an optional `.env` file.

All variables use the `MUSICTEACHER_` prefix, e.g. `MUSICTEACHER_ANTHROPIC_API_KEY`.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="MUSICTEACHER_", extra="ignore"
    )

    anthropic_api_key: SecretStr | None = None

    # Verify current model names in the Anthropic docs.
    generator_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-haiku-5-5"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    seed: int = 42
    top_k: int = 5
    temperature: float = 0.0

    data_dir: Path = ROOT / "data"
    evals_dir: Path = ROOT / "evals"
    cache_dir: Path = ROOT / ".cache"
    models_dir: Path = ROOT / "models"


def get_settings() -> Settings:
    return Settings()
