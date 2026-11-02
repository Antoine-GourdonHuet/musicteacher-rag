"""Central configuration, loaded from environment variables and an optional `.env` file.

All variables use the `MUSICTEACHER_` prefix, e.g. `MUSICTEACHER_GENERATOR_MODEL`.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="MUSICTEACHER_", extra="ignore"
    )

    # Local LLMs served by Ollama (https://ollama.com/library).
    ollama_base_url: str = "http://localhost:11434"
    generator_model: str = "qwen3.5:9b"
    # Placeholder: a judge from another model family is chosen with the evaluation.
    judge_model: str = "qwen3.5:9b"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Wikimedia API etiquette: identify the client and do not hammer the servers.
    wikipedia_user_agent: str = (
        "MusicTeacherRAG/0.1 (educational research project; "
        "https://github.com/Antoine-GourdonHuet/musicteacher-rag)"
    )
    wikipedia_delay_s: float = 1.0

    # Retrieval: hybrid fuses the top `fusion_depth` of dense and BM25 with RRF (k=60, the value
    # from the original RRF paper); the reranker re-scores the top `rerank_candidates` results.
    fusion_depth: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 20

    seed: int = 42
    top_k: int = 5
    temperature: float = 0.0

    data_dir: Path = ROOT / "data"
    evals_dir: Path = ROOT / "evals"
    cache_dir: Path = ROOT / ".cache"
    models_dir: Path = ROOT / "models"


def get_settings() -> Settings:
    return Settings()
