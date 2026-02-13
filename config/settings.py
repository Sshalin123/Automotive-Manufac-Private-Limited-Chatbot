"""
Centralized configuration for AMPL Chatbot.

All settings are loaded from environment variables via .env file.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Brand
    brand_name: str = Field(default="AMPL", env="BRAND_NAME")

    # AWS / Bedrock
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", env="BEDROCK_EMBED_MODEL_ID"
    )
    bedrock_llm_model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-20250514-v1:0", env="BEDROCK_LLM_MODEL_ID"
    )

    # OpenAI (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_embed_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    openai_llm_model: str = Field(default="gpt-4o-mini", env="OPENAI_LLM_MODEL")

    # LLM provider selection
    llm_provider: str = Field(default="bedrock", env="LLM_PROVIDER")  # bedrock | openai
    max_tokens: int = Field(default=1024, env="MAX_TOKENS")
    temperature: float = Field(default=0.3, env="TEMPERATURE")

    # Pinecone
    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="ampl-chatbot", env="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", env="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", env="PINECONE_REGION")

    # RAG
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.5, env="SIMILARITY_THRESHOLD")

    # Lead scoring
    lead_score_threshold_hot: int = Field(default=70, env="LEAD_SCORE_THRESHOLD_HOT")
    lead_score_threshold_warm: int = Field(default=50, env="LEAD_SCORE_THRESHOLD_WARM")
    auto_route_leads: bool = Field(default=True, env="AUTO_ROUTE_LEADS")

    # CRM
    crm_webhook_url: Optional[str] = Field(default=None, env="CRM_WEBHOOK_URL")
    crm_api_key: Optional[str] = Field(default=None, env="CRM_API_KEY")
    crm_provider: str = Field(default="custom", env="CRM_PROVIDER")

    # Database
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_title: str = Field(default="AMPL Automotive Chatbot API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    ampl_api_key: Optional[str] = Field(default=None, env="AMPL_API_KEY")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")

    # Data
    data_directory: str = Field(default="./data", env="DATA_DIRECTORY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def is_bedrock(self) -> bool:
        return self.llm_provider.lower() == "bedrock"

    @property
    def is_openai(self) -> bool:
        return self.llm_provider.lower() == "openai"

    @property
    def embed_model_id(self) -> str:
        if self.is_openai:
            return self.openai_embed_model
        return self.bedrock_embed_model_id

    @property
    def llm_model_id(self) -> str:
        if self.is_openai:
            return self.openai_llm_model
        return self.bedrock_llm_model_id


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
