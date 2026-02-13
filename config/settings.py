"""
Centralized configuration for AMPL Chatbot.

All settings are loaded from environment variables via .env file.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

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

    # RAG Optimization Feature Flags
    enable_reranker: bool = Field(default=True, env="ENABLE_RERANKER")
    enable_namespace_routing: bool = Field(default=True, env="ENABLE_NAMESPACE_ROUTING")
    enable_query_preprocessing: bool = Field(default=True, env="ENABLE_QUERY_PREPROCESSING")
    enable_hinglish_expansion: bool = Field(default=True, env="ENABLE_HINGLISH_EXPANSION")
    enable_conversation_aware_retrieval: bool = Field(default=True, env="ENABLE_CONVERSATION_AWARE_RETRIEVAL")
    enable_adaptive_thresholds: bool = Field(default=True, env="ENABLE_ADAPTIVE_THRESHOLDS")
    enable_parallel_pipeline: bool = Field(default=True, env="ENABLE_PARALLEL_PIPELINE")
    enable_query_decomposition: bool = Field(default=False, env="ENABLE_QUERY_DECOMPOSITION")  # off by default, advanced
    enable_hybrid_search: bool = Field(default=True, env="ENABLE_HYBRID_SEARCH")
    enable_llm_intent_fallback: bool = Field(default=False, env="ENABLE_LLM_INTENT_FALLBACK")  # off by default, costs LLM calls
    max_sub_queries: int = Field(default=4, env="MAX_SUB_QUERIES")

    # Adaptive threshold levels
    similarity_threshold_high: float = Field(default=0.45, env="SIMILARITY_THRESHOLD_HIGH")
    similarity_threshold_medium: float = Field(default=0.35, env="SIMILARITY_THRESHOLD_MEDIUM")
    similarity_threshold_low: float = Field(default=0.25, env="SIMILARITY_THRESHOLD_LOW")
    similarity_threshold_fallback: float = Field(default=0.15, env="SIMILARITY_THRESHOLD_FALLBACK")

    # Lead scoring
    lead_score_threshold_hot: int = Field(default=70, env="LEAD_SCORE_THRESHOLD_HOT")
    lead_score_threshold_warm: int = Field(default=50, env="LEAD_SCORE_THRESHOLD_WARM")
    auto_route_leads: bool = Field(default=True, env="AUTO_ROUTE_LEADS")

    # CRM
    crm_webhook_url: Optional[str] = Field(default=None, env="CRM_WEBHOOK_URL")
    crm_api_key: Optional[str] = Field(default=None, env="CRM_API_KEY")
    crm_provider: str = Field(default="custom", env="CRM_PROVIDER")

    # Database (Gap 14.1)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    db_pool_size: int = Field(default=5, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, env="DB_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Embedding cache (Gap 5.2)
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    embedding_cache_size: int = Field(default=2000, env="EMBEDDING_CACHE_SIZE")

    # JWT Auth (Gap 14.2)
    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60, env="JWT_EXPIRE_MINUTES")
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")

    # WhatsApp (Gap 14.4)
    whatsapp_provider: str = Field(default="meta", env="WHATSAPP_PROVIDER")  # meta | gupshup
    whatsapp_api_token: Optional[str] = Field(default=None, env="WHATSAPP_API_TOKEN")
    whatsapp_phone_number_id: Optional[str] = Field(default=None, env="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_verify_token: Optional[str] = Field(default=None, env="WHATSAPP_VERIFY_TOKEN")
    gupshup_api_key: Optional[str] = Field(default=None, env="GUPSHUP_API_KEY")
    gupshup_app_name: Optional[str] = Field(default=None, env="GUPSHUP_APP_NAME")

    # Email (Gap 14.4)
    email_provider: str = Field(default="sendgrid", env="EMAIL_PROVIDER")  # sendgrid | ses
    sendgrid_api_key: Optional[str] = Field(default=None, env="SENDGRID_API_KEY")
    ses_region: str = Field(default="us-east-1", env="SES_REGION")
    email_from_address: str = Field(default="", env="EMAIL_FROM_ADDRESS")
    email_from_name: str = Field(default="AMPL", env="EMAIL_FROM_NAME")

    # CORS
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_title: str = Field(default="AMPL Automotive Chatbot API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    ampl_api_key: Optional[str] = Field(default=None, env="AMPL_API_KEY")

    # RM (Relationship Manager) details — included in enquiry greetings
    rm_name: str = Field(default="AMPL Sales Team", env="RM_NAME")
    rm_phone: str = Field(default="", env="RM_PHONE")
    rm_email: str = Field(default="", env="RM_EMAIL")
    website_url: str = Field(default="https://www.amplindia.com", env="WEBSITE_URL")
    toll_free_number: str = Field(default="", env="TOLL_FREE_NUMBER")

    # Escalation matrix contacts (JSON string)
    # Format: [{"role":"Sales Head","name":"...","phone":"...","email":"..."}]
    escalation_contacts: str = Field(default="[]", env="ESCALATION_CONTACTS")

    # Service milestones for reminders (JSON string)
    # Format: [{"name":"Welcome Call","days":7},{"name":"1st Free Service","km":1000,"days":30}, ...]
    service_milestones: str = Field(
        default='[{"name":"Welcome Call","days":7},{"name":"1st Free Service","km":1000,"days":30},{"name":"2nd Free Service","km":5000,"days":180},{"name":"3rd Free Service","km":10000,"days":365}]',
        env="SERVICE_MILESTONES"
    )

    # Follow-up intervals in days after enquiry [immediate, day+1, day+15]
    followup_intervals: str = Field(default="[0, 1, 15]", env="FOLLOWUP_INTERVALS")

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

    @property
    def escalation_contacts_list(self) -> List[Dict]:
        return json.loads(self.escalation_contacts)

    @property
    def service_milestones_list(self) -> List[Dict]:
        return json.loads(self.service_milestones)

    @property
    def followup_intervals_list(self) -> List[int]:
        return json.loads(self.followup_intervals)

    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
