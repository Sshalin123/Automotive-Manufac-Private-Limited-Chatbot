"""
Service initialization and dependency injection for AMPL Chatbot API.

Creates and manages all service instances used by the API.
"""

import logging
from typing import Optional

from config.settings import get_settings, Settings
from retrieval.embedder import EmbeddingService, EmbeddingConfig, EmbeddingProvider
from retrieval.pinecone_client import PineconeClient, PineconeConfig
from retrieval.context_builder import ContextBuilder
from retrieval.reranker import Reranker
from lead_scoring.intent_classifier import IntentClassifier
from lead_scoring.entity_extractor import EntityExtractor
from lead_scoring.scoring_model import LeadScorer
from lead_scoring.lead_router import LeadRouter, CRMProvider
from llm.orchestrator import ChatOrchestrator, LLMProvider

logger = logging.getLogger(__name__)


class Services:
    """Container for all application services."""

    def __init__(self):
        self.settings: Optional[Settings] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.pinecone_client: Optional[PineconeClient] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.reranker: Optional[Reranker] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        self.lead_scorer: Optional[LeadScorer] = None
        self.lead_router: Optional[LeadRouter] = None
        self.orchestrator: Optional[ChatOrchestrator] = None
        self._initialized = False

    def initialize(self):
        """Initialize all services."""
        if self._initialized:
            return

        self.settings = get_settings()
        logger.info(f"Initializing services with provider: {self.settings.llm_provider}")

        try:
            self._init_embedding()
            self._init_pinecone()
            self._init_context_builder()
            self._init_reranker()
            self._init_lead_scoring()
            self._init_orchestrator()
            self._initialized = True
            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            # Allow API to start even if some services fail
            self._initialized = True
            logger.warning("API starting in degraded mode")

    def _init_embedding(self):
        """Initialize embedding service."""
        s = self.settings

        if s.is_openai:
            provider = EmbeddingProvider.OPENAI
            model_id = s.openai_embed_model
        else:
            provider = EmbeddingProvider.BEDROCK_TITAN
            model_id = s.bedrock_embed_model_id

        config = EmbeddingConfig(
            provider=provider,
            model_id=model_id,
            aws_region=s.aws_region,
            openai_api_key=s.openai_api_key,
        )
        self.embedding_service = EmbeddingService(config)
        logger.info(f"Embedding service ready: {provider.value}")

    def _init_pinecone(self):
        """Initialize Pinecone client."""
        s = self.settings

        if not s.pinecone_api_key:
            logger.warning("PINECONE_API_KEY not set, vector search disabled")
            return

        config = PineconeConfig(
            api_key=s.pinecone_api_key,
            index_name=s.pinecone_index_name,
            cloud=s.pinecone_cloud,
            region=s.pinecone_region,
            dimension=self.embedding_service.get_dimension() if self.embedding_service else 1024,
        )
        self.pinecone_client = PineconeClient(config)
        logger.info("Pinecone client ready")

    def _init_context_builder(self):
        """Initialize context builder."""
        self.context_builder = ContextBuilder(
            max_tokens=self.settings.max_tokens * 3,
            max_chunks=self.settings.top_k,
        )

    def _init_reranker(self):
        """Initialize reranker."""
        self.reranker = Reranker()

    def _init_lead_scoring(self):
        """Initialize lead scoring components."""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.lead_scorer = LeadScorer()
        self.lead_scorer.adjust_thresholds(
            hot=self.settings.lead_score_threshold_hot,
            warm=self.settings.lead_score_threshold_warm,
        )

        crm_map = {
            "zoho": CRMProvider.ZOHO,
            "hubspot": CRMProvider.HUBSPOT,
            "salesforce": CRMProvider.SALESFORCE,
        }
        self.lead_router = LeadRouter(
            webhook_url=self.settings.crm_webhook_url,
            crm_provider=crm_map.get(self.settings.crm_provider, CRMProvider.CUSTOM),
            api_key=self.settings.crm_api_key,
            min_score_for_routing=self.settings.lead_score_threshold_warm,
            auto_route_hot_leads=self.settings.auto_route_leads,
        )
        logger.info("Lead scoring services ready")

    def _init_orchestrator(self):
        """Initialize the chat orchestrator."""
        s = self.settings

        llm_provider = LLMProvider.OPENAI if s.is_openai else LLMProvider.BEDROCK_CLAUDE

        self.orchestrator = ChatOrchestrator(
            embedding_service=self.embedding_service,
            pinecone_client=self.pinecone_client,
            context_builder=self.context_builder,
            intent_classifier=self.intent_classifier,
            entity_extractor=self.entity_extractor,
            lead_scorer=self.lead_scorer,
            lead_router=self.lead_router,
            llm_provider=llm_provider,
            llm_model_id=s.llm_model_id,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            brand_name=s.brand_name,
        )

        # Inject RM details and service config into orchestrator
        self.orchestrator.set_service_config({
            "rm_name": s.rm_name,
            "rm_phone": s.rm_phone,
            "rm_email": s.rm_email,
            "website_url": s.website_url,
            "toll_free_number": s.toll_free_number,
            "escalation_contacts": s.escalation_contacts_list,
        })
        logger.info("Chat orchestrator ready")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.orchestrator is not None

    def health(self) -> dict:
        """Return health status of all services."""
        return {
            "initialized": self._initialized,
            "embedding": self.embedding_service is not None,
            "pinecone": self.pinecone_client is not None,
            "lead_scoring": self.lead_scorer is not None,
            "orchestrator": self.orchestrator is not None,
        }


# Singleton
_services = Services()


def get_services() -> Services:
    """Get the global services instance."""
    return _services


def initialize_services():
    """Initialize all services (called at startup)."""
    _services.initialize()
