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
from retrieval.token_estimator import TokenEstimator
from lead_scoring.intent_classifier import IntentClassifier
from lead_scoring.entity_extractor import EntityExtractor
from lead_scoring.scoring_model import LeadScorer
from lead_scoring.lead_router import LeadRouter, CRMProvider
from lead_scoring.cumulative_scorer import CumulativeScorer
from llm.orchestrator import ChatOrchestrator, LLMProvider
from llm.guardrails import ResponseVerifier
from retrieval.namespace_router import NamespaceRouter
from retrieval.query_preprocessor import QueryPreprocessor
from retrieval.query_decomposer import QueryDecomposer
from api.handoff.manager import HandoffManager
from api.experiments.engine import ExperimentEngine
from api.flows.engine import FlowEngine
from api.flows.definitions import register_all_flows
from api.knowledge.version_tracker import VersionTracker
from api.tenants.manager import TenantManager
from api.analytics.collector import AnalyticsCollector

logger = logging.getLogger(__name__)


class Services:
    """Container for all application services."""

    def __init__(self):
        self.settings: Optional[Settings] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.pinecone_client: Optional[PineconeClient] = None
        self.context_builder: Optional[ContextBuilder] = None
        self.reranker: Optional[Reranker] = None
        self.namespace_router: Optional[NamespaceRouter] = None
        self.query_preprocessor: Optional[QueryPreprocessor] = None
        self.query_decomposer: Optional[QueryDecomposer] = None
        self.intent_classifier: Optional[IntentClassifier] = None
        self.entity_extractor: Optional[EntityExtractor] = None
        self.lead_scorer: Optional[LeadScorer] = None
        self.lead_router: Optional[LeadRouter] = None
        self.orchestrator: Optional[ChatOrchestrator] = None
        self.token_estimator: Optional[TokenEstimator] = None
        self.cumulative_scorer: Optional[CumulativeScorer] = None
        self.response_verifier: Optional[ResponseVerifier] = None
        self.handoff_manager: Optional[HandoffManager] = None
        self.experiment_engine: Optional[ExperimentEngine] = None
        self.flow_engine: Optional[FlowEngine] = None
        self.version_tracker: Optional[VersionTracker] = None
        self.tenant_manager: Optional[TenantManager] = None
        self.analytics_collector: Optional[AnalyticsCollector] = None
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
            self._init_token_estimator()
            self._init_context_builder()
            self._init_reranker()
            self._init_namespace_router()
            self._init_query_preprocessor()
            self._init_query_decomposer()
            self._init_lead_scoring()
            self._init_guardrails()
            self._init_handoff()
            self._init_experiments()
            self._init_flows()
            self._init_knowledge()
            self._init_tenants()
            self._init_analytics()
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
        cache_size = s.embedding_cache_size if s.enable_embedding_cache else 0
        self.embedding_service = EmbeddingService(config, cache_size=cache_size)
        logger.info(f"Embedding service ready: {provider.value} (cache={cache_size})")

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

    def _init_token_estimator(self):
        """Initialize token estimator."""
        provider = "openai" if self.settings.is_openai else "bedrock"
        self.token_estimator = TokenEstimator(provider=provider)
        logger.info(f"Token estimator ready: {provider}")

    def _init_context_builder(self):
        """Initialize context builder with token estimator."""
        self.context_builder = ContextBuilder(
            max_tokens=self.settings.max_tokens * 3,
            max_chunks=self.settings.top_k,
            token_estimator=self.token_estimator,
        )

    def _init_reranker(self):
        """Initialize reranker (if enabled)."""
        if not self.settings.enable_reranker:
            logger.info("Reranker disabled via feature flag")
            return
        self.reranker = Reranker()
        logger.info("Reranker ready")

    def _init_namespace_router(self):
        """Initialize namespace router (if enabled)."""
        if not self.settings.enable_namespace_routing:
            logger.info("Namespace routing disabled via feature flag")
            return
        self.namespace_router = NamespaceRouter()
        logger.info("Namespace router ready")

    def _init_query_preprocessor(self):
        """Initialize query preprocessor (if enabled)."""
        if not self.settings.enable_query_preprocessing:
            logger.info("Query preprocessing disabled via feature flag")
            return
        self.query_preprocessor = QueryPreprocessor(
            enable_hinglish=self.settings.enable_hinglish_expansion,
            enable_abbreviations=True,
        )
        logger.info("Query preprocessor ready")

    def _init_query_decomposer(self):
        """Initialize query decomposer (if enabled)."""
        if not self.settings.enable_query_decomposition:
            logger.info("Query decomposition disabled via feature flag")
            return
        self.query_decomposer = QueryDecomposer(max_sub_queries=self.settings.max_sub_queries)
        logger.info("Query decomposer ready")

    def _init_lead_scoring(self):
        """Initialize lead scoring components."""
        self.intent_classifier = IntentClassifier()

        # Optionally enable LLM intent fallback
        if self.settings.enable_llm_intent_fallback:
            self._init_intent_llm_client()

        self.entity_extractor = EntityExtractor()
        self.lead_scorer = LeadScorer()
        self.lead_scorer.adjust_thresholds(
            hot=self.settings.lead_score_threshold_hot,
            warm=self.settings.lead_score_threshold_warm,
        )

        # Cumulative scorer wraps the base scorer
        self.cumulative_scorer = CumulativeScorer(
            base_scorer=self.lead_scorer,
            hot_threshold=self.settings.lead_score_threshold_hot,
            warm_threshold=self.settings.lead_score_threshold_warm,
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
        logger.info("Lead scoring services ready (cumulative scorer enabled)")

    def _init_intent_llm_client(self):
        """Set up LLM client for intent classifier fallback."""
        s = self.settings
        try:
            if s.is_openai:
                from openai import OpenAI
                self.intent_classifier.llm_client = OpenAI()
                self.intent_classifier._llm_model_id = s.openai_llm_model
            else:
                import boto3
                self.intent_classifier.llm_client = boto3.client("bedrock-runtime")
                self.intent_classifier._llm_model_id = s.bedrock_llm_model_id
            logger.info("LLM intent fallback enabled")
        except Exception as e:
            logger.warning(f"LLM intent fallback init failed: {e}")

    def _init_guardrails(self):
        """Initialize response verifier / guardrails."""
        self.response_verifier = ResponseVerifier(brand_name=self.settings.brand_name)
        logger.info("Response guardrails ready")

    def _init_handoff(self):
        """Initialize human agent handoff manager."""
        self.handoff_manager = HandoffManager()
        logger.info("Handoff manager ready")

    def _init_experiments(self):
        """Initialize A/B testing engine."""
        self.experiment_engine = ExperimentEngine()
        logger.info("Experiment engine ready")

    def _init_flows(self):
        """Initialize conversation flow engine with pre-built flows."""
        self.flow_engine = FlowEngine()
        register_all_flows(self.flow_engine)
        logger.info("Flow engine ready")

    def _init_knowledge(self):
        """Initialize knowledge base version tracker."""
        self.version_tracker = VersionTracker()
        logger.info("Knowledge version tracker ready")

    def _init_tenants(self):
        """Initialize multi-tenancy manager."""
        self.tenant_manager = TenantManager()
        logger.info("Tenant manager ready")

    def _init_analytics(self):
        """Initialize analytics collector."""
        self.analytics_collector = AnalyticsCollector()
        logger.info("Analytics collector ready")

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
            reranker=self.reranker,
            llm_provider=llm_provider,
            llm_model_id=s.llm_model_id,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            brand_name=s.brand_name,
        )

        # Inject namespace router and query preprocessor
        if self.namespace_router:
            self.orchestrator.set_namespace_router(self.namespace_router)
        if self.query_preprocessor:
            self.orchestrator.set_query_preprocessor(self.query_preprocessor)
        if self.query_decomposer:
            self.orchestrator.set_query_decomposer(self.query_decomposer)

        # Inject new components
        if self.response_verifier:
            self.orchestrator.set_response_verifier(self.response_verifier)
        if self.cumulative_scorer:
            self.orchestrator.set_cumulative_scorer(self.cumulative_scorer)
        if self.handoff_manager:
            self.orchestrator.set_handoff_manager(self.handoff_manager)
        if self.experiment_engine:
            self.orchestrator.set_experiment_engine(self.experiment_engine)
        if self.flow_engine:
            self.orchestrator.set_flow_engine(self.flow_engine)
        if self.token_estimator:
            self.orchestrator.set_token_estimator(self.token_estimator)

        # Inject feature flags for runtime optimization toggles
        self.orchestrator.set_feature_flags({
            "enable_conversation_aware_retrieval": s.enable_conversation_aware_retrieval,
            "enable_adaptive_thresholds": s.enable_adaptive_thresholds,
            "enable_parallel_pipeline": s.enable_parallel_pipeline,
            "enable_hybrid_search": s.enable_hybrid_search,
            "enable_llm_intent_fallback": s.enable_llm_intent_fallback,
        })

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
            "guardrails": self.response_verifier is not None,
            "handoff": self.handoff_manager is not None,
            "experiments": self.experiment_engine is not None,
            "flows": self.flow_engine is not None,
            "tenants": self.tenant_manager is not None,
            "analytics": self.analytics_collector is not None,
        }


# Singleton
_services = Services()


def get_services() -> Services:
    """Get the global services instance."""
    return _services


def initialize_services():
    """Initialize all services (called at startup)."""
    _services.initialize()
