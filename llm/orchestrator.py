"""
Chat Orchestrator for AMPL Chatbot.

Orchestrates the full RAG pipeline from query to response.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .prompt_templates import PromptTemplates, PromptType

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    BEDROCK_CLAUDE = "bedrock_claude"
    OPENAI = "openai"


@dataclass
class ChatRequest:
    """Request for chat completion."""
    conversation_id: str
    message: str
    user_context: Optional[Dict[str, Any]] = None
    max_chunks: int = 5
    include_sources: bool = True


@dataclass
class ChatResponse:
    """Response from chat completion."""
    response: str
    conversation_id: str
    query: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    lead_score: Optional[int] = None
    lead_priority: Optional[str] = None
    intent: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "sources": self.sources,
            "lead_score": self.lead_score,
            "lead_priority": self.lead_priority,
            "intent": self.intent,
            "suggested_actions": self.suggested_actions,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ChatOrchestrator:
    """
    Orchestrates the chat pipeline.

    Pipeline:
    1. Receive user message
    2. Classify intent
    3. Extract entities
    4. Generate query embedding
    5. Retrieve relevant context
    6. Build prompt
    7. Generate LLM response
    8. Score lead
    9. Route if qualified
    10. Return response
    """

    def __init__(
        self,
        embedding_service: Any,
        pinecone_client: Any,
        context_builder: Any,
        intent_classifier: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
        lead_scorer: Optional[Any] = None,
        lead_router: Optional[Any] = None,
        reranker: Optional[Any] = None,
        llm_provider: LLMProvider = LLMProvider.BEDROCK_CLAUDE,
        llm_model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        brand_name: str = "AMPL"
    ):
        """
        Initialize the orchestrator.

        Args:
            embedding_service: Service for generating embeddings
            pinecone_client: Client for vector operations
            context_builder: Builder for assembling context
            intent_classifier: Optional intent classifier
            entity_extractor: Optional entity extractor
            lead_scorer: Optional lead scorer
            lead_router: Optional lead router
            reranker: Optional reranker for result reranking
            llm_provider: LLM provider to use
            llm_model_id: Model ID for LLM
            max_tokens: Max tokens for response
            temperature: Temperature for generation
            brand_name: Brand name for prompts
        """
        self.embedding_service = embedding_service
        self.pinecone_client = pinecone_client
        self.context_builder = context_builder
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.lead_scorer = lead_scorer
        self.lead_router = lead_router
        self.reranker = reranker
        self.namespace_router = None  # set via set_namespace_router()
        self.query_preprocessor = None  # set via set_query_preprocessor()
        self.query_decomposer = None  # set via set_query_decomposer()
        self.llm_provider = llm_provider
        self.llm_model_id = llm_model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.brand_name = brand_name

        # New components (set via setter methods)
        self._response_verifier = None
        self._cumulative_scorer = None
        self._handoff_manager = None
        self._experiment_engine = None
        self._flow_engine = None
        self._token_estimator = None

        # Feature flags (set via set_feature_flags())
        self._feature_flags: Dict[str, Any] = {}

        # Conversation state
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self._conversation_turns: Dict[str, int] = {}
        self._conversation_stages: Dict[str, str] = {}       # enquiry / booked / delivered / servicing
        self._conversation_created: Dict[str, datetime] = {}  # for 15-day tracking

        # Context window limits (80% of model max for safety margin)
        self._context_limits = {
            LLMProvider.OPENAI: 102_400,       # 128K * 0.8
            LLMProvider.BEDROCK_CLAUDE: 160_000,  # 200K * 0.8
        }

        # Service config (RM details, escalation contacts, etc.)
        self._service_config: Dict[str, Any] = {}

        # Notification callback for outbound messages (webhooks, WhatsApp, etc.)
        self._notification_callback: Optional[Callable] = None

        # Initialize LLM client
        self._llm_client = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM client."""
        if self.llm_provider == LLMProvider.BEDROCK_CLAUDE:
            try:
                import boto3
                self._llm_client = boto3.client("bedrock-runtime")
                logger.info(f"Bedrock LLM client initialized with {self.llm_model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock: {e}")
                raise
        elif self.llm_provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
                self._llm_client = OpenAI()
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                raise

    async def process(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request through the full pipeline.

        Args:
            request: Chat request

        Returns:
            Chat response
        """
        start_time = time.time()

        try:
            # Step 0: Check handoff — if conversation is in handoff, skip bot response
            if self._handoff_manager and self._handoff_manager.is_in_handoff(request.conversation_id):
                processing_time = (time.time() - start_time) * 1000
                return ChatResponse(
                    response="You are currently connected with a human agent. They will respond shortly.",
                    conversation_id=request.conversation_id,
                    query=request.message,
                    processing_time_ms=round(processing_time, 2),
                    metadata={"handoff_active": True},
                )

            # Step 1: Update conversation state
            self._update_conversation(request.conversation_id, "user", request.message)
            turns = self._conversation_turns.get(request.conversation_id, 1)

            # Step 1a: Apply A/B experiment overrides
            experiment_overrides = {}
            if self._experiment_engine:
                experiment_overrides = self._experiment_engine.get_overrides(request.conversation_id)

            # Step 2-4: Pipeline (parallel or sequential based on feature flag)
            # Preprocessing is fast/sync, runs before the parallel group
            embedding_query = request.message
            if self.query_preprocessor:
                embedding_query = self.query_preprocessor.preprocess(request.message)

            # Conversation-aware retrieval enrichment
            if self._feature_flags.get("enable_conversation_aware_retrieval", True):
                retrieval_query = self._build_retrieval_query(embedding_query, request.conversation_id)
            else:
                retrieval_query = embedding_query

            use_llm_intent = self._feature_flags.get("enable_llm_intent_fallback", False)

            if self._feature_flags.get("enable_parallel_pipeline", True):
                # Parallel: run intent, entity extraction, and embedding concurrently
                async def _classify_intent():
                    if self.intent_classifier:
                        conversation_history = self._conversations.get(request.conversation_id, [])
                        return await asyncio.to_thread(
                            self.intent_classifier.classify,
                            request.message,
                            conversation_history=conversation_history[-5:],
                            use_llm=use_llm_intent,
                        )
                    return None

                async def _extract_entities():
                    if self.entity_extractor:
                        return await asyncio.to_thread(
                            self.entity_extractor.extract,
                            request.message
                        )
                    return None

                intent_result, entities, query_embedding = await asyncio.gather(
                    _classify_intent(),
                    _extract_entities(),
                    self.embedding_service.embed_text(retrieval_query),
                )
            else:
                # Sequential fallback
                intent_result = None
                if self.intent_classifier:
                    conversation_history = self._conversations.get(request.conversation_id, [])
                    intent_result = self.intent_classifier.classify(
                        request.message,
                        conversation_history=conversation_history[-5:],
                        use_llm=use_llm_intent,
                    )

                entities = None
                if self.entity_extractor:
                    entities = self.entity_extractor.extract(request.message)

                query_embedding = await self.embedding_service.embed_text(retrieval_query)

            # Step 5: Retrieve relevant context (with smart namespace routing + adaptive threshold)
            target_namespaces = None
            if self.namespace_router:
                target_namespaces = self.namespace_router.route(intent_result, entities)

            if self._feature_flags.get("enable_adaptive_thresholds", True):
                adaptive_threshold = self._get_adaptive_threshold(intent_result)
            else:
                adaptive_threshold = 0.5  # original hard-coded value

            # Step 5a: Query decomposition for multi-facet queries
            sub_queries = None
            if self.query_decomposer:
                sub_queries = self.query_decomposer.decompose(retrieval_query, entities)

            if sub_queries and len(sub_queries) > 1:
                # Multi-query fanout: embed sub-queries and merge results
                sub_embeddings = await self.embedding_service.embed_texts(sub_queries)
                all_results = []
                seen_ids = set()

                for sub_emb in sub_embeddings:
                    results = self.pinecone_client.query_all_namespaces(
                        embedding=sub_emb,
                        top_k=request.max_chunks,
                        namespaces=target_namespaces,
                        min_score=adaptive_threshold,
                    )
                    for r in results:
                        if r.id not in seen_ids:
                            seen_ids.add(r.id)
                            all_results.append(r)

                # Sort by score and take top_k
                all_results.sort(key=lambda x: x.score, reverse=True)
                search_results = all_results[:request.max_chunks]
            else:
                # Standard single-query retrieval with optional hybrid search
                metadata_filter = None
                if self._feature_flags.get("enable_hybrid_search", True):
                    metadata_filter = self._build_metadata_filter(entities)

                if metadata_filter:
                    # Hybrid search: run filtered (high precision) + unfiltered (high recall) in parallel
                    filtered_results, unfiltered_results = await asyncio.gather(
                        asyncio.to_thread(
                            self.pinecone_client.query_all_namespaces,
                            embedding=query_embedding,
                            top_k=3,
                            namespaces=target_namespaces,
                            filter=metadata_filter,
                            min_score=adaptive_threshold,
                        ),
                        asyncio.to_thread(
                            self.pinecone_client.query_all_namespaces,
                            embedding=query_embedding,
                            top_k=request.max_chunks,
                            namespaces=target_namespaces,
                            min_score=adaptive_threshold,
                        ),
                    )
                    # Merge + deduplicate, filtered results first
                    seen_ids = set()
                    merged = []
                    for r in filtered_results + unfiltered_results:
                        if r.id not in seen_ids:
                            seen_ids.add(r.id)
                            merged.append(r)
                    search_results = merged[:request.max_chunks]
                else:
                    search_results = self.pinecone_client.query_all_namespaces(
                        embedding=query_embedding,
                        top_k=request.max_chunks,
                        namespaces=target_namespaces,
                        min_score=adaptive_threshold,
                    )

            # Fallback retry: if no results, retry with minimal threshold
            if not search_results:
                search_results = self.pinecone_client.query_all_namespaces(
                    embedding=query_embedding,
                    top_k=1,
                    namespaces=target_namespaces,
                    min_score=0.15,
                )
                if search_results:
                    logger.info("Adaptive threshold fallback: retrieved best-effort result")

            # Step 5b: Rerank results (if reranker available)
            if self.reranker and search_results:
                intent_str_for_rerank = intent_result.primary_intent.value if intent_result else None
                reranked = self.reranker.rerank(
                    query=request.message,
                    results=search_results,
                    top_k=request.max_chunks,
                    intent=intent_str_for_rerank,
                    entities=entities,
                )
                search_results = [r.original for r in reranked]

            # Step 6: Build context
            context = self.context_builder.build(
                results=search_results,
                query=request.message
            )

            # Step 7: Determine prompt type and build prompt
            intent_str = intent_result.primary_intent.value if intent_result else None
            prompt_type = PromptTemplates.detect_prompt_type(request.message, intent_str)

            # Detect language for multilingual support
            language_instruction = self._get_language_instruction(request.message)

            system_prompt = PromptTemplates.get_system_prompt(
                prompt_type=prompt_type,
                brand_name=self.brand_name,
                custom_instructions=language_instruction
            )

            # For first message (turn 1), inject RM details into prompt if available
            if turns == 1 and self._service_config.get("rm_name"):
                rm_info = (
                    f"\n\nIMPORTANT: This is the customer's first message. Include these RM details in your response:\n"
                    f"- RM Name: {self._service_config.get('rm_name', '')}\n"
                    f"- Phone: {self._service_config.get('rm_phone', '')}\n"
                    f"- Email: {self._service_config.get('rm_email', '')}\n"
                    f"- Website: {self._service_config.get('website_url', '')}"
                )
                system_prompt += rm_info

            # Pre-booking enforcement: if BUY intent + IMMEDIATE/THIS_MONTH timeline, check mandatory fields
            if (intent_result and entities and
                intent_result.primary_intent.value == "buy" and
                intent_result.timeline.value in ("immediate", "this_month") and
                not entities.has_contact_info()):
                system_prompt += (
                    "\n\nIMPORTANT: The customer appears ready to purchase soon. "
                    "Please ask for their name and contact number to proceed with pre-booking."
                )

            # Flow engine: inject missing-field prompt
            if self._flow_engine and intent_str:
                flow_prompt = self._flow_engine.get_next_prompt(
                    request.conversation_id, intent_str, entities
                )
                if flow_prompt:
                    system_prompt += f"\n\nIMPORTANT: Ask the customer: {flow_prompt}"

            conversation_history = self._format_conversation_history(request.conversation_id)

            # Context window management (Gap 9.3)
            managed_history, managed_context, managed_sources = self._manage_context_window(
                system_prompt=system_prompt,
                user_prompt=request.message,
                conversation_history=conversation_history if turns > 1 else "",
                context_text=context.text,
                context_sources=context.sources,
            )

            user_prompt = PromptTemplates.build_rag_prompt(
                query=request.message,
                context=managed_context,
                conversation_history=managed_history if turns > 1 else None
            )

            # Apply experiment overrides to LLM parameters
            llm_temperature = experiment_overrides.get("temperature", self.temperature)
            llm_max_tokens = experiment_overrides.get("max_tokens", self.max_tokens)

            # Step 8: Generate LLM response
            response_text = await self._generate_response(system_prompt, user_prompt)

            # Step 8a: Response verification / guardrails
            if self._response_verifier:
                verification = self._response_verifier.verify(
                    response=response_text,
                    context_text=managed_context,
                    query=request.message,
                )
                response_text = verification.sanitized_response

            # Step 8b: Check handoff trigger
            if self._handoff_manager:
                confidence = getattr(intent_result, 'confidence', 1.0) if intent_result else 1.0
                trigger = self._handoff_manager.check_trigger(
                    conversation_id=request.conversation_id,
                    message=request.message,
                    intent=intent_str,
                    confidence=confidence,
                    lead_score=0.0,
                )
                if trigger:
                    session = self._handoff_manager.initiate_handoff(request.conversation_id, trigger)
                    response_text += (
                        "\n\nI'm connecting you with a human agent who can assist you better. "
                        "Please hold on, someone will be with you shortly."
                    )

            # Step 9: Score lead (cumulative or single-message)
            lead_score = None
            lead_priority = None
            cumulative_data = None
            if self.lead_scorer and intent_result and entities:
                if self._cumulative_scorer:
                    cum_result = self._cumulative_scorer.score(
                        message=request.message,
                        conversation_id=request.conversation_id,
                        intent_result=intent_result,
                        entities=entities,
                    )
                    lead_score = round(cum_result.cumulative_score)
                    lead_priority = cum_result.temperature
                    cumulative_data = cum_result.to_dict()
                else:
                    score_result = self.lead_scorer.score(
                        intent_result=intent_result,
                        entities=entities,
                        conversation_turns=turns
                    )
                    lead_score = score_result.score
                    lead_priority = score_result.priority.value

                # Step 10: Route lead if qualified
                # Map temperature/priority to lead_scoring.scoring_model.LeadPriority
                from lead_scoring.scoring_model import LeadPriority
                priority_map = {"hot": LeadPriority.HOT, "warm": LeadPriority.WARM, "cold": LeadPriority.COLD}
                resolved_priority = priority_map.get(lead_priority, LeadPriority.COLD)

                if self.lead_router and self.lead_router.should_route(
                    type("Lead", (), {"score": lead_score, "priority": resolved_priority})()
                ):
                    try:
                        from lead_scoring.lead_router import Lead as CRMLead
                        lead_obj = CRMLead(
                            lead_id=str(uuid.uuid4()),
                            conversation_id=request.conversation_id,
                            score=lead_score,
                            priority=resolved_priority,
                            name=entities.name,
                            phone=entities.phone_numbers[0] if entities.phone_numbers else None,
                            email=entities.emails[0] if entities.emails else None,
                            city=entities.city,
                            primary_intent=intent_str,
                            models_interested=entities.models_mentioned,
                            budget_min=entities.budget_min,
                            budget_max=entities.budget_max,
                            timeline=intent_result.timeline.value if intent_result.timeline else None,
                            last_message=request.message,
                            conversation_turns=turns,
                        )
                        await self.lead_router.route_lead(lead_obj)
                        logger.info(f"Lead routed: {lead_obj.lead_id} (score={lead_score})")
                    except Exception as e:
                        logger.error(f"Lead routing failed: {e}")

            # Sentiment analysis for feedback messages
            sentiment_data = None
            if intent_result and intent_result.primary_intent.value == "feedback":
                sentiment_data = await self._analyze_sentiment(request.message)

            # Update conversation stage
            if intent_result:
                self._update_stage(request.conversation_id, intent_result.primary_intent.value)

            # Build sources
            sources = []
            if request.include_sources:
                sources = [
                    {
                        "id": s.id,
                        "source": s.source,
                        "score": s.score,
                        "preview": s.text_preview[:150] + "..." if len(s.text_preview) > 150 else s.text_preview
                    }
                    for s in context.sources[:5]
                ]

            # Build suggested actions
            suggested_actions = self._get_suggested_actions(intent_result, entities, turns)

            # Update conversation with response
            self._update_conversation(request.conversation_id, "assistant", response_text)

            processing_time = (time.time() - start_time) * 1000

            return ChatResponse(
                response=response_text,
                conversation_id=request.conversation_id,
                query=request.message,
                sources=sources,
                lead_score=lead_score,
                lead_priority=lead_priority,
                intent=intent_str,
                suggested_actions=suggested_actions,
                processing_time_ms=round(processing_time, 2),
                metadata={
                    "chunks_used": context.chunk_count,
                    "prompt_type": prompt_type.value,
                    "conversation_turns": turns,
                    "conversation_stage": self._conversation_stages.get(request.conversation_id, "enquiry"),
                    "sentiment": sentiment_data,
                    "entities_extracted": entities.to_dict() if entities else None,
                    "reranked": self.reranker is not None,
                    "cumulative_scoring": cumulative_data,
                    "experiment_overrides": experiment_overrides or None,
                    "handoff_active": self._handoff_manager.is_in_handoff(request.conversation_id) if self._handoff_manager else False,
                }
            )

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            raise

    async def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using LLM."""
        if self.llm_provider == LLMProvider.BEDROCK_CLAUDE:
            return await self._generate_bedrock(system_prompt, user_prompt)
        elif self.llm_provider == LLMProvider.OPENAI:
            return await self._generate_openai(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    async def _generate_bedrock(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Bedrock Claude."""
        try:
            response = self._llm_client.invoke_model(
                modelId=self.llm_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}]
                        }
                    ]
                }),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            if "content" in response_body and response_body["content"]:
                return response_body["content"][0]["text"].strip()

            return "I apologize, but I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Bedrock generation failed: {e}")
            raise

    async def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI."""
        try:
            response = self._llm_client.chat.completions.create(
                model=self.llm_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def set_feature_flags(self, flags: Dict[str, Any]):
        """Set feature flags for toggling optimizations at runtime."""
        self._feature_flags = flags

    def set_service_config(self, config: Dict[str, Any]):
        """Set service configuration (RM details, escalation contacts, etc.)."""
        self._service_config = config

    def set_notification_callback(self, callback: Callable):
        """Set callback for outbound notifications."""
        self._notification_callback = callback

    def set_namespace_router(self, router: Any):
        """Set the namespace router for smart namespace selection."""
        self.namespace_router = router

    def set_query_preprocessor(self, preprocessor: Any):
        """Set the query preprocessor for embedding optimization."""
        self.query_preprocessor = preprocessor

    def set_query_decomposer(self, decomposer: Any):
        """Set the query decomposer for multi-facet queries."""
        self.query_decomposer = decomposer

    def set_response_verifier(self, verifier: Any):
        """Set the response verifier / guardrails."""
        self._response_verifier = verifier

    def set_cumulative_scorer(self, scorer: Any):
        """Set the cumulative lead scorer."""
        self._cumulative_scorer = scorer

    def set_handoff_manager(self, manager: Any):
        """Set the human handoff manager."""
        self._handoff_manager = manager

    def set_experiment_engine(self, engine: Any):
        """Set the A/B testing experiment engine."""
        self._experiment_engine = engine

    def set_flow_engine(self, engine: Any):
        """Set the conversation flow engine."""
        self._flow_engine = engine

    def set_token_estimator(self, estimator: Any):
        """Set the token estimator for context window management."""
        self._token_estimator = estimator

    def _build_metadata_filter(self, entities: Optional[Any]) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone metadata filter from extracted entities for hybrid search.

        Returns None if no useful filters can be constructed.
        """
        if not entities:
            return None

        filters = {}

        models = getattr(entities, "models_mentioned", None) or []
        if models:
            filters["model"] = {"$in": [m.lower() for m in models]}

        fuel = getattr(entities, "fuel_type", None)
        if fuel:
            filters["fuel_type"] = fuel.lower()

        body_type = getattr(entities, "body_type", None)
        if body_type:
            filters["body_type"] = body_type.lower()

        if not filters:
            return None

        # If multiple filters, wrap in $and
        if len(filters) > 1:
            return {"$and": [{k: v} for k, v in filters.items()]}
        return filters

    def _get_adaptive_threshold(self, intent_result: Optional[Any]) -> float:
        """
        Return a dynamic similarity threshold based on intent confidence.

        High-confidence specific intents get stricter thresholds (better precision).
        Low-confidence / vague queries get relaxed thresholds (better recall).
        """
        if not intent_result:
            return 0.25  # No intent → very permissive

        confidence = getattr(intent_result, "confidence", 0.5)
        intent_value = intent_result.primary_intent.value

        # High-value specific intents with good confidence
        high_value_intents = {"buy", "finance", "insurance", "booking_confirm", "payment_confirm"}
        if intent_value in high_value_intents and confidence > 0.7:
            return 0.45

        # Medium confidence
        if confidence > 0.5:
            return 0.35

        # Low confidence / vague
        return 0.25

    def _build_retrieval_query(self, message: str, conversation_id: str) -> str:
        """
        Enrich query with conversation context for better retrieval.

        Detects anaphoric references (pronouns, short messages) and
        prepends recent conversation context to the embedding query.

        Args:
            message: Current user message (may be preprocessed)
            conversation_id: Conversation ID for history lookup

        Returns:
            Enriched query string for embedding
        """
        # Anaphoric indicators that need context
        ANAPHORIC_WORDS = {
            "it", "this", "that", "these", "those", "one",
            "its", "them", "same", "other", "another",
            # Hindi
            "uska", "iska", "woh", "yeh", "wo", "ye",
            "uski", "iski", "unka", "inka",
        }

        words = set(message.lower().split())
        is_short = len(message.split()) < 5
        has_anaphora = bool(words & ANAPHORIC_WORDS)

        if not (is_short or has_anaphora):
            return message

        # Get recent user messages for context
        history = self._conversations.get(conversation_id, [])
        if not history:
            return message

        # Collect last 2 user messages (excluding current)
        recent_user_msgs = []
        for msg in reversed(history[:-1]):  # exclude current
            if msg["role"] == "user":
                recent_user_msgs.append(msg["content"])
                if len(recent_user_msgs) >= 2:
                    break

        if not recent_user_msgs:
            return message

        # Build enriched query: recent context + current message
        context_str = " ".join(reversed(recent_user_msgs))
        enriched = f"{context_str} {message}"

        # Cap at ~200 words to avoid embedding degradation
        words_list = enriched.split()
        if len(words_list) > 200:
            enriched = " ".join(words_list[-200:])

        logger.debug(f"Conversation-aware query: '{message}' → '{enriched[:100]}...'")
        return enriched

    def _get_language_instruction(self, message: str) -> Optional[str]:
        """Detect language and return instruction for LLM to respond in same language."""
        # Check for Devanagari script (Hindi, Marathi)
        if re.search(r'[\u0900-\u097F]', message):
            return "IMPORTANT: The customer is writing in Hindi/Marathi. Respond in the same language using Devanagari script."
        # Check for Gujarati script
        if re.search(r'[\u0A80-\u0AFF]', message):
            return "IMPORTANT: The customer is writing in Gujarati. Respond in Gujarati."
        # Check for transliterated Hindi keywords
        hindi_indicators = ["mujhe", "kya", "chahiye", "gaadi", "kitna", "kab", "kaise", "hai", "hain", "nahi", "acha"]
        words = message.lower().split()
        hindi_count = sum(1 for w in words if w in hindi_indicators)
        if hindi_count >= 2:
            return "IMPORTANT: The customer is writing in Hindi (Roman script). Respond in Hindi using Roman script (Hinglish)."
        return None

    async def _analyze_sentiment(self, message: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of a customer message using LLM."""
        try:
            prompt = PromptTemplates.get_user_prompt("sentiment_analysis", message=message)
            system = "You are a sentiment analysis engine. Return only valid JSON."
            response = await self._generate_response(system, prompt)

            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        return None

    def _update_stage(self, conversation_id: str, intent_value: str):
        """Update conversation stage based on intent."""
        stage_map = {
            "buy": "enquiry",
            "booking_confirm": "booked",
            "payment_confirm": "booked",
            "delivery_update": "delivery",
            "service": "servicing",
            "service_reminder": "servicing",
            "feedback": "feedback",
        }
        new_stage = stage_map.get(intent_value)
        if new_stage:
            self._conversation_stages[conversation_id] = new_stage

    def _update_conversation(self, conversation_id: str, role: str, content: str):
        """Update conversation history."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
            self._conversation_turns[conversation_id] = 0
            self._conversation_created[conversation_id] = datetime.utcnow()
            self._conversation_stages[conversation_id] = "enquiry"

        self._conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

        if role == "user":
            self._conversation_turns[conversation_id] += 1

        # Keep only last 20 messages
        if len(self._conversations[conversation_id]) > 20:
            self._conversations[conversation_id] = self._conversations[conversation_id][-20:]

    def _format_conversation_history(self, conversation_id: str) -> str:
        """Format conversation history for prompt."""
        if conversation_id not in self._conversations:
            return ""

        history = self._conversations[conversation_id][:-1]  # Exclude current message
        if not history:
            return ""

        formatted = []
        for msg in history[-6:]:  # Last 6 messages
            role = "Customer" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def _manage_context_window(
        self,
        system_prompt: str,
        user_prompt: str,
        conversation_history: str,
        context_text: str,
        context_sources: List[Any],
    ) -> tuple:
        """
        Manage context window to fit within model limits (Gap 9.3).

        Priority-based truncation:
        1. System prompt (never truncate)
        2. User message (never truncate)
        3. RAG context (drop lowest-scored chunks)
        4. Conversation history (drop oldest messages)

        Args:
            system_prompt: System prompt text
            user_prompt: User prompt (includes RAG context + history)
            conversation_history: Formatted conversation history
            context_text: RAG context text
            context_sources: Source objects for context chunks

        Returns:
            (truncated_history, truncated_context_text, truncated_sources)
        """
        if not self._token_estimator:
            # No estimator, return as-is
            return conversation_history, context_text, context_sources

        budget = self._context_limits.get(self.llm_provider, 100_000)

        # Estimate tokens for each component
        system_tokens = self._token_estimator.estimate(system_prompt)
        user_msg_tokens = self._token_estimator.estimate(user_prompt)
        history_tokens = self._token_estimator.estimate(conversation_history) if conversation_history else 0
        context_tokens = self._token_estimator.estimate(context_text) if context_text else 0
        response_budget = self.max_tokens

        total = system_tokens + user_msg_tokens + history_tokens + context_tokens + response_budget

        if total <= budget:
            return conversation_history, context_text, context_sources

        # Need to truncate
        overflow = total - budget
        logger.info(f"Context window overflow: {total} tokens > {budget} budget, trimming {overflow} tokens")

        # Step 1: Trim conversation history (oldest first)
        if overflow > 0 and conversation_history:
            lines = conversation_history.split("\n")
            while overflow > 0 and len(lines) > 1:
                removed = lines.pop(0)
                removed_tokens = self._token_estimator.estimate(removed)
                overflow -= removed_tokens
                history_tokens -= removed_tokens
            conversation_history = "\n".join(lines)

        # Step 2: Trim RAG context (drop lowest-scored chunks)
        if overflow > 0 and context_sources:
            # Sort by score ascending (remove lowest first)
            indexed = list(enumerate(context_sources))
            indexed.sort(key=lambda x: getattr(x[1], 'score', 0))
            removed_indices = set()
            for idx, source in indexed:
                if overflow <= 0:
                    break
                chunk_text = getattr(source, 'text', '') or getattr(source, 'text_preview', '')
                chunk_tokens = self._token_estimator.estimate(chunk_text)
                removed_indices.add(idx)
                overflow -= chunk_tokens

            if removed_indices:
                context_sources = [s for i, s in enumerate(context_sources) if i not in removed_indices]
                # Rebuild context_text from remaining sources
                remaining_texts = []
                for s in context_sources:
                    txt = getattr(s, 'text', '') or getattr(s, 'text_preview', '')
                    remaining_texts.append(txt)
                context_text = "\n\n---\n\n".join(remaining_texts)

        return conversation_history, context_text, context_sources

    def _get_suggested_actions(
        self,
        intent_result: Any,
        entities: Any,
        turns: int
    ) -> List[str]:
        """Get suggested actions based on conversation state."""
        actions = []

        if intent_result:
            intent = intent_result.primary_intent.value

            if intent == "buy":
                actions.append("Book test drive")
                actions.append("Get price quote")
            elif intent == "finance":
                actions.append("Calculate EMI")
                actions.append("Check loan eligibility")
            elif intent == "test_drive":
                actions.append("Schedule test drive")
                actions.append("Home test drive")
            elif intent == "service" or intent == "service_reminder":
                actions.append("Book service")
                actions.append("Find service center")
            elif intent == "feedback":
                actions.append("Rate your experience")
                actions.append("Talk to manager")
            elif intent == "escalation" or intent == "complaint":
                actions.append("View escalation contacts")
                actions.append("Request callback")
            elif intent == "delivery_update":
                actions.append("Check delivery status")
                actions.append("Contact sales team")
            elif intent == "booking_confirm" or intent == "payment_confirm":
                actions.append("View booking details")
                actions.append("Download receipt")

        # General actions based on conversation state
        if turns >= 3 and not entities:
            actions.append("Talk to sales executive")

        if entities and entities.models_mentioned:
            actions.append("View vehicle details")
            actions.append("Compare models")

        return actions[:4]  # Limit to 4 actions

    def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._conversations.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str):
        """Clear a conversation."""
        self._conversations.pop(conversation_id, None)
        self._conversation_turns.pop(conversation_id, None)
        self._conversation_stages.pop(conversation_id, None)
        self._conversation_created.pop(conversation_id, None)
