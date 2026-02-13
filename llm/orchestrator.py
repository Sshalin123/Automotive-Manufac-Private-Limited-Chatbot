"""
Chat Orchestrator for AMPL Chatbot.

Orchestrates the full RAG pipeline from query to response.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
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
        self.llm_provider = llm_provider
        self.llm_model_id = llm_model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.brand_name = brand_name

        # Conversation state
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self._conversation_turns: Dict[str, int] = {}

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
            # Step 1: Update conversation state
            self._update_conversation(request.conversation_id, "user", request.message)
            turns = self._conversation_turns.get(request.conversation_id, 1)

            # Step 2: Classify intent (if classifier available)
            intent_result = None
            if self.intent_classifier:
                conversation_history = self._conversations.get(request.conversation_id, [])
                intent_result = self.intent_classifier.classify(
                    request.message,
                    conversation_history=conversation_history[-5:]
                )

            # Step 3: Extract entities (if extractor available)
            entities = None
            if self.entity_extractor:
                entities = self.entity_extractor.extract(request.message)

            # Step 4: Generate query embedding
            query_embedding = await self.embedding_service.embed_text(request.message)

            # Step 5: Retrieve relevant context
            search_results = self.pinecone_client.query_all_namespaces(
                embedding=query_embedding,
                top_k=request.max_chunks,
                min_score=0.5
            )

            # Step 6: Build context
            context = self.context_builder.build(
                results=search_results,
                query=request.message
            )

            # Step 7: Determine prompt type and build prompt
            intent_str = intent_result.primary_intent.value if intent_result else None
            prompt_type = PromptTemplates.detect_prompt_type(request.message, intent_str)

            system_prompt = PromptTemplates.get_system_prompt(
                prompt_type=prompt_type,
                brand_name=self.brand_name
            )

            conversation_history = self._format_conversation_history(request.conversation_id)
            user_prompt = PromptTemplates.build_rag_prompt(
                query=request.message,
                context=context.text,
                conversation_history=conversation_history if turns > 1 else None
            )

            # Step 8: Generate LLM response
            response_text = await self._generate_response(system_prompt, user_prompt)

            # Step 9: Score lead (if scorer available)
            lead_score = None
            lead_priority = None
            if self.lead_scorer and intent_result and entities:
                score_result = self.lead_scorer.score(
                    intent_result=intent_result,
                    entities=entities,
                    conversation_turns=turns
                )
                lead_score = score_result.score
                lead_priority = score_result.priority.value

                # Step 10: Route lead if qualified
                if self.lead_router and self.lead_router.should_route(
                    type("Lead", (), {"score": lead_score, "priority": score_result.priority})()
                ):
                    # Create and route lead (async)
                    pass  # Lead routing happens in background

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

    def _update_conversation(self, conversation_id: str, role: str, content: str):
        """Update conversation history."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
            self._conversation_turns[conversation_id] = 0

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
            elif intent == "service":
                actions.append("Book service")
                actions.append("Find service center")

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
