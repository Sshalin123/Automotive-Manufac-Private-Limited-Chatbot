"""
Smart Namespace Router for AMPL Chatbot.

Maps intents and entities to relevant Pinecone namespaces,
avoiding blind queries across all namespaces.
"""

import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class NamespaceRouter:
    """
    Routes queries to relevant Pinecone namespaces based on
    classified intent and extracted entities.

    Instead of querying all 5 namespaces for every request,
    this targets 1-3 namespaces per query for better precision
    and reduced API calls.
    """

    # Intent-to-namespace mapping (order = priority)
    INTENT_NAMESPACE_MAP = {
        "buy":              ["inventory", "sales", "faq"],
        "finance":          ["sales", "faq"],
        "test_drive":       ["inventory", "sales", "faq"],
        "exchange":         ["inventory", "sales", "faq"],
        "insurance":        ["insurance", "faq"],
        "service":          ["faq", "sales"],
        "service_reminder": ["faq"],
        "complaint":        ["faq"],
        "escalation":       ["faq"],
        "info":             ["faq", "inventory", "sales"],
        "comparison":       ["inventory", "faq"],
        "booking_confirm":  ["sales", "faq"],
        "payment_confirm":  ["sales", "faq"],
        "delivery_update":  ["sales", "faq"],
        "feedback":         ["faq"],
    }

    def route(
        self,
        intent_result: Optional[Any] = None,
        entities: Optional[Any] = None,
    ) -> List[str]:
        """
        Determine which namespaces to search.

        Args:
            intent_result: Classified intent result
            entities: Extracted entities

        Returns:
            List of namespace strings to query
        """
        namespaces = set()

        # 1. Intent-based routing
        if intent_result:
            intent_value = intent_result.primary_intent.value
            mapped = self.INTENT_NAMESPACE_MAP.get(intent_value, [])
            namespaces.update(mapped)

        # 2. Entity-based augmentation
        if entities:
            # Model/variant mentioned → include inventory
            models = getattr(entities, "models_mentioned", None) or []
            variants = getattr(entities, "variants_mentioned", None) or []
            if models or variants:
                namespaces.add("inventory")

            # Insurance keywords → include insurance
            insurance_interest = getattr(entities, "insurance_interest", None)
            if insurance_interest:
                namespaces.add("insurance")

            # Budget/EMI → include sales
            budget_min = getattr(entities, "budget_min", None)
            budget_max = getattr(entities, "budget_max", None)
            if budget_min or budget_max:
                namespaces.add("sales")

        # 3. Always include faq as fallback
        namespaces.add("faq")

        # 4. If nothing resolved (no intent, no entities), query all
        if len(namespaces) <= 1:  # only faq
            return None  # signals to use all namespaces

        result = list(namespaces)
        logger.debug(f"Namespace routing: intent={getattr(intent_result, 'primary_intent', None)} → {result}")
        return result
