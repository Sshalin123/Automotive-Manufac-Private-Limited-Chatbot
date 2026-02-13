"""
Query Decomposer for AMPL Chatbot.

Decomposes complex multi-facet queries into focused sub-queries
for more granular vector search. Uses entity-based decomposition
(no LLM call) to keep latency low.
"""

import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    Decomposes multi-facet queries into focused sub-queries.

    Example:
        "diesel SUV under 15 lakh with sunroof" →
        [
            "diesel SUV under 15 lakh with sunroof",  # original always first
            "diesel SUV",
            "SUV under 15 lakh",
            "SUV with sunroof",
        ]

    Only activates when 3+ entity types are present, otherwise
    returns just the original query.
    """

    def __init__(self, max_sub_queries: int = 4):
        self.max_sub_queries = max_sub_queries

    def decompose(
        self,
        original_query: str,
        entities: Optional[Any] = None,
    ) -> List[str]:
        """
        Decompose a query into focused sub-queries based on entities.

        Args:
            original_query: The preprocessed query
            entities: Extracted entities

        Returns:
            List of sub-queries (original always first).
            Returns [original_query] if decomposition not needed.
        """
        if not entities:
            return [original_query]

        # Collect entity facets
        facets = []

        models = getattr(entities, "models_mentioned", None) or []
        if models:
            facets.append(("model", models[0]))

        variants = getattr(entities, "variants_mentioned", None) or []
        if variants:
            facets.append(("variant", variants[0]))

        fuel = getattr(entities, "fuel_type", None)
        if fuel:
            facets.append(("fuel", fuel))

        body_type = getattr(entities, "body_type", None)
        if body_type:
            facets.append(("body_type", body_type))

        budget_min = getattr(entities, "budget_min", None)
        budget_max = getattr(entities, "budget_max", None)
        if budget_min or budget_max:
            budget_str = ""
            if budget_max:
                facets.append(("budget", f"under {budget_max}"))
            elif budget_min:
                facets.append(("budget", f"above {budget_min}"))

        colors = getattr(entities, "colors_mentioned", None) or []
        if colors:
            facets.append(("color", colors[0]))

        features = getattr(entities, "features_mentioned", None) or []
        for feat in features[:2]:  # max 2 features
            facets.append(("feature", feat))

        # Only decompose if 3+ distinct facet types
        facet_types = {f[0] for f in facets}
        if len(facet_types) < 3:
            return [original_query]

        # Build sub-queries by combining 2 facets at a time
        sub_queries = [original_query]  # original always first

        # Get a base term (model or body_type or generic "car")
        base = models[0] if models else (body_type if body_type else "car")

        for facet_type, facet_value in facets:
            if facet_type in ("model", "body_type"):
                continue  # base is already covered
            sub_q = f"{base} {facet_value}"
            if sub_q not in sub_queries:
                sub_queries.append(sub_q)

        # Trim to max
        sub_queries = sub_queries[:self.max_sub_queries]

        if len(sub_queries) > 1:
            logger.debug(f"Query decomposed: '{original_query}' → {sub_queries}")

        return sub_queries
