"""
FAQ and Knowledge Base Loader for AMPL Chatbot.

Handles loading and processing FAQ documents, knowledge base articles,
and common questions.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import csv

logger = logging.getLogger(__name__)


@dataclass
class FAQDocument:
    """Represents a processed FAQ document ready for embedding."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FAQItem:
    """Represents a single FAQ item."""
    question: str
    answer: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority = more important


class FAQLoader:
    """
    Loads and processes FAQ and knowledge base content for RAG ingestion.

    Supports:
    - JSON FAQ files
    - CSV FAQ files
    - Markdown knowledge base articles
    - Individual FAQ items
    """

    # Common automotive FAQ categories
    CATEGORIES = [
        "pricing",
        "financing",
        "insurance",
        "test_drive",
        "service",
        "warranty",
        "features",
        "availability",
        "exchange",
        "documentation",
        "delivery",
        "general",
    ]

    def __init__(self, namespace: str = "faq"):
        """
        Initialize the FAQ loader.

        Args:
            namespace: Pinecone namespace for FAQ documents
        """
        self.namespace = namespace
        self._documents: List[FAQDocument] = []
        self._faq_items: List[FAQItem] = []

    def load_from_json(self, file_path: Union[str, Path]) -> List[FAQDocument]:
        """
        Load FAQs from a JSON file.

        Expected format:
        {
            "faqs": [
                {
                    "question": "...",
                    "answer": "...",
                    "category": "...",
                    "tags": ["...", "..."]
                }
            ]
        }

        Args:
            file_path: Path to the JSON file

        Returns:
            List of FAQDocument objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            faq_data = data
        elif isinstance(data, dict):
            faq_data = data.get("faqs", data.get("questions", data.get("items", [])))
        else:
            faq_data = []

        documents = []
        for item in faq_data:
            try:
                faq_item = FAQItem(
                    question=item.get("question", item.get("q", "")),
                    answer=item.get("answer", item.get("a", "")),
                    category=item.get("category", "general"),
                    tags=item.get("tags", []),
                    related_questions=item.get("related", []),
                    priority=item.get("priority", 0),
                )

                doc = self._create_document(faq_item)
                documents.append(doc)
                self._faq_items.append(faq_item)

            except Exception as e:
                logger.warning(f"Failed to parse FAQ item: {item}. Error: {e}")
                continue

        self._documents.extend(documents)
        logger.info(f"Loaded {len(documents)} FAQs from {file_path}")

        return documents

    def load_from_csv(self, file_path: Union[str, Path]) -> List[FAQDocument]:
        """
        Load FAQs from a CSV file.

        Expected columns: question, answer, category (optional), tags (optional)

        Args:
            file_path: Path to the CSV file

        Returns:
            List of FAQDocument objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Normalize column names
                    question = row.get("question", row.get("Question", row.get("Q", "")))
                    answer = row.get("answer", row.get("Answer", row.get("A", "")))
                    category = row.get("category", row.get("Category", "general"))
                    tags_str = row.get("tags", row.get("Tags", ""))

                    # Parse tags
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

                    if not question or not answer:
                        continue

                    faq_item = FAQItem(
                        question=question,
                        answer=answer,
                        category=category.lower().replace(" ", "_"),
                        tags=tags,
                    )

                    doc = self._create_document(faq_item)
                    documents.append(doc)
                    self._faq_items.append(faq_item)

                except Exception as e:
                    logger.warning(f"Failed to parse CSV row: {row}. Error: {e}")
                    continue

        self._documents.extend(documents)
        logger.info(f"Loaded {len(documents)} FAQs from {file_path}")

        return documents

    def add_faq(self, faq: FAQItem) -> FAQDocument:
        """
        Add a single FAQ item.

        Args:
            faq: FAQItem object

        Returns:
            FAQDocument object
        """
        doc = self._create_document(faq)
        self._documents.append(doc)
        self._faq_items.append(faq)
        return doc

    def add_faqs(self, faqs: List[FAQItem]) -> List[FAQDocument]:
        """
        Add multiple FAQ items.

        Args:
            faqs: List of FAQItem objects

        Returns:
            List of FAQDocument objects
        """
        documents = []
        for faq in faqs:
            doc = self.add_faq(faq)
            documents.append(doc)
        return documents

    def _create_document(self, faq: FAQItem) -> FAQDocument:
        """Create a FAQDocument from a FAQItem."""
        # Create content optimized for embedding
        content_parts = [
            f"Question: {faq.question}",
            f"\nAnswer: {faq.answer}",
        ]

        if faq.related_questions:
            content_parts.append("\nRelated Questions:")
            for related in faq.related_questions:
                content_parts.append(f"  - {related}")

        content = "\n".join(content_parts)

        # Generate unique ID
        doc_id = hashlib.md5(f"faq:{faq.question}".encode()).hexdigest()

        return FAQDocument(
            id=doc_id,
            content=content,
            metadata={
                "source": "faq",
                "document_type": "faq",
                "question": faq.question,
                "category": faq.category,
                "tags": faq.tags,
                "priority": faq.priority,
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

    def load_default_automotive_faqs(self) -> List[FAQDocument]:
        """
        Load a set of default automotive FAQs.

        Returns:
            List of FAQDocument objects
        """
        default_faqs = [
            # Pricing FAQs
            FAQItem(
                question="What is the on-road price of the car?",
                answer="The on-road price includes ex-showroom price plus registration, road tax, insurance, and other applicable charges. The exact on-road price varies by city and state. Please contact our sales team for the accurate on-road price in your location.",
                category="pricing",
                tags=["price", "on-road", "cost"],
                priority=10,
            ),
            FAQItem(
                question="What is included in the ex-showroom price?",
                answer="The ex-showroom price includes the cost of the vehicle, GST, and manufacturer's warranty. It does not include registration charges, road tax, insurance, or any accessories.",
                category="pricing",
                tags=["price", "ex-showroom", "cost"],
                priority=9,
            ),

            # Financing FAQs
            FAQItem(
                question="What financing options are available?",
                answer="We offer multiple financing options through leading banks and NBFCs. You can get loans with competitive interest rates starting from 7.5% per annum, tenure up to 7 years, and down payment as low as 10%. We also offer zero down payment schemes for select customers.",
                category="financing",
                tags=["finance", "loan", "emi", "bank"],
                priority=10,
            ),
            FAQItem(
                question="What is the EMI for this car?",
                answer="The EMI depends on the loan amount, interest rate, and tenure. For example, for a ₹10 lakh loan at 8% interest for 5 years, the EMI would be approximately ₹20,276. Our finance team can help you calculate the exact EMI based on your requirements.",
                category="financing",
                tags=["emi", "loan", "monthly payment"],
                priority=9,
            ),
            FAQItem(
                question="What documents are required for car loan?",
                answer="For a car loan, you typically need: 1) Identity proof (Aadhaar/PAN), 2) Address proof, 3) Income proof (salary slips for salaried, ITR for self-employed), 4) Bank statements (last 6 months), 5) Passport-size photographs. Additional documents may be required based on the lender.",
                category="financing",
                tags=["documents", "loan", "kyc"],
                priority=8,
            ),

            # Test Drive FAQs
            FAQItem(
                question="How can I book a test drive?",
                answer="You can book a test drive through our website, by calling our showroom, or by visiting us directly. We offer flexible test drive timings including weekends. We can also arrange a home test drive for your convenience.",
                category="test_drive",
                tags=["test drive", "booking", "demo"],
                priority=10,
            ),
            FAQItem(
                question="Is home test drive available?",
                answer="Yes, we offer home test drive facility. Our executive will bring the car to your preferred location at a convenient time. This service is complimentary. Please book in advance to ensure availability.",
                category="test_drive",
                tags=["test drive", "home", "doorstep"],
                priority=8,
            ),

            # Insurance FAQs
            FAQItem(
                question="What insurance options are available?",
                answer="We offer comprehensive insurance from leading insurers including third-party liability, own damage cover, and add-ons like zero depreciation, roadside assistance, and engine protection. Our team will help you choose the best coverage for your needs.",
                category="insurance",
                tags=["insurance", "coverage", "protection"],
                priority=9,
            ),
            FAQItem(
                question="What is zero depreciation insurance?",
                answer="Zero depreciation insurance covers the full cost of replacement parts without any deduction for depreciation. This is especially useful for new cars as it ensures you get the full claim amount for repairs.",
                category="insurance",
                tags=["insurance", "zero dep", "claim"],
                priority=8,
            ),

            # Service FAQs
            FAQItem(
                question="What is the service schedule for the car?",
                answer="We recommend servicing your car every 10,000 km or 1 year, whichever comes first. The first free service is typically at 1,000 km or 1 month. Our service centers are equipped with trained technicians and genuine parts.",
                category="service",
                tags=["service", "maintenance", "schedule"],
                priority=8,
            ),
            FAQItem(
                question="What is covered under warranty?",
                answer="Our vehicles come with a standard warranty of 2 years/40,000 km covering manufacturing defects. Extended warranty options are available up to 5 years. Warranty does not cover wear and tear items like tyres, brake pads, and wiper blades.",
                category="warranty",
                tags=["warranty", "guarantee", "coverage"],
                priority=9,
            ),

            # Exchange FAQs
            FAQItem(
                question="Do you accept old car exchange?",
                answer="Yes, we have a comprehensive exchange program. We offer free evaluation of your old car and provide the best exchange value. The exchange bonus can be adjusted against the new car purchase, making the upgrade more affordable.",
                category="exchange",
                tags=["exchange", "trade-in", "old car"],
                priority=9,
            ),

            # Delivery FAQs
            FAQItem(
                question="What is the delivery time for a new car?",
                answer="Delivery time varies by model and variant. Popular models are typically available within 2-4 weeks. Some variants may have a waiting period of 1-3 months. Please check with our sales team for the current delivery timeline.",
                category="delivery",
                tags=["delivery", "waiting period", "availability"],
                priority=8,
            ),
        ]

        return self.add_faqs(default_faqs)

    def get_documents(self) -> List[FAQDocument]:
        """Get all loaded documents."""
        return self._documents.copy()

    def get_faq_items(self) -> List[FAQItem]:
        """Get all loaded FAQ items."""
        return self._faq_items.copy()

    def get_by_category(self, category: str) -> List[FAQDocument]:
        """Get documents by category."""
        return [
            doc for doc in self._documents
            if doc.metadata.get("category") == category
        ]

    def clear(self):
        """Clear loaded documents."""
        self._documents.clear()
        self._faq_items.clear()
