"""
Sales Documents Loader for AMPL Chatbot.

Handles loading and processing sales brochures, promotional materials,
and offer documents.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import re

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class SalesDocument:
    """Represents a processed sales document ready for embedding."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Offer:
    """Represents a promotional offer."""
    title: str
    description: str
    discount_type: str  # percentage, flat, emi, exchange
    discount_value: Optional[float] = None
    applicable_models: List[str] = field(default_factory=list)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    terms: List[str] = field(default_factory=list)


class SalesDocsLoader:
    """
    Loads and processes sales documents for RAG ingestion.

    Supports:
    - PDF brochures
    - Text documents
    - Offer sheets (structured)
    """

    def __init__(self, namespace: str = "sales"):
        """
        Initialize the sales document loader.

        Args:
            namespace: Pinecone namespace for sales documents
        """
        self.namespace = namespace
        self._documents: List[SalesDocument] = []

    def load_pdf(
        self,
        file_path: Union[str, Path],
        document_type: str = "brochure",
        extract_offers: bool = True
    ) -> List[SalesDocument]:
        """
        Load and process a PDF document.

        Args:
            file_path: Path to the PDF file
            document_type: Type of document (brochure, offer_sheet, spec_sheet)
            extract_offers: Whether to try extracting offer information

        Returns:
            List of SalesDocument objects ready for embedding
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents = []

        try:
            reader = PdfReader(str(file_path))

            # Process each page
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if not text or len(text.strip()) < 50:
                    continue

                # Clean and process text
                cleaned_text = self._clean_text(text)

                # Create document
                doc_id = self._generate_document_id(file_path, page_num)

                metadata = {
                    "source": str(file_path),
                    "document_type": document_type,
                    "page_number": page_num + 1,
                    "total_pages": len(reader.pages),
                    "file_name": file_path.name,
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                # Try to extract offers if applicable
                if extract_offers and document_type in ["offer_sheet", "brochure"]:
                    offers = self._extract_offers(cleaned_text)
                    if offers:
                        metadata["offers_found"] = len(offers)
                        metadata["offer_types"] = list(set(o.discount_type for o in offers))

                documents.append(SalesDocument(
                    id=doc_id,
                    content=cleaned_text,
                    metadata=metadata
                ))

            self._documents.extend(documents)
            logger.info(f"Loaded {len(documents)} pages from PDF: {file_path}")

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise

        return documents

    def load_text(
        self,
        file_path: Union[str, Path],
        document_type: str = "document"
    ) -> List[SalesDocument]:
        """
        Load and process a text document.

        Args:
            file_path: Path to the text file
            document_type: Type of document

        Returns:
            List of SalesDocument objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        cleaned_text = self._clean_text(content)

        # Split into chunks if document is large
        chunks = self._split_into_chunks(cleaned_text, max_size=1500, overlap=150)

        documents = []
        for idx, chunk in enumerate(chunks):
            doc_id = self._generate_document_id(file_path, idx)

            documents.append(SalesDocument(
                id=doc_id,
                content=chunk,
                metadata={
                    "source": str(file_path),
                    "document_type": document_type,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "file_name": file_path.name,
                    "ingested_at": datetime.utcnow().isoformat(),
                }
            ))

        self._documents.extend(documents)
        logger.info(f"Loaded {len(documents)} chunks from text file: {file_path}")

        return documents

    def load_offer(self, offer: Offer) -> SalesDocument:
        """
        Create a document from a structured offer.

        Args:
            offer: Offer object with offer details

        Returns:
            SalesDocument object
        """
        # Create rich content
        parts = [
            f"Special Offer: {offer.title}",
            f"\n{offer.description}",
        ]

        if offer.discount_type == "percentage" and offer.discount_value:
            parts.append(f"\nDiscount: {offer.discount_value}% off")
        elif offer.discount_type == "flat" and offer.discount_value:
            parts.append(f"\nDiscount: ₹{offer.discount_value:,.0f} off")
        elif offer.discount_type == "emi":
            parts.append(f"\nSpecial EMI Offer Available")
        elif offer.discount_type == "exchange":
            parts.append(f"\nExchange Bonus Available")

        if offer.applicable_models:
            parts.append(f"\nApplicable Models: {', '.join(offer.applicable_models)}")

        if offer.valid_from and offer.valid_until:
            parts.append(f"\nValid: {offer.valid_from.strftime('%d %b %Y')} to {offer.valid_until.strftime('%d %b %Y')}")
        elif offer.valid_until:
            parts.append(f"\nValid until: {offer.valid_until.strftime('%d %b %Y')}")

        if offer.terms:
            parts.append(f"\nTerms & Conditions:")
            for term in offer.terms:
                parts.append(f"  - {term}")

        content = "\n".join(parts)

        doc_id = hashlib.md5(f"offer:{offer.title}:{datetime.utcnow().isoformat()}".encode()).hexdigest()

        # Determine validity status
        now = datetime.utcnow()
        is_active = True
        if offer.valid_until and offer.valid_until < now:
            is_active = False
        if offer.valid_from and offer.valid_from > now:
            is_active = False

        document = SalesDocument(
            id=doc_id,
            content=content,
            metadata={
                "source": "offers",
                "document_type": "offer",
                "offer_title": offer.title,
                "discount_type": offer.discount_type,
                "discount_value": offer.discount_value,
                "applicable_models": offer.applicable_models,
                "is_active": is_active,
                "valid_from": offer.valid_from.isoformat() if offer.valid_from else None,
                "valid_until": offer.valid_until.isoformat() if offer.valid_until else None,
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

        self._documents.append(document)
        return document

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"₹$%&()/-]', '', text)

        # Fix common OCR issues
        text = text.replace('|', 'I')
        text = text.replace('0', 'O')  # Context-dependent, may need adjustment

        return text.strip()

    def _split_into_chunks(
        self,
        text: str,
        max_size: int = 1500,
        overlap: int = 150
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the boundary
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + max_size // 2:
                        end = last_punct + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def _extract_offers(self, text: str) -> List[Offer]:
        """Extract offer information from text using pattern matching."""
        offers = []

        # Pattern for percentage discounts
        pct_pattern = r'(\d+)%\s*(?:off|discount|savings?)'
        pct_matches = re.findall(pct_pattern, text, re.IGNORECASE)

        for match in pct_matches:
            offers.append(Offer(
                title=f"{match}% Discount",
                description=f"Get {match}% off",
                discount_type="percentage",
                discount_value=float(match)
            ))

        # Pattern for flat discounts (in Lakhs or thousands)
        flat_pattern = r'(?:₹|Rs\.?)\s*([\d,]+)\s*(?:off|discount|savings?)'
        flat_matches = re.findall(flat_pattern, text, re.IGNORECASE)

        for match in flat_matches:
            value = float(match.replace(',', ''))
            offers.append(Offer(
                title=f"₹{value:,.0f} Off",
                description=f"Save ₹{value:,.0f}",
                discount_type="flat",
                discount_value=value
            ))

        # Pattern for EMI offers
        emi_pattern = r'(?:zero|0%?|low)\s*(?:down\s*payment|emi|interest)'
        if re.search(emi_pattern, text, re.IGNORECASE):
            offers.append(Offer(
                title="Special EMI Offer",
                description="Easy EMI options available",
                discount_type="emi"
            ))

        # Pattern for exchange bonus
        exchange_pattern = r'exchange\s*(?:bonus|offer|benefit)'
        if re.search(exchange_pattern, text, re.IGNORECASE):
            offers.append(Offer(
                title="Exchange Bonus",
                description="Special exchange bonus on your old car",
                discount_type="exchange"
            ))

        return offers

    def _generate_document_id(self, file_path: Path, index: int) -> str:
        """Generate a unique document ID."""
        unique_string = f"{file_path}:{index}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def load_directory(
        self,
        directory: Union[str, Path],
        document_type: str = "document"
    ) -> List[SalesDocument]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to the directory
            document_type: Default document type

        Returns:
            List of all loaded SalesDocument objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_documents = []

        # Process PDFs
        for pdf_file in directory.glob("**/*.pdf"):
            try:
                docs = self.load_pdf(pdf_file, document_type=document_type)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load PDF {pdf_file}: {e}")

        # Process text files
        for txt_file in directory.glob("**/*.txt"):
            try:
                docs = self.load_text(txt_file, document_type=document_type)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load text file {txt_file}: {e}")

        logger.info(f"Loaded {len(all_documents)} documents from {directory}")
        return all_documents

    def get_documents(self) -> List[SalesDocument]:
        """Get all loaded documents."""
        return self._documents.copy()

    def clear(self):
        """Clear loaded documents."""
        self._documents.clear()
