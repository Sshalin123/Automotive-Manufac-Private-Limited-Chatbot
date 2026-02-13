"""
Chunking Strategies for AMPL Chatbot.

Provides different chunking strategies for various document types.
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk."""
    content: str
    index: int
    metadata: Dict[str, Any]


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of Chunk objects
        """
        pass


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking with overlap.

    Best for: Technical specifications, structured data
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        min_chunk_size: int = 100
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (discard smaller chunks)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        text = text.strip()
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at sentence boundary
            if end < len(text):
                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_punct = text.rfind(punct, start + self.chunk_size // 2, end)
                    if last_punct > start:
                        end = last_punct + len(punct)
                        break

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = {
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_size": len(chunk_text),
                    **(metadata or {})
                }

                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    metadata=chunk_metadata
                ))
                index += 1

            start = end - self.overlap

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking based on document structure.

    Best for: Narrative content, articles, descriptions
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        respect_paragraphs: bool = True,
        respect_headers: bool = True
    ):
        """
        Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size
            respect_paragraphs: Keep paragraphs together when possible
            respect_headers: Start new chunks at headers
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_paragraphs = respect_paragraphs
        self.respect_headers = respect_headers

        # Header patterns
        self.header_pattern = re.compile(
            r'^(?:#{1,6}\s|[A-Z][^.!?]*:$|\d+\.\s|[A-Z]{2,}[A-Z\s]*$)',
            re.MULTILINE
        )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into semantic chunks."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        text = text.strip()

        # Split into sections based on headers if enabled
        if self.respect_headers:
            sections = self._split_by_headers(text)
        else:
            sections = [text]

        chunks = []
        index = 0

        for section in sections:
            # Split section into paragraphs
            if self.respect_paragraphs:
                paragraphs = self._split_into_paragraphs(section)
            else:
                paragraphs = [section]

            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If adding this paragraph exceeds max size, save current chunk
                if len(current_chunk) + len(para) + 1 > self.max_chunk_size:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunk_metadata = {
                            "chunk_index": index,
                            "chunk_size": len(current_chunk),
                            **(metadata or {})
                        }
                        chunks.append(Chunk(
                            content=current_chunk.strip(),
                            index=index,
                            metadata=chunk_metadata
                        ))
                        index += 1

                    # Start new chunk with current paragraph
                    current_chunk = para
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # Don't forget the last chunk in the section
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                chunk_metadata = {
                    "chunk_index": index,
                    "chunk_size": len(current_chunk),
                    **(metadata or {})
                }
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    index=index,
                    metadata=chunk_metadata
                ))
                index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by header patterns."""
        # Find all header positions
        matches = list(self.header_pattern.finditer(text))

        if not matches:
            return [text]

        sections = []
        prev_end = 0

        for match in matches:
            if match.start() > prev_end:
                section = text[prev_end:match.start()].strip()
                if section:
                    sections.append(section)
            prev_end = match.start()

        # Add remaining text
        if prev_end < len(text):
            section = text[prev_end:].strip()
            if section:
                sections.append(section)

        return sections if sections else [text]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]


class TableAwareChunker(ChunkingStrategy):
    """
    Chunking strategy that preserves tables.

    Best for: Documents with pricing tables, comparison charts
    """

    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize table-aware chunker.

        Args:
            max_chunk_size: Maximum chunk size
        """
        self.max_chunk_size = max_chunk_size

        # Table patterns (markdown and ASCII tables)
        self.table_pattern = re.compile(
            r'(?:\|[^\n]+\|[\r\n]+)+|'  # Markdown tables
            r'(?:[+\-]+[+\-]+[\r\n]+)+|'  # ASCII box tables
            r'(?:\s*\w+\s*:\s*[^\n]+[\r\n]+){3,}'  # Key-value pairs
        )

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text while preserving tables."""
        if not text or len(text.strip()) < 50:
            return []

        text = text.strip()
        chunks = []
        index = 0

        # Find all tables
        tables = list(self.table_pattern.finditer(text))

        if not tables:
            # No tables found, use semantic chunking
            semantic = SemanticChunker(max_chunk_size=self.max_chunk_size)
            return semantic.chunk(text, metadata)

        prev_end = 0

        for table_match in tables:
            # Process text before table
            pre_text = text[prev_end:table_match.start()].strip()
            if pre_text and len(pre_text) >= 50:
                chunk_metadata = {
                    "chunk_index": index,
                    "chunk_type": "text",
                    "chunk_size": len(pre_text),
                    **(metadata or {})
                }
                chunks.append(Chunk(
                    content=pre_text,
                    index=index,
                    metadata=chunk_metadata
                ))
                index += 1

            # Process table
            table_text = table_match.group().strip()
            if table_text:
                chunk_metadata = {
                    "chunk_index": index,
                    "chunk_type": "table",
                    "chunk_size": len(table_text),
                    **(metadata or {})
                }
                chunks.append(Chunk(
                    content=table_text,
                    index=index,
                    metadata=chunk_metadata
                ))
                index += 1

            prev_end = table_match.end()

        # Process remaining text after last table
        post_text = text[prev_end:].strip()
        if post_text and len(post_text) >= 50:
            chunk_metadata = {
                "chunk_index": index,
                "chunk_type": "text",
                "chunk_size": len(post_text),
                **(metadata or {})
            }
            chunks.append(Chunk(
                content=post_text,
                index=index,
                metadata=chunk_metadata
            ))

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks


class VehicleSpecChunker(ChunkingStrategy):
    """
    Specialized chunker for vehicle specification documents.

    Best for: Vehicle spec sheets, comparison documents
    """

    def __init__(self):
        """Initialize vehicle spec chunker."""
        self.spec_sections = [
            "engine",
            "performance",
            "dimensions",
            "safety",
            "features",
            "comfort",
            "exterior",
            "interior",
            "price",
            "warranty",
        ]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split vehicle specs into logical sections."""
        if not text or len(text.strip()) < 50:
            return []

        text = text.strip()
        chunks = []
        index = 0

        # Try to find section headers
        current_section = "general"
        current_content = []

        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this line is a section header
            new_section = None
            for section in self.spec_sections:
                if section in line_lower and len(line_lower) < 50:
                    new_section = section
                    break

            if new_section:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content) >= 50:
                        chunk_metadata = {
                            "chunk_index": index,
                            "spec_section": current_section,
                            "chunk_size": len(content),
                            **(metadata or {})
                        }
                        chunks.append(Chunk(
                            content=content,
                            index=index,
                            metadata=chunk_metadata
                        ))
                        index += 1

                current_section = new_section
                current_content = [line]
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if len(content) >= 50:
                chunk_metadata = {
                    "chunk_index": index,
                    "spec_section": current_section,
                    "chunk_size": len(content),
                    **(metadata or {})
                }
                chunks.append(Chunk(
                    content=content,
                    index=index,
                    metadata=chunk_metadata
                ))

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks


def get_chunker_for_document_type(document_type: str) -> ChunkingStrategy:
    """
    Get the appropriate chunker for a document type.

    Args:
        document_type: Type of document (inventory, sales, insurance, faq, etc.)

    Returns:
        Appropriate ChunkingStrategy instance
    """
    chunker_map = {
        "inventory": VehicleSpecChunker(),
        "vehicle": VehicleSpecChunker(),
        "specs": VehicleSpecChunker(),
        "sales": SemanticChunker(max_chunk_size=1500),
        "brochure": SemanticChunker(max_chunk_size=1500),
        "insurance": FixedSizeChunker(chunk_size=1000, overlap=100),
        "finance": FixedSizeChunker(chunk_size=1000, overlap=100),
        "faq": FixedSizeChunker(chunk_size=800, overlap=50),
        "pricing": TableAwareChunker(max_chunk_size=1500),
        "comparison": TableAwareChunker(max_chunk_size=1500),
    }

    return chunker_map.get(document_type, SemanticChunker())
