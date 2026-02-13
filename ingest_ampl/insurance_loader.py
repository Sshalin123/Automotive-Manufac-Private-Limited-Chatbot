"""
Insurance and Finance Documents Loader for AMPL Chatbot.

Handles loading and processing insurance policies, financing options,
and EMI-related documents.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class InsuranceDocument:
    """Represents a processed insurance/finance document ready for embedding."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsurancePolicy:
    """Represents an insurance policy."""
    policy_name: str
    provider: str
    coverage_type: str  # comprehensive, third_party, zero_dep
    premium_range: Optional[tuple] = None  # (min, max)
    coverage_amount: Optional[float] = None
    key_features: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    claim_process: Optional[str] = None


@dataclass
class FinanceOption:
    """Represents a financing option."""
    name: str
    provider: str
    interest_rate: Optional[float] = None
    min_down_payment: Optional[float] = None
    max_tenure_months: int = 84
    processing_fee: Optional[float] = None
    eligibility: List[str] = field(default_factory=list)
    documents_required: List[str] = field(default_factory=list)


class InsuranceFinanceLoader:
    """
    Loads and processes insurance and finance documents for RAG ingestion.

    Supports:
    - Insurance policy documents (PDF, text)
    - Finance scheme documents
    - EMI calculator information
    - Terms and conditions
    """

    def __init__(self, namespace: str = "insurance"):
        """
        Initialize the insurance/finance loader.

        Args:
            namespace: Pinecone namespace for insurance documents
        """
        self.namespace = namespace
        self._documents: List[InsuranceDocument] = []

    def load_insurance_policy(self, policy: InsurancePolicy) -> InsuranceDocument:
        """
        Create a document from a structured insurance policy.

        Args:
            policy: InsurancePolicy object

        Returns:
            InsuranceDocument object
        """
        parts = [
            f"Insurance Policy: {policy.policy_name}",
            f"Provider: {policy.provider}",
            f"Coverage Type: {policy.coverage_type}",
        ]

        if policy.premium_range:
            parts.append(f"Premium Range: ₹{policy.premium_range[0]:,.0f} - ₹{policy.premium_range[1]:,.0f}")

        if policy.coverage_amount:
            parts.append(f"Coverage Amount: ₹{policy.coverage_amount:,.0f}")

        if policy.key_features:
            parts.append("\nKey Features:")
            for feature in policy.key_features:
                parts.append(f"  • {feature}")

        if policy.exclusions:
            parts.append("\nExclusions:")
            for exclusion in policy.exclusions:
                parts.append(f"  • {exclusion}")

        if policy.claim_process:
            parts.append(f"\nClaim Process: {policy.claim_process}")

        content = "\n".join(parts)
        doc_id = hashlib.md5(f"insurance:{policy.policy_name}:{policy.provider}".encode()).hexdigest()

        document = InsuranceDocument(
            id=doc_id,
            content=content,
            metadata={
                "source": "insurance",
                "document_type": "insurance_policy",
                "policy_name": policy.policy_name,
                "provider": policy.provider,
                "coverage_type": policy.coverage_type,
                "has_zero_dep": "zero" in policy.coverage_type.lower(),
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

        self._documents.append(document)
        return document

    def load_finance_option(self, option: FinanceOption) -> InsuranceDocument:
        """
        Create a document from a structured finance option.

        Args:
            option: FinanceOption object

        Returns:
            InsuranceDocument object
        """
        parts = [
            f"Finance Scheme: {option.name}",
            f"Finance Provider: {option.provider}",
        ]

        if option.interest_rate is not None:
            parts.append(f"Interest Rate: {option.interest_rate}% per annum")

        if option.min_down_payment is not None:
            parts.append(f"Minimum Down Payment: {option.min_down_payment}%")

        parts.append(f"Maximum Tenure: {option.max_tenure_months} months ({option.max_tenure_months // 12} years)")

        if option.processing_fee is not None:
            parts.append(f"Processing Fee: {option.processing_fee}%")

        if option.eligibility:
            parts.append("\nEligibility Criteria:")
            for item in option.eligibility:
                parts.append(f"  • {item}")

        if option.documents_required:
            parts.append("\nDocuments Required:")
            for doc in option.documents_required:
                parts.append(f"  • {doc}")

        content = "\n".join(parts)
        doc_id = hashlib.md5(f"finance:{option.name}:{option.provider}".encode()).hexdigest()

        document = InsuranceDocument(
            id=doc_id,
            content=content,
            metadata={
                "source": "finance",
                "document_type": "finance_option",
                "scheme_name": option.name,
                "provider": option.provider,
                "interest_rate": option.interest_rate,
                "max_tenure_months": option.max_tenure_months,
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

        self._documents.append(document)
        return document

    def load_pdf(
        self,
        file_path: Union[str, Path],
        document_type: str = "insurance"
    ) -> List[InsuranceDocument]:
        """
        Load and process a PDF document.

        Args:
            file_path: Path to the PDF file
            document_type: Type of document (insurance, finance, terms)

        Returns:
            List of InsuranceDocument objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        documents = []

        try:
            reader = PdfReader(str(file_path))

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if not text or len(text.strip()) < 50:
                    continue

                cleaned_text = self._clean_text(text)
                doc_id = self._generate_document_id(file_path, page_num)

                # Extract key information
                extracted_info = self._extract_insurance_info(cleaned_text)

                metadata = {
                    "source": str(file_path),
                    "document_type": document_type,
                    "page_number": page_num + 1,
                    "total_pages": len(reader.pages),
                    "file_name": file_path.name,
                    **extracted_info,
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                documents.append(InsuranceDocument(
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

    def load_emi_table(
        self,
        vehicle_price: float,
        interest_rates: List[float],
        tenures: List[int],
        down_payment_percentages: List[float]
    ) -> List[InsuranceDocument]:
        """
        Generate EMI calculation documents for different scenarios.

        Args:
            vehicle_price: Base vehicle price
            interest_rates: List of interest rates to calculate for
            tenures: List of tenure periods in months
            down_payment_percentages: List of down payment percentages

        Returns:
            List of InsuranceDocument objects with EMI calculations
        """
        documents = []

        for dp_pct in down_payment_percentages:
            for rate in interest_rates:
                for tenure in tenures:
                    # Calculate EMI
                    down_payment = vehicle_price * (dp_pct / 100)
                    loan_amount = vehicle_price - down_payment
                    monthly_rate = rate / 12 / 100

                    if monthly_rate > 0:
                        emi = loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure) / (((1 + monthly_rate) ** tenure) - 1)
                    else:
                        emi = loan_amount / tenure

                    total_payment = emi * tenure
                    total_interest = total_payment - loan_amount

                    content = f"""EMI Calculation
Vehicle Price: ₹{vehicle_price:,.0f}
Down Payment: {dp_pct}% (₹{down_payment:,.0f})
Loan Amount: ₹{loan_amount:,.0f}
Interest Rate: {rate}% per annum
Tenure: {tenure} months ({tenure // 12} years {tenure % 12} months)

Monthly EMI: ₹{emi:,.0f}
Total Payment: ₹{total_payment:,.0f}
Total Interest: ₹{total_interest:,.0f}

This EMI calculation is indicative. Actual EMI may vary based on credit score and lender terms."""

                    doc_id = hashlib.md5(
                        f"emi:{vehicle_price}:{dp_pct}:{rate}:{tenure}".encode()
                    ).hexdigest()

                    documents.append(InsuranceDocument(
                        id=doc_id,
                        content=content,
                        metadata={
                            "source": "emi_calculator",
                            "document_type": "emi_calculation",
                            "vehicle_price": vehicle_price,
                            "down_payment_percent": dp_pct,
                            "interest_rate": rate,
                            "tenure_months": tenure,
                            "emi_amount": round(emi, 2),
                            "ingested_at": datetime.utcnow().isoformat(),
                        }
                    ))

        self._documents.extend(documents)
        logger.info(f"Generated {len(documents)} EMI calculation documents")

        return documents

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'"₹$%&()/-]', '', text)
        return text.strip()

    def _extract_insurance_info(self, text: str) -> Dict[str, Any]:
        """Extract key insurance information from text."""
        info = {}

        # Extract coverage amounts
        coverage_pattern = r'(?:coverage|sum\s*insured|idv)\s*(?:of|:)?\s*(?:₹|Rs\.?)\s*([\d,]+)'
        coverage_match = re.search(coverage_pattern, text, re.IGNORECASE)
        if coverage_match:
            info["coverage_amount"] = float(coverage_match.group(1).replace(',', ''))

        # Extract premium
        premium_pattern = r'(?:premium|annual\s*premium)\s*(?:of|:)?\s*(?:₹|Rs\.?)\s*([\d,]+)'
        premium_match = re.search(premium_pattern, text, re.IGNORECASE)
        if premium_match:
            info["premium_amount"] = float(premium_match.group(1).replace(',', ''))

        # Check for zero depreciation
        if re.search(r'zero\s*dep(?:reciation)?', text, re.IGNORECASE):
            info["has_zero_dep"] = True

        # Check for roadside assistance
        if re.search(r'roadside\s*assist(?:ance)?', text, re.IGNORECASE):
            info["has_roadside_assistance"] = True

        return info

    def _generate_document_id(self, file_path: Path, index: int) -> str:
        """Generate a unique document ID."""
        unique_string = f"{file_path}:{index}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def get_documents(self) -> List[InsuranceDocument]:
        """Get all loaded documents."""
        return self._documents.copy()

    def clear(self):
        """Clear loaded documents."""
        self._documents.clear()
