"""
Entity Extraction for AMPL Chatbot.

Extracts key entities from customer messages:
- Vehicle models mentioned
- Budget/price range
- Timeline
- Contact information
- Trade-in details
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Container for extracted entities from a message."""

    # Vehicle related
    models_mentioned: List[str] = field(default_factory=list)
    variants_mentioned: List[str] = field(default_factory=list)
    categories_mentioned: List[str] = field(default_factory=list)  # SUV, Sedan, etc.
    fuel_types_mentioned: List[str] = field(default_factory=list)  # Petrol, Diesel, etc.

    # Budget related
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    budget_mentioned: bool = False
    budget_text: Optional[str] = None

    # Timeline
    timeline_text: Optional[str] = None
    target_date: Optional[datetime] = None

    # Contact
    phone_numbers: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    # Trade-in
    trade_in_mentioned: bool = False
    trade_in_model: Optional[str] = None
    trade_in_year: Optional[int] = None

    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None

    # Features/requirements
    features_requested: List[str] = field(default_factory=list)
    color_preference: Optional[str] = None
    seating_requirement: Optional[int] = None

    # Need analysis (6-question profiling)
    purchase_mode: Optional[str] = None        # cash / finance
    annual_income: Optional[str] = None        # text range
    daily_travel_km: Optional[int] = None
    competitors_considered: List[str] = field(default_factory=list)
    top_priority: Optional[str] = None         # safety / mileage / features / comfort
    usage_pattern: Optional[str] = None        # daily commute / weekend / highway / city

    # Buyer classification
    buyer_type: Optional[str] = None           # first_time / replacement / additional
    customer_type: Optional[str] = None        # individual / corporate

    # Demographics
    gender: Optional[str] = None
    prefix: Optional[str] = None               # Dr / Mr / Mrs / Ms
    marital_status: Optional[str] = None
    age: Optional[int] = None

    # Feedback & NPS
    feedback_rating: Optional[str] = None      # poor / fair / very good / excellent
    nps_score: Optional[int] = None            # 0-10

    # Metadata
    extraction_confidence: float = 0.0
    raw_text: Optional[str] = None

    def has_contact_info(self) -> bool:
        """Check if any contact information was extracted."""
        return bool(self.phone_numbers or self.email_addresses)

    def has_budget(self) -> bool:
        """Check if budget information was extracted."""
        return self.budget_mentioned or self.budget_min is not None or self.budget_max is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models_mentioned": self.models_mentioned,
            "variants_mentioned": self.variants_mentioned,
            "categories_mentioned": self.categories_mentioned,
            "fuel_types_mentioned": self.fuel_types_mentioned,
            "budget_min": self.budget_min,
            "budget_max": self.budget_max,
            "budget_mentioned": self.budget_mentioned,
            "timeline_text": self.timeline_text,
            "phone_numbers": self.phone_numbers,
            "email_addresses": self.email_addresses,
            "trade_in_mentioned": self.trade_in_mentioned,
            "trade_in_model": self.trade_in_model,
            "city": self.city,
            "features_requested": self.features_requested,
            "color_preference": self.color_preference,
            "purchase_mode": self.purchase_mode,
            "annual_income": self.annual_income,
            "daily_travel_km": self.daily_travel_km,
            "competitors_considered": self.competitors_considered,
            "top_priority": self.top_priority,
            "usage_pattern": self.usage_pattern,
            "buyer_type": self.buyer_type,
            "customer_type": self.customer_type,
            "gender": self.gender,
            "prefix": self.prefix,
            "marital_status": self.marital_status,
            "age": self.age,
            "feedback_rating": self.feedback_rating,
            "nps_score": self.nps_score,
        }


class EntityExtractor:
    """
    Extracts entities from customer messages.

    Uses pattern matching and optional NER for entity extraction.
    """

    # Vehicle categories
    VEHICLE_CATEGORIES = [
        "suv", "sedan", "hatchback", "crossover", "mpv", "muv",
        "coupe", "convertible", "wagon", "estate", "pickup", "truck",
    ]

    # Fuel types
    FUEL_TYPES = [
        "petrol", "diesel", "electric", "ev", "hybrid", "cng", "lpg",
        "plug-in hybrid", "phev", "bev", "mild hybrid",
    ]

    # Popular car brands in India
    BRANDS = [
        "maruti", "suzuki", "hyundai", "tata", "mahindra", "kia",
        "toyota", "honda", "mg", "skoda", "volkswagen", "renault",
        "nissan", "jeep", "citroen", "ford", "fiat", "bmw",
        "mercedes", "audi", "volvo", "jaguar", "land rover", "porsche",
    ]

    # Indian cities
    MAJOR_CITIES = [
        "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "kolkata",
        "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "kanpur",
        "nagpur", "indore", "thane", "bhopal", "visakhapatnam", "patna",
        "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad",
        "meerut", "rajkot", "varanasi", "srinagar", "aurangabad", "dhanbad",
        "amritsar", "navi mumbai", "allahabad", "ranchi", "gwalior", "jabalpur",
        "coimbatore", "vijayawada", "jodhpur", "madurai", "raipur", "kota",
    ]

    # Color names
    COLORS = [
        "white", "black", "silver", "grey", "gray", "red", "blue",
        "brown", "beige", "green", "orange", "yellow", "maroon",
        "pearl white", "midnight black", "fiery red", "arctic blue",
    ]

    # Features
    FEATURES = [
        "sunroof", "automatic", "manual", "leather seats", "cruise control",
        "android auto", "apple carplay", "reverse camera", "parking sensors",
        "airbags", "abs", "esp", "traction control", "alloy wheels",
        "led headlights", "touchscreen", "wireless charging", "ventilated seats",
        "panoramic sunroof", "360 camera", "adas", "adaptive cruise",
        "blind spot monitor", "lane assist", "heated seats", "cooled seats",
    ]

    def __init__(self, vehicle_models: Optional[List[str]] = None):
        """
        Initialize the entity extractor.

        Args:
            vehicle_models: Optional list of specific vehicle models to recognize
        """
        self.vehicle_models = vehicle_models or []

        # Build regex patterns
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for entity extraction."""
        # Phone number patterns (Indian format)
        self.phone_pattern = re.compile(
            r'(?:\+91[\s-]?)?'  # Optional +91 prefix
            r'(?:'
            r'\d{10}'  # 10 digits together
            r'|\d{5}[\s-]?\d{5}'  # 5-5 format
            r'|\d{4}[\s-]?\d{3}[\s-]?\d{3}'  # 4-3-3 format
            r'|\d{3}[\s-]?\d{3}[\s-]?\d{4}'  # 3-3-4 format
            r')'
        )

        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # Budget patterns (Indian currency format)
        self.budget_patterns = [
            # Lakh format: 10 lakh, 10L, 10 L, 10lacs
            re.compile(
                r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac|lacs|l)\b',
                re.IGNORECASE
            ),
            # Crore format: 1 crore, 1Cr
            re.compile(
                r'(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:crore|cr)\b',
                re.IGNORECASE
            ),
            # Direct amount: ₹1000000, Rs. 10,00,000
            re.compile(
                r'(?:₹|rs\.?|inr)\s*([\d,]+)(?:\.\d{2})?\b',
                re.IGNORECASE
            ),
            # Range: 10-15 lakh, between 10 and 15 lakh
            re.compile(
                r'(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac|lacs|l)\b',
                re.IGNORECASE
            ),
        ]

        # Year pattern (for trade-in)
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')

        # Pincode pattern
        self.pincode_pattern = re.compile(r'\b[1-9]\d{5}\b')

        # Seating pattern
        self.seating_pattern = re.compile(
            r'(\d+)\s*(?:seater|seat|seats|passenger)',
            re.IGNORECASE
        )

    def extract(self, message: str) -> ExtractedEntities:
        """
        Extract all entities from a message.

        Args:
            message: Customer message

        Returns:
            ExtractedEntities object with all extracted entities
        """
        entities = ExtractedEntities(raw_text=message)
        message_lower = message.lower()

        # Extract different entity types
        entities.phone_numbers = self._extract_phones(message)
        entities.email_addresses = self._extract_emails(message)

        budget_min, budget_max, budget_text = self._extract_budget(message)
        entities.budget_min = budget_min
        entities.budget_max = budget_max
        entities.budget_text = budget_text
        entities.budget_mentioned = budget_min is not None or budget_max is not None

        entities.categories_mentioned = self._extract_categories(message_lower)
        entities.fuel_types_mentioned = self._extract_fuel_types(message_lower)
        entities.models_mentioned = self._extract_models(message_lower)

        entities.city = self._extract_city(message_lower)
        pincode = self._extract_pincode(message)
        entities.pincode = pincode

        entities.color_preference = self._extract_color(message_lower)
        entities.seating_requirement = self._extract_seating(message_lower)
        entities.features_requested = self._extract_features(message_lower)

        trade_in, trade_model, trade_year = self._extract_trade_in(message_lower)
        entities.trade_in_mentioned = trade_in
        entities.trade_in_model = trade_model
        entities.trade_in_year = trade_year

        # Need analysis fields
        entities.purchase_mode = self._extract_purchase_mode(message_lower)
        entities.annual_income = self._extract_annual_income(message_lower)
        entities.daily_travel_km = self._extract_daily_travel(message_lower)
        entities.competitors_considered = self._extract_competitors(message_lower)
        entities.top_priority = self._extract_top_priority(message_lower)
        entities.usage_pattern = self._extract_usage_pattern(message_lower)

        # Buyer & customer classification
        entities.buyer_type = self._extract_buyer_type(message_lower)
        entities.customer_type = self._extract_customer_type(message_lower)

        # Demographics
        entities.prefix = self._extract_prefix(message_lower)
        entities.gender = self._extract_gender(message_lower)
        entities.marital_status = self._extract_marital_status(message_lower)
        entities.age = self._extract_age(message)

        # Feedback & NPS
        entities.feedback_rating = self._extract_feedback_rating(message_lower)
        entities.nps_score = self._extract_nps_score(message)

        # Calculate confidence based on entities found
        entities.extraction_confidence = self._calculate_confidence(entities)

        return entities

    def _extract_phones(self, message: str) -> List[str]:
        """Extract phone numbers from message."""
        matches = self.phone_pattern.findall(message)
        # Clean and normalize phone numbers
        phones = []
        for match in matches:
            # Remove spaces and dashes
            clean = re.sub(r'[\s-]', '', match)
            # Remove +91 prefix if present
            if clean.startswith('+91'):
                clean = clean[3:]
            if len(clean) == 10 and clean.isdigit():
                phones.append(clean)
        return list(set(phones))

    def _extract_emails(self, message: str) -> List[str]:
        """Extract email addresses from message."""
        return list(set(self.email_pattern.findall(message)))

    def _extract_budget(self, message: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract budget information from message.

        Returns:
            Tuple of (min_budget, max_budget, original_text)
        """
        # Check for range pattern first
        range_pattern = self.budget_patterns[3]
        range_match = range_pattern.search(message)

        if range_match:
            min_val = float(range_match.group(1)) * 100000  # Convert lakh to rupees
            max_val = float(range_match.group(2)) * 100000
            return min_val, max_val, range_match.group(0)

        # Check for single value patterns
        for pattern in self.budget_patterns[:3]:
            match = pattern.search(message)
            if match:
                value_str = match.group(1)
                # Remove commas for direct amount
                value_str = value_str.replace(',', '')
                value = float(value_str)

                # Convert based on unit
                match_text = match.group(0).lower()
                if 'crore' in match_text or 'cr' in match_text:
                    value *= 10000000  # Crore to rupees
                elif 'lakh' in match_text or 'lac' in match_text or match_text.endswith('l'):
                    value *= 100000  # Lakh to rupees

                # If no unit and value is small, assume lakh
                elif value < 1000:
                    value *= 100000

                return value, None, match.group(0)

        return None, None, None

    def _extract_categories(self, message_lower: str) -> List[str]:
        """Extract vehicle categories from message."""
        found = []
        for category in self.VEHICLE_CATEGORIES:
            if re.search(rf'\b{category}s?\b', message_lower):
                found.append(category)
        return found

    def _extract_fuel_types(self, message_lower: str) -> List[str]:
        """Extract fuel types from message."""
        found = []
        for fuel in self.FUEL_TYPES:
            if fuel in message_lower:
                found.append(fuel)
        return found

    def _extract_models(self, message_lower: str) -> List[str]:
        """Extract vehicle models from message."""
        found = []

        # Check known models
        for model in self.vehicle_models:
            if model.lower() in message_lower:
                found.append(model)

        # Check brand names
        for brand in self.BRANDS:
            if brand in message_lower:
                # Try to extract model name after brand
                pattern = rf'{brand}\s+(\w+)'
                match = re.search(pattern, message_lower)
                if match:
                    found.append(f"{brand.title()} {match.group(1).title()}")
                else:
                    found.append(brand.title())

        return list(set(found))

    def _extract_city(self, message_lower: str) -> Optional[str]:
        """Extract city from message."""
        for city in self.MAJOR_CITIES:
            if city in message_lower:
                return city.title()
        return None

    def _extract_pincode(self, message: str) -> Optional[str]:
        """Extract pincode from message."""
        match = self.pincode_pattern.search(message)
        return match.group(0) if match else None

    def _extract_color(self, message_lower: str) -> Optional[str]:
        """Extract color preference from message."""
        for color in self.COLORS:
            if color in message_lower:
                return color.title()
        return None

    def _extract_seating(self, message_lower: str) -> Optional[int]:
        """Extract seating requirement from message."""
        match = self.seating_pattern.search(message_lower)
        if match:
            return int(match.group(1))

        # Check for specific terms
        if '7 seater' in message_lower or 'seven seater' in message_lower:
            return 7
        if '5 seater' in message_lower or 'five seater' in message_lower:
            return 5
        if '6 seater' in message_lower or 'six seater' in message_lower:
            return 6

        return None

    def _extract_features(self, message_lower: str) -> List[str]:
        """Extract requested features from message."""
        found = []
        for feature in self.FEATURES:
            if feature in message_lower:
                found.append(feature)
        return found

    def _extract_trade_in(self, message_lower: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Extract trade-in information from message.

        Returns:
            Tuple of (trade_in_mentioned, model, year)
        """
        trade_keywords = ['exchange', 'trade', 'old car', 'sell my', 'part exchange']
        trade_mentioned = any(kw in message_lower for kw in trade_keywords)

        if not trade_mentioned:
            return False, None, None

        # Try to extract year
        year_match = self.year_pattern.search(message_lower)
        year = int(year_match.group(0)) if year_match else None

        # Try to extract model
        model = None
        for brand in self.BRANDS:
            if brand in message_lower:
                pattern = rf'{brand}\s+(\w+)'
                match = re.search(pattern, message_lower)
                if match:
                    model = f"{brand.title()} {match.group(1).title()}"
                break

        return True, model, year

    # ── Need Analysis Extractors ────────────────────────────────────

    def _extract_purchase_mode(self, message_lower: str) -> Optional[str]:
        """Extract purchase mode: cash or finance."""
        finance_kw = ["emi", "loan", "finance", "installment", "down payment", "bank loan"]
        cash_kw = ["cash", "full payment", "pay full", "one time payment", "lump sum"]
        if any(kw in message_lower for kw in finance_kw):
            return "finance"
        if any(kw in message_lower for kw in cash_kw):
            return "cash"
        return None

    def _extract_annual_income(self, message_lower: str) -> Optional[str]:
        """Extract annual income mention."""
        patterns = [
            re.compile(r'(?:income|salary|earning|ctc)\s*(?:is|of|around|about)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac|l|lpa)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|l|lpa)\s*(?:income|salary|per annum|annual)', re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message_lower)
            if match:
                return f"{match.group(1)} lakh per annum"
        return None

    def _extract_daily_travel(self, message_lower: str) -> Optional[int]:
        """Extract daily travel distance in km."""
        patterns = [
            re.compile(r'(\d+)\s*(?:km|kms|kilometer)\s*(?:daily|per day|a day|everyday)', re.IGNORECASE),
            re.compile(r'(?:travel|drive|commute)\s*(?:about|around)?\s*(\d+)\s*(?:km|kms)', re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message_lower)
            if match:
                return int(match.group(1))
        return None

    def _extract_competitors(self, message_lower: str) -> List[str]:
        """Extract competitor models being considered."""
        competitor_kw = ["considering", "looking at", "compared to", "vs", "versus", "or maybe", "also checking"]
        if not any(kw in message_lower for kw in competitor_kw):
            return []
        # Use existing model extraction to find mentioned models
        return self._extract_models(message_lower)

    def _extract_top_priority(self, message_lower: str) -> Optional[str]:
        """Extract customer's top priority."""
        priority_map = {
            "safety": ["safety", "airbag", "ncap", "crash", "safe"],
            "mileage": ["mileage", "fuel efficiency", "fuel economy", "kmpl", "km/l"],
            "features": ["features", "tech", "infotainment", "adas", "connected"],
            "comfort": ["comfort", "ride quality", "suspension", "space", "spacious"],
            "performance": ["performance", "power", "speed", "engine", "torque"],
            "price": ["affordable", "cheap", "value for money", "budget friendly"],
        }
        for priority, keywords in priority_map.items():
            if any(kw in message_lower for kw in keywords):
                return priority
        return None

    def _extract_usage_pattern(self, message_lower: str) -> Optional[str]:
        """Extract vehicle usage pattern."""
        pattern_map = {
            "daily_commute": ["daily commute", "office", "work travel", "daily use"],
            "weekend": ["weekend", "weekend drives", "leisure", "family trips"],
            "highway": ["highway", "long drive", "road trip", "outstation"],
            "city": ["city driving", "city use", "traffic", "urban"],
        }
        for pattern, keywords in pattern_map.items():
            if any(kw in message_lower for kw in keywords):
                return pattern
        return None

    # ── Buyer & Customer Type Extractors ─────────────────────────────

    def _extract_buyer_type(self, message_lower: str) -> Optional[str]:
        """Extract buyer type: first_time, replacement, or additional."""
        first_time_kw = ["first car", "first time", "first vehicle", "never owned", "new buyer", "pehli gaadi"]
        replacement_kw = ["replace", "replacing", "upgrade", "exchange my", "instead of my old"]
        additional_kw = ["additional", "second car", "another car", "one more", "extra vehicle"]
        if any(kw in message_lower for kw in first_time_kw):
            return "first_time"
        if any(kw in message_lower for kw in replacement_kw):
            return "replacement"
        if any(kw in message_lower for kw in additional_kw):
            return "additional"
        # If trade-in is mentioned, likely replacement
        if self.trade_in_mentioned_check(message_lower):
            return "replacement"
        return None

    def trade_in_mentioned_check(self, message_lower: str) -> bool:
        """Quick check if trade-in keywords are present."""
        return any(kw in message_lower for kw in ['exchange', 'trade', 'old car', 'sell my'])

    def _extract_customer_type(self, message_lower: str) -> Optional[str]:
        """Extract customer type: individual or corporate."""
        corporate_kw = ["corporate", "company", "fleet", "business", "firm", "organization", "bulk"]
        if any(kw in message_lower for kw in corporate_kw):
            return "corporate"
        # Default to individual if any personal indicators
        individual_kw = ["personal", "family", "my wife", "my husband", "myself", "home"]
        if any(kw in message_lower for kw in individual_kw):
            return "individual"
        return None

    # ── Demographics Extractors ──────────────────────────────────────

    def _extract_prefix(self, message_lower: str) -> Optional[str]:
        """Extract name prefix (Dr/Mr/Mrs/Ms)."""
        match = re.search(r'\b(dr|mr|mrs|ms|miss|shri|smt)\b\.?', message_lower)
        if match:
            prefix_map = {"dr": "Dr", "mr": "Mr", "mrs": "Mrs", "ms": "Ms", "miss": "Miss", "shri": "Shri", "smt": "Smt"}
            return prefix_map.get(match.group(1), match.group(1).title())
        return None

    def _extract_gender(self, message_lower: str) -> Optional[str]:
        """Extract gender from context clues."""
        male_kw = [" he ", " his ", "mr ", "shri ", "husband", "father", "son"]
        female_kw = [" she ", " her ", "mrs ", "ms ", "smt ", "wife", "mother", "daughter"]
        if any(kw in message_lower for kw in female_kw):
            return "female"
        if any(kw in message_lower for kw in male_kw):
            return "male"
        return None

    def _extract_marital_status(self, message_lower: str) -> Optional[str]:
        """Extract marital status."""
        married_kw = ["married", "wife", "husband", "spouse", "family of", "mrs"]
        single_kw = ["single", "unmarried", "bachelor"]
        if any(kw in message_lower for kw in married_kw):
            return "married"
        if any(kw in message_lower for kw in single_kw):
            return "single"
        return None

    def _extract_age(self, message: str) -> Optional[int]:
        """Extract age from message."""
        patterns = [
            re.compile(r'(?:age|aged|i am|i\'m)\s*(\d{2})\b', re.IGNORECASE),
            re.compile(r'(\d{2})\s*(?:years?\s*old|yrs?\s*old|yo)\b', re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                age = int(match.group(1))
                if 18 <= age <= 100:
                    return age
        return None

    # ── Feedback & NPS Extractors ────────────────────────────────────

    def _extract_feedback_rating(self, message_lower: str) -> Optional[str]:
        """Extract feedback rating category."""
        if any(kw in message_lower for kw in ["excellent", "amazing", "outstanding", "fantastic"]):
            return "excellent"
        if any(kw in message_lower for kw in ["very good", "great", "really good"]):
            return "very_good"
        if any(kw in message_lower for kw in ["fair", "okay", "ok", "average", "decent"]):
            return "fair"
        if any(kw in message_lower for kw in ["poor", "bad", "terrible", "worst", "horrible"]):
            return "poor"
        return None

    def _extract_nps_score(self, message: str) -> Optional[int]:
        """Extract NPS score (0-10) from message."""
        # Look for patterns like "8 out of 10", "score: 9", "rating 7", or standalone digit in feedback context
        patterns = [
            re.compile(r'(\d{1,2})\s*(?:out of|/)\s*10', re.IGNORECASE),
            re.compile(r'(?:score|rating|rate)\s*(?:is|:)?\s*(\d{1,2})', re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 10:
                    return score
        return None

    def _calculate_confidence(self, entities: ExtractedEntities) -> float:
        """Calculate extraction confidence score."""
        scores = []

        # Phone/email extraction is usually accurate
        if entities.phone_numbers:
            scores.append(0.95)
        if entities.email_addresses:
            scores.append(0.95)

        # Budget extraction
        if entities.budget_mentioned:
            scores.append(0.8)

        # Model/category extraction
        if entities.models_mentioned:
            scores.append(0.7)
        if entities.categories_mentioned:
            scores.append(0.8)

        # Location extraction
        if entities.city:
            scores.append(0.85)
        if entities.pincode:
            scores.append(0.95)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
