"""
Vehicle Inventory Loader for AMPL Chatbot.

Handles loading and processing vehicle inventory data from CSV, JSON, or API sources.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class VehicleDocument:
    """Represents a processed vehicle document ready for embedding."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VehicleSpec(BaseModel):
    """Vehicle specification model."""
    model_name: str
    variant: Optional[str] = None
    price: float
    ex_showroom_price: Optional[float] = None
    on_road_price: Optional[float] = None
    category: str  # SUV, Sedan, Hatchback, etc.
    fuel_type: str  # Petrol, Diesel, Electric, Hybrid
    transmission: str  # Manual, Automatic, CVT
    engine_cc: Optional[int] = None
    power_hp: Optional[int] = None
    torque_nm: Optional[int] = None
    mileage_kmpl: Optional[float] = None
    seating_capacity: int = 5
    launch_year: int
    colors: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    safety_features: List[str] = Field(default_factory=list)
    availability: str = "available"  # available, limited, sold_out, coming_soon
    image_urls: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    class Config:
        extra = "allow"


class InventoryLoader:
    """
    Loads and processes vehicle inventory for RAG ingestion.

    Supports:
    - CSV files with vehicle data
    - JSON files with vehicle data
    - REST API endpoints
    """

    def __init__(self, namespace: str = "inventory"):
        """
        Initialize the inventory loader.

        Args:
            namespace: Pinecone namespace for inventory documents
        """
        self.namespace = namespace
        self._loaded_vehicles: List[VehicleSpec] = []

    def load_from_csv(self, file_path: Union[str, Path]) -> List[VehicleDocument]:
        """
        Load vehicle inventory from a CSV file.

        Expected columns:
        - model_name, variant, price, category, fuel_type, transmission, etc.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of VehicleDocument objects ready for embedding
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        vehicles = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse and validate vehicle data
                    vehicle = self._parse_csv_row(row)
                    if vehicle:
                        vehicles.append(vehicle)
                except Exception as e:
                    logger.warning(f"Failed to parse row: {row}. Error: {e}")
                    continue

        self._loaded_vehicles.extend(vehicles)
        logger.info(f"Loaded {len(vehicles)} vehicles from {file_path}")

        return self._convert_to_documents(vehicles)

    def load_from_json(self, file_path: Union[str, Path]) -> List[VehicleDocument]:
        """
        Load vehicle inventory from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of VehicleDocument objects ready for embedding
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array and object with "vehicles" key
        if isinstance(data, list):
            vehicle_data = data
        elif isinstance(data, dict) and "vehicles" in data:
            vehicle_data = data["vehicles"]
        else:
            vehicle_data = [data]

        vehicles = []
        for item in vehicle_data:
            try:
                vehicle = VehicleSpec(**item)
                vehicles.append(vehicle)
            except Exception as e:
                logger.warning(f"Failed to parse vehicle: {item}. Error: {e}")
                continue

        self._loaded_vehicles.extend(vehicles)
        logger.info(f"Loaded {len(vehicles)} vehicles from {file_path}")

        return self._convert_to_documents(vehicles)

    async def load_from_api(
        self,
        api_url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[VehicleDocument]:
        """
        Load vehicle inventory from a REST API.

        Args:
            api_url: URL of the inventory API
            headers: Optional HTTP headers
            params: Optional query parameters

        Returns:
            List of VehicleDocument objects ready for embedding
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        # Handle different response formats
        if isinstance(data, list):
            vehicle_data = data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["vehicles", "data", "items", "results"]:
                if key in data:
                    vehicle_data = data[key]
                    break
            else:
                vehicle_data = [data]
        else:
            vehicle_data = []

        vehicles = []
        for item in vehicle_data:
            try:
                vehicle = VehicleSpec(**item)
                vehicles.append(vehicle)
            except Exception as e:
                logger.warning(f"Failed to parse vehicle from API: {item}. Error: {e}")
                continue

        self._loaded_vehicles.extend(vehicles)
        logger.info(f"Loaded {len(vehicles)} vehicles from API: {api_url}")

        return self._convert_to_documents(vehicles)

    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[VehicleSpec]:
        """Parse a CSV row into a VehicleSpec object."""
        # Map common CSV column names to VehicleSpec fields
        mapping = {
            "model": "model_name",
            "name": "model_name",
            "vehicle_name": "model_name",
            "price_inr": "price",
            "cost": "price",
            "type": "category",
            "body_type": "category",
            "fuel": "fuel_type",
            "trans": "transmission",
            "gearbox": "transmission",
            "year": "launch_year",
            "model_year": "launch_year",
            "seats": "seating_capacity",
            "status": "availability",
        }

        # Normalize row keys
        normalized = {}
        for key, value in row.items():
            key_lower = key.lower().strip().replace(" ", "_")
            mapped_key = mapping.get(key_lower, key_lower)
            normalized[mapped_key] = value.strip() if isinstance(value, str) else value

        # Parse required fields
        try:
            # Handle price parsing (remove currency symbols, commas)
            price_str = normalized.get("price", "0")
            if isinstance(price_str, str):
                price_str = price_str.replace(",", "").replace("₹", "").replace("Rs", "").strip()
            price = float(price_str) if price_str else 0.0

            # Handle year parsing
            year_str = normalized.get("launch_year", str(datetime.now().year))
            launch_year = int(year_str) if year_str else datetime.now().year

            # Parse list fields
            colors = self._parse_list_field(normalized.get("colors", ""))
            features = self._parse_list_field(normalized.get("features", ""))
            safety_features = self._parse_list_field(normalized.get("safety_features", ""))

            return VehicleSpec(
                model_name=normalized.get("model_name", "Unknown"),
                variant=normalized.get("variant"),
                price=price,
                ex_showroom_price=self._parse_float(normalized.get("ex_showroom_price")),
                on_road_price=self._parse_float(normalized.get("on_road_price")),
                category=normalized.get("category", "Car"),
                fuel_type=normalized.get("fuel_type", "Petrol"),
                transmission=normalized.get("transmission", "Manual"),
                engine_cc=self._parse_int(normalized.get("engine_cc")),
                power_hp=self._parse_int(normalized.get("power_hp")),
                torque_nm=self._parse_int(normalized.get("torque_nm")),
                mileage_kmpl=self._parse_float(normalized.get("mileage_kmpl")),
                seating_capacity=self._parse_int(normalized.get("seating_capacity")) or 5,
                launch_year=launch_year,
                colors=colors,
                features=features,
                safety_features=safety_features,
                availability=normalized.get("availability", "available"),
                description=normalized.get("description"),
            )
        except Exception as e:
            logger.error(f"Error parsing CSV row: {e}")
            return None

    def _parse_list_field(self, value: str) -> List[str]:
        """Parse a comma or semicolon separated string into a list."""
        if not value:
            return []

        # Try different separators
        for sep in [";", "|", ","]:
            if sep in value:
                return [item.strip() for item in value.split(sep) if item.strip()]

        return [value.strip()] if value.strip() else []

    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse a float value."""
        if value is None or value == "":
            return None
        try:
            if isinstance(value, str):
                value = value.replace(",", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """Safely parse an integer value."""
        if value is None or value == "":
            return None
        try:
            if isinstance(value, str):
                value = value.replace(",", "").strip()
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def _convert_to_documents(self, vehicles: List[VehicleSpec]) -> List[VehicleDocument]:
        """Convert VehicleSpec objects to VehicleDocument objects for embedding."""
        documents = []

        for vehicle in vehicles:
            # Generate unique ID
            doc_id = self._generate_document_id(vehicle)

            # Create rich text content for embedding
            content = self._create_vehicle_content(vehicle)

            # Create metadata for filtering
            metadata = self._create_vehicle_metadata(vehicle)

            documents.append(VehicleDocument(
                id=doc_id,
                content=content,
                metadata=metadata
            ))

        return documents

    def _generate_document_id(self, vehicle: VehicleSpec) -> str:
        """Generate a unique document ID for a vehicle."""
        unique_string = f"{vehicle.model_name}:{vehicle.variant or 'base'}:{vehicle.launch_year}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _create_vehicle_content(self, vehicle: VehicleSpec) -> str:
        """Create rich text content for a vehicle, optimized for embedding."""
        parts = [
            f"Vehicle: {vehicle.model_name}",
        ]

        if vehicle.variant:
            parts.append(f"Variant: {vehicle.variant}")

        parts.extend([
            f"Category: {vehicle.category}",
            f"Price: ₹{vehicle.price:,.0f}",
        ])

        if vehicle.ex_showroom_price:
            parts.append(f"Ex-showroom Price: ₹{vehicle.ex_showroom_price:,.0f}")

        if vehicle.on_road_price:
            parts.append(f"On-road Price: ₹{vehicle.on_road_price:,.0f}")

        parts.extend([
            f"Fuel Type: {vehicle.fuel_type}",
            f"Transmission: {vehicle.transmission}",
        ])

        if vehicle.engine_cc:
            parts.append(f"Engine: {vehicle.engine_cc}cc")

        if vehicle.power_hp:
            parts.append(f"Power: {vehicle.power_hp} HP")

        if vehicle.torque_nm:
            parts.append(f"Torque: {vehicle.torque_nm} Nm")

        if vehicle.mileage_kmpl:
            parts.append(f"Mileage: {vehicle.mileage_kmpl} km/l")

        parts.append(f"Seating Capacity: {vehicle.seating_capacity} passengers")
        parts.append(f"Launch Year: {vehicle.launch_year}")

        if vehicle.colors:
            parts.append(f"Available Colors: {', '.join(vehicle.colors)}")

        if vehicle.features:
            parts.append(f"Key Features: {', '.join(vehicle.features[:10])}")

        if vehicle.safety_features:
            parts.append(f"Safety Features: {', '.join(vehicle.safety_features[:5])}")

        parts.append(f"Availability: {vehicle.availability}")

        if vehicle.description:
            parts.append(f"\n{vehicle.description}")

        return "\n".join(parts)

    def _create_vehicle_metadata(self, vehicle: VehicleSpec) -> Dict[str, Any]:
        """Create metadata dictionary for a vehicle."""
        return {
            "source": "inventory",
            "document_type": "vehicle",
            "model_name": vehicle.model_name,
            "variant": vehicle.variant or "base",
            "price": vehicle.price,
            "category": vehicle.category,
            "fuel_type": vehicle.fuel_type,
            "transmission": vehicle.transmission,
            "launch_year": vehicle.launch_year,
            "availability": vehicle.availability,
            "seating_capacity": vehicle.seating_capacity,
            "ingested_at": datetime.utcnow().isoformat(),
        }

    def get_loaded_vehicles(self) -> List[VehicleSpec]:
        """Get all loaded vehicles."""
        return self._loaded_vehicles.copy()

    def clear(self):
        """Clear loaded vehicles."""
        self._loaded_vehicles.clear()
