"""Tests for Lead Scoring components."""

import pytest
from lead_scoring.intent_classifier import IntentClassifier, Intent, Timeline
from lead_scoring.entity_extractor import EntityExtractor
from lead_scoring.scoring_model import LeadScorer, LeadPriority


@pytest.fixture
def classifier():
    return IntentClassifier()


@pytest.fixture
def extractor():
    return EntityExtractor()


@pytest.fixture
def scorer():
    return LeadScorer()


# ── Intent Classifier ─────────────────────────────────

class TestIntentClassifier:
    def test_buy_intent(self, classifier):
        result = classifier.classify("I want to purchase a new car")
        assert result.primary_intent == Intent.BUY

    def test_finance_intent(self, classifier):
        result = classifier.classify("What are your EMI options?")
        assert result.primary_intent == Intent.FINANCE

    def test_test_drive_intent(self, classifier):
        result = classifier.classify("Can I book a test drive?")
        assert result.primary_intent == Intent.TEST_DRIVE

    def test_service_intent(self, classifier):
        result = classifier.classify("I need to get my car serviced")
        assert result.primary_intent == Intent.SERVICE

    def test_info_intent(self, classifier):
        result = classifier.classify("Hello, what do you sell?")
        assert result.primary_intent in (Intent.INFO, Intent.UNKNOWN)

    def test_exchange_intent(self, classifier):
        result = classifier.classify("I want to exchange my old car for a new one")
        assert result.primary_intent == Intent.EXCHANGE

    def test_timeline_immediate(self, classifier):
        result = classifier.classify("I want to buy a car today urgently")
        assert result.timeline in (Timeline.IMMEDIATE, Timeline.THIS_WEEK)

    def test_timeline_exploring(self, classifier):
        result = classifier.classify("Just browsing your vehicles")
        assert result.timeline in (Timeline.EXPLORING, Timeline.UNKNOWN)


# ── Entity Extractor ──────────────────────────────────

class TestEntityExtractor:
    def test_extract_budget(self, extractor):
        entities = extractor.extract("My budget is around 15 lakhs")
        assert entities.budget_range is not None

    def test_extract_phone(self, extractor):
        entities = extractor.extract("Call me at 9876543210")
        assert entities.phone is not None

    def test_extract_email(self, extractor):
        entities = extractor.extract("Email me at test@example.com")
        assert entities.email == "test@example.com"

    def test_extract_fuel_type(self, extractor):
        entities = extractor.extract("I prefer a diesel SUV")
        assert "diesel" in [f.lower() for f in entities.fuel_types]

    def test_extract_city(self, extractor):
        entities = extractor.extract("I am based in Mumbai")
        assert any("mumbai" in c.lower() for c in entities.cities)

    def test_no_entities(self, extractor):
        entities = extractor.extract("Hello")
        assert entities.phone is None
        assert entities.email is None


# ── Lead Scorer ───────────────────────────────────────

class TestLeadScorer:
    def test_hot_lead(self, classifier, extractor, scorer):
        intent = classifier.classify("I want to buy a Nexon today, my budget is 12 lakhs. Call me at 9876543210")
        entities = extractor.extract("I want to buy a Nexon today, my budget is 12 lakhs. Call me at 9876543210")
        result = scorer.score(intent_result=intent, entities=entities, conversation_turns=3)
        assert result.priority == LeadPriority.HOT
        assert result.score >= 70

    def test_warm_lead(self, classifier, extractor, scorer):
        intent = classifier.classify("What financing options do you have for Creta?")
        entities = extractor.extract("What financing options do you have for Creta?")
        result = scorer.score(intent_result=intent, entities=entities, conversation_turns=2)
        assert result.priority in (LeadPriority.WARM, LeadPriority.HOT)
        assert result.score >= 40

    def test_cold_lead(self, classifier, extractor, scorer):
        intent = classifier.classify("What colors are available?")
        entities = extractor.extract("What colors are available?")
        result = scorer.score(intent_result=intent, entities=entities, conversation_turns=1)
        assert result.priority in (LeadPriority.COLD, LeadPriority.WARM)
        assert result.score < 70

    def test_score_range(self, classifier, extractor, scorer):
        intent = classifier.classify("Hi")
        entities = extractor.extract("Hi")
        result = scorer.score(intent_result=intent, entities=entities, conversation_turns=1)
        assert 0 <= result.score <= 100

    def test_custom_thresholds(self, scorer):
        scorer.adjust_thresholds(hot=80, warm=60)
        # Thresholds adjusted — verify they're stored
        assert scorer.hot_threshold == 80
        assert scorer.warm_threshold == 60
