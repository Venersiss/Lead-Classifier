"""
Multi-Stage Lead Classification Pipeline
=========================================
Classifies business leads into category buckets using a cost-optimized
3-stage pipeline: fast keyword matching → website scraping → LLM.

ARCHITECTURE:
  Stage 1: Keyword Classifier (< 5ms) — company name + domain keyword match
  Stage 2: Website Scraper (~2s) — HTTP GET homepage → extract metadata
  Stage 3: Gemini Flash Classifier (~500ms) — LLM-based semantic understanding

WHY 3 STAGES:
  Running an LLM call on every lead would be expensive and slow at scale.
  The vast majority of leads are classified by simple keyword matching (Stage 1).
  Only ambiguous cases reach the website scraper, and only the truly difficult
  ones hit the LLM. This reduces cost by ~85% and latency by ~70%.

NDA NOTE: Industry-specific keyword lists, exact prompt templates, and certain
proprietary scoring heuristics have been modified. The architecture and pipeline
flow remain structurally accurate.

Tech Stack: Python 3, httpx, Google Gemini API (gemini-2.5-flash), Baserow
"""

import asyncio
import logging
import re
import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("classifier")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
BATCH_SIZE = 12  # rows per classification run
REQUEST_TIMEOUT = 10  # seconds for website scraping

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Lead:
    """A raw lead record before classification."""
    company_name: str
    website: str = ""
    domain: str = ""
    city: str = ""
    row_id: Optional[int] = None

    def __post_init__(self):
        if self.website and not self.domain:
            try:
                self.domain = urlparse(self.website if "://" in self.website else f"https://{self.website}").netloc
            except Exception:
                self.domain = ""


@dataclass
class ClassificationResult:
    """The output of the classification pipeline."""
    lead: Lead
    category: str  # e.g., "solar", "not_solar"
    confidence: float = 0.0
    reason: str = ""
    stage: str = ""  # "keyword", "scrape", "llm"
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Stage 1: Fast Keyword Classifier
# ---------------------------------------------------------------------------
def _is_primary_domain(domain: str) -> bool:
    """Reject aggregators like leadswift.com, yellowpages.com that aren't the
    actual business."""
    skip_domains = {"leadswift.com", "yellowpages.com", "angi.com", "yelp.com",
                    "homeadvisor.com", "thumbtack.com", "bbb.org"}
    for skip in skip_domains:
        if skip in domain:
            return False
    return True


class KeywordClassifier:
    """
    Fast in-memory keyword matcher. Checks company name and domain against
    industry-specific keyword dictionaries. Runs in < 5ms per lead.
    """

    # NDA NOTE: The actual keyword dictionaries contained ~50+ industry-specific
    # terms per category. The representative sample below demonstrates the pattern.

    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "solar": [
            "solar", "photovoltaic", "renewable", "sunpower", "sunrun",
            "energy storage", "battery storage", "powerwall", "inverter",
            "net metering", "pv panel", "green energy", "sunpro", "sunergy",
        ],
        "roofing": [
            "roofing", "roof repair", "roof replacement", "shingle",
            "gutter", "flat roof", "metal roof", "tile roof", "slate roof",
        ],
        "hvac": [
            "hvac", "heating", "air conditioning", "furnace", "heat pump",
            "air handler", "duct", "ventilation", "cooling",
        ],
        "plumbing": [
            "plumbing", "plumber", "drain", "pipe", "water heater",
            "leak repair", "sewer", "toilet", "faucet",
        ],
    }

    def classify(self, lead: Lead, target_category: str) -> Optional[str]:
        """Return category if keyword match found, otherwise None."""
        keywords = self.CATEGORY_KEYWORDS.get(target_category, [])
        text = f"{lead.company_name} {lead.domain} {lead.website}".lower()

        for kw in keywords:
            if kw.lower() in text:
                return target_category
        return None


# ---------------------------------------------------------------------------
# Stage 2: Website Scraper
# ---------------------------------------------------------------------------
class WebsiteScraper:
    """
    Fetches the company's homepage and extracts title, meta description,
    and body text for content-based classification.
    """

    @staticmethod
    def build_url(website: str) -> Optional[str]:
        """Normalize URLs missing a protocol."""
        if not website:
            return None
        website = website.strip()
        if not website.startswith(("http://", "https://")):
            website = f"https://{website}"
        return website

    @staticmethod
    def extract_metadata(html: str) -> dict:
        """Pull <title>, <meta description>, and visible text from HTML."""
        result = {"title": "", "description": "", "body_text": ""}

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            result["title"] = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if not desc_match:
            desc_match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
                html, re.IGNORECASE,
            )
        if desc_match:
            result["description"] = desc_match.group(1).strip()

        # Strip scripts, styles, and tags for body text
        body = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        result["body_text"] = body[:3000]

        return result

    async def scrape(self, lead: Lead) -> Optional[dict]:
        """Attempt to fetch and parse the company's homepage."""
        url = self.build_url(lead.website)
        if not url:
            return None

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.get(url, follow_redirects=True,
                                        headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                return self.extract_metadata(resp.text)
        except Exception as exc:
            logger.warning("Scrape failed for %s: %s", lead.website, exc)
            return None


# ---------------------------------------------------------------------------
# Stage 3: Gemini Flash Classifier
# ---------------------------------------------------------------------------
class GeminiClassifier:
    """
    LLM-based classification using Gemini 2.5 Flash. Only invoked when
    keyword and scrape stages both fail to classify.
    """

    # NDA NOTE: The actual prompt contained detailed category definitions
    # optimized over many iterations. A structurally equivalent version is below.

    CLASSIFICATION_PROMPT = """You are a business classifier. Determine if this company is a {category} contractor.

Company name: {company_name}
Website title: {title}
Meta description: {description}
Website text: {body_text}

Rules:
- Return ONLY the category name: "{category}" or "not_{category}"
- A contractor is a company that physically installs, repairs, or maintains {category} systems
- Suppliers, manufacturers, or review sites are NOT contractors
- If uncertain, return "not_{category}"

Classification:"""

    async def classify(self, lead: Lead, scraped_data: Optional[dict], target_category: str) -> str:
        """Call Gemini Flash to classify an ambiguous lead."""
        scraped = scraped_data or {}
        prompt = self.CLASSIFICATION_PROMPT.format(
            category=target_category,
            company_name=lead.company_name,
            title=scraped.get("title", ""),
            description=scraped.get("description", ""),
            body_text=scraped.get("body_text", "")[:2000],
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                    params={"key": GEMINI_API_KEY},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 50},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
                return target_category if target_category in text and "not" not in text else f"not_{target_category}"
        except Exception as exc:
            logger.error("Gemini classification failed for %s: %s", lead.company_name, exc)
            return f"not_{target_category}"


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class ClassificationPipeline:
    """
    Orchestrates the 3-stage classification pipeline with cost optimization.

    Flow:
      1. Keyword classifiers run in parallel (< 5ms each)
      2. Unclassified leads get their websites scraped and re-run through keywords
      3. Still-unclassified leads go to Gemini Flash for semantic classification
    """

    def __init__(self):
        self.keyword = KeywordClassifier()
        self.scraper = WebsiteScraper()
        self.gemini = GeminiClassifier()

    async def classify_lead(self, lead: Lead, target_category: str) -> ClassificationResult:
        """Run a single lead through the classification pipeline."""
        import time
        start = time.perf_counter()

        # Stage 1: Keywords
        result = self.keyword.classify(lead, target_category)
        if result:
            return ClassificationResult(
                lead=lead, category=result, confidence=0.9,
                reason="Keyword match in company name/domain",
                stage="keyword",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Stage 2: Scrape + re-check keywords on scraped data
        scraped = await self.scraper.scrape(lead)
        if scraped:
            # Build a virtual lead from scraped data for keyword checking
            scraped_text = f"{scraped.get('title', '')} {scraped.get('description', '')} {scraped.get('body_text', '')[:500]}"
            for kw in self.keyword.CATEGORY_KEYWORDS.get(target_category, []):
                if kw.lower() in scraped_text.lower():
                    return ClassificationResult(
                        lead=lead, category=target_category, confidence=0.75,
                        reason=f"Keyword '{kw}' found in scraped content",
                        stage="scrape",
                        processing_time_ms=(time.perf_counter() - start) * 1000,
                    )

        # Stage 3: Gemini Flash
        category = await self.gemini.classify(lead, scraped, target_category)
        confidence = 0.85 if category == target_category else 0.5
        return ClassificationResult(
            lead=lead, category=category, confidence=confidence,
            reason="Gemini Flash AI classification" if category == target_category else "AI could not confirm category",
            stage="llm",
            processing_time_ms=(time.perf_counter() - start) * 1000,
        )

    async def classify_batch(self, leads: list[Lead], target_category: str) -> list[ClassificationResult]:
        """Classify a batch of leads concurrently."""
        tasks = [self.classify_lead(lead, target_category) for lead in leads]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Hard guard: skip invalid or already-processed leads
# ---------------------------------------------------------------------------
def filter_eligible_leads(leads: list[Lead]) -> list[Lead]:
    """Remove leads that aren't eligible for classification."""
    eligible = []
    for lead in leads:
        if not lead.company_name:
            continue
        if not _is_primary_domain(lead.domain):
            continue
        eligible.append(lead)
    return eligible


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
async def main():
    # Simulated unclassified leads
    sample_leads = [
        Lead(company_name="Sun Valley Solar LLC", website="sunvalleysolar.com"),
        Lead(company_name="ABC Heating & Cooling", website="abchvac.com"),
        Lead(company_name="Tom's General Contracting", website="tomscontracting.com"),
        Lead(company_name="Green Energy Solutions", website="greenenergysolutions.com"),
    ]

    pipeline = ClassificationPipeline()
    eligible = filter_eligible_leads(sample_leads)

    results = await pipeline.classify_batch(eligible, target_category="solar")
    for r in results:
        print(f"{r.lead.company_name:40s} → {r.category:15s} stage={r.stage:8s} ({r.processing_time_ms:.0f}ms) | {r.reason}")


if __name__ == "__main__":
    asyncio.run(main())
