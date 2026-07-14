# Lead Classification Pipeline

**Role:** Lead Developer & Automation Architect  
**Tech Stack:** Python 3, Google Gemini 2.5 Flash, httpx, Baserow, n8n

## What It Does

Classifies business leads into industry categories using a cost-optimized multi-stage pipeline. When thousands of leads come in from data providers, most are unlabeled — this system determines "Is this company actually a solar contractor? A roofer? An HVAC company?" before they enter the outreach workflow.

### 3-Stage Classification Pipeline

```
┌─────────────────────────────────────────────────────────┐
│           STAGE 1: Keyword Classifier (< 5ms)           │
│  Checks company name + domain against 50+ industry      │
│  keywords. ~70% of leads classified here.               │
├─────────────────────────────────────────────────────────┤
│           STAGE 2: Website Scraper (~2s)                │
│  HTTP GET homepage → extract title, meta description,   │
│  body text. Re-checks keywords on scraped content.      │
│  ~20% of remaining leads classified here.               │
├─────────────────────────────────────────────────────────┤
│           STAGE 3: Gemini Flash Classifier (~500ms)    │
│  Sends company name + scraped metadata to Gemini Flash. │
│  Only ~10% of all leads ever reach this stage.          │
│  Cost: ~$0.0001 per classification.                     │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Why not just use LLM for everything?**
At scale (thousands of leads/month), running LLM calls on every lead would cost $50-100/month and add 30+ minutes of processing time. The keyword classifier handles 70% of cases in < 5ms with virtually zero cost. Only the truly ambiguous cases hit the LLM.

**Why scrape websites instead of using LLM first?**
Website scraping is free and often more reliable than LLM. A company with "Solar" in their meta description IS a solar company — no AI needed. The scraper also collects data that feeds into later stages (video script generation, personalization).

**Batch processing:**
The classifier runs every 10 minutes in n8n, processing 12 leads per batch. Each batch takes ~30 seconds max. Rows are locked with "processing" status to prevent duplicate work if the schedule overlaps.

**Auditability:**
Every classification stores the `reason` (which keyword matched, which stage classified it) and `confidence` score. This enables prompt refinement and false-positive detection over time.

## Reliability & Error Handling

1. **Website scrape timeout:** 10-second timeout per site, graceful failure (falls through to LLM)
2. **Primary domain guard:** Skips aggregator domains (Yelp, Angi, HomeAdvisor) that aren't the actual business
3. **LLM constraint:** Temperature 0.1, max 50 output tokens, return ONLY the category name
4. **Processing lock:** Row status set to "processing" before pipeline runs, preventing concurrent workers from double-processing
5. **Gemini fallback:** If Gemini API fails, lead is marked as unclassified (not incorrectly classified) for manual review

## NDA Disclaimer

**Some implementation details have been modified** to comply with confidentiality obligations:
- Industry-specific keyword lists are representative samples, not the full dictionaries
- Exact Gemini prompt templates have been generalized
- Certain proprietary scoring heuristics and confidence thresholds are omitted
- Database schema details (Baserow table IDs, field names) have been removed

The 3-stage architecture, cost optimization strategy, and error handling patterns are accurate representations of the production system.

## Relevance to Application Questions

- **Q12 (AI/automation systems):** Production classification system processing real business leads
- **Q15 (AI producing wrong answers):** The keyword classifier sometimes misclassified "Green Energy Solutions" as solar when it was actually a general consulting firm. We detected this through the `classifier_reason` audit trail and refined by adding the website scrape + LLM fallback stages, and by adding negative keywords (e.g., "consulting", "broker") that force the LLM review
