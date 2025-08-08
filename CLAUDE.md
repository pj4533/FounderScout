# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FounderScout is a CLI tool that discovers overlooked builders and interesting projects on Hacker News and GitHub. It scouts for undiscovered founder types by scoring projects based on how overlooked, unique, and passionate they appear, rather than validating "real" founders.

## Commands

### Run the tool
```bash
python3 founder_scout.py --days 7
```

### Common options
- `--days N` - Number of days to look back (default: 7)
- `--no-llm` - Skip LLM keyword/summary generation for faster results
- `--top N` - Show top N results (default: 20)
- `--verbose` - Show detailed information including score breakdowns
- `--output json` - Export results to JSON format

### Install dependencies
```bash
pip3 install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5  # Can be gpt-5, gpt-5-mini, or gpt-5-nano
GITHUB_TOKEN=your_github_token_here  # Optional for higher rate limits
```

## Architecture

The codebase consists of a single main script `founder_scout.py` with the `FounderScout` class that:

1. **Data Collection**: Fetches data from Hacker News Firebase API and GitHub API in parallel
   - HN: Searches showstories, askstories, newstories, topstories for builder signals
   - GitHub: Queries for new repos with various filters including interesting languages

2. **Scoring Algorithm**: Ranks projects by multiple factors (not "founder realness"):
   - **Overlooked** (35%): Low visibility/stars but has substance
   - **Weird/Unique** (20%): Unusual tech choices, niche domains, experimental
   - **Passion** (15%): Creator enthusiasm and excitement detected in text
   - **Builder** (15%): Evidence of actually building something
   - **Engagement** (10%): Some activity but not too popular
   - **Completeness** (5%): For GitHub repos, how substantial the project is

3. **LLM Post-Processing** (optional): After scoring, enriches top N results with:
   - Keywords extraction (3-5 technical/domain keywords)
   - Summary generation (what makes the project interesting)
   - Vibe detection (passionate/exploratory/pragmatic/playful/serious)
   - All done in a single batch request for efficiency

4. **Display**: Presents results in a rich table format or exports to JSON

Key methods:
- `scan_hackernews_for_founders()` - Casts wide net for any builder signals on HN
- `scan_github_for_builders()` - Finds repos including those in interesting languages
- `score_and_rank_projects()` - Multi-factor scoring focused on finding hidden gems
- `enrich_with_llm()` - Batch post-processing for keywords/summaries (top N only)
- `display_founders()` - Shows results with score breakdowns and enriched data

## Testing

To verify functionality:
```bash
# Basic run without LLM
python3 founder_scout.py --days 1 --no-llm

# With LLM enrichment (requires API key)
python3 founder_scout.py --days 3 --top 10

# Verbose output with score breakdowns
python3 founder_scout.py --days 7 --verbose

# Export to JSON
python3 founder_scout.py --days 7 --output json > results.json
```

## Key Implementation Notes

- Philosophy: Finds overlooked gems rather than validating "real" founders
- Scoring prioritizes low-visibility projects with interesting characteristics
- LLM is used only for post-processing enrichment, not filtering
- Batch LLM processing sends all top N results in one request
- More inclusive detection - accepts various builder signals
- GitHub searches include cutting-edge languages (Rust, Zig, etc.)
- Results show what makes each project interesting/unique