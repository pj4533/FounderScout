# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProjectScout is a CLI tool that discovers overlooked projects and interesting builders on Hacker News and GitHub. It scouts for undiscovered projects by scoring them based on how overlooked, unique, and passionate they appear.

## Commands

### Run the tool
```bash
python3 project_scout.py --days 7
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

The codebase consists of a single main script `project_scout.py` with the `ProjectScout` class that:

1. **Data Collection**: Fetches data from Hacker News Firebase API and GitHub API in parallel
   - HN: Searches showstories, askstories, newstories, topstories for builder signals
   - GitHub: Queries for new repos focusing on substance over specific languages

2. **Scoring Algorithm**: Ranks projects by balancing passion vs engagement:
   - **Passion**: How much effort/substance the creator put in
     - For HN: Length of post text, detailed explanations
     - For GitHub: Repository size, quality of description
   - **Engagement**: How much attention it has received
     - For HN: Points and comments
     - For GitHub: Stars and forks
   - **Goal**: Surface high-passion, low-engagement projects (overlooked gems)

3. **LLM Post-Processing** (optional): After scoring, enriches top N results with:
   - Keywords extraction (3-5 technical/domain keywords)
   - Summary generation (what makes the project interesting)
   - Vibe detection (passionate/exploratory/pragmatic/playful/serious)
   - All done in a single batch request for efficiency

4. **Display**: Presents results in a rich table format or exports to JSON

Key methods:
- `scan_hackernews_for_projects()` - Casts wide net for any project signals on HN
- `scan_github_for_projects()` - Finds repos with substantial content
- `score_and_rank_projects()` - Balances passion/effort against engagement
- `enrich_with_llm()` - Batch post-processing for keywords/summaries (top N only)
- `display_projects()` - Shows results with score breakdowns and enriched data

## Testing

To verify functionality:
```bash
# Basic run without LLM
python3 project_scout.py --days 1 --no-llm

# With LLM enrichment (requires API key)
python3 project_scout.py --days 3 --top 10

# Verbose output with score breakdowns
python3 project_scout.py --days 7 --verbose

# Export to JSON
python3 project_scout.py --days 7 --output json > results.json
```

## Key Implementation Notes

- Philosophy: Finds overlooked gems by balancing passion with engagement
- Scoring rewards high effort/substance with low visibility
- No language or keyword bias - focuses on effort and substance
- LLM is used only for post-processing enrichment, not filtering
- Batch LLM processing sends all top N results in one request
- More inclusive detection - accepts various builder signals
- Results show passionate builders who haven't gotten attention yet