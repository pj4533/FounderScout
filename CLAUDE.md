# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FounderScout is a CLI tool that discovers overlooked builders and founders on Hacker News and GitHub. It uses OpenAI's GPT models to analyze founder signals and identify real builders who are actively creating projects.

## Commands

### Run the tool
```bash
python3 founder_scout.py --days 7
```

### Common options
- `--days N` - Number of days to look back (default: 7)
- `--no-llm` - Skip OpenAI analysis for faster results
- `--top N` - Show top N results (default: 20)
- `--verbose` - Show detailed information
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
2. **Analysis**: Uses OpenAI GPT models to analyze founder signals and extract insights
3. **Scoring**: Calculates an "overlooked score" based on visibility, activity, and founder confidence
4. **Display**: Presents results in a rich table format or exports to JSON

Key methods:
- `scan_hackernews_for_founders()` - Searches HN for Show HN posts and founder language
- `scan_github_for_builders()` - Finds new GitHub repos with builder characteristics  
- `analyze_with_llm()` - Uses GPT to determine founder likelihood
- `score_and_filter_founders()` - Ranks candidates by overlooked potential
- `display_founders()` - Shows results in formatted table

## Testing

Currently, there are no formal tests. To verify functionality:
```bash
# Basic run
python3 founder_scout.py --days 1 --no-llm

# With LLM analysis (requires API key)
python3 founder_scout.py --days 3 --top 5

# Verbose output
python3 founder_scout.py --days 7 --verbose
```

## Key Implementation Notes

- Uses concurrent.futures for parallel API fetching
- Implements rate limiting and error handling for API calls
- GPT model selection is configurable via environment variable
- Results are scored based on a combination of low visibility + high founder confidence
- The tool specifically looks for "Show HN" posts and founder-specific language patterns