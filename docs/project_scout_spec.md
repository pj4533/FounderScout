# ProjectScout – Automated Project Discovery

## Goal
A zero-configuration CLI tool that automatically surfaces **overlooked projects** from Hacker News and GitHub.

Just run `projectscout --days 7` and get a curated list of overlooked projects and the builders behind them.

**Core principle:** The tool does all the thinking for you. No search terms, no configuration - just smart detection of founder signals and overlooked gems.

---

## How It Works

### 1. Simple Input
```bash
projectscout --days 7  # Only parameter needed
```

### 2. Automated Discovery Process

The script automatically:
1. **Fetches all recent activity** from HN and GitHub within the time window
2. **Analyzes each item** for founder signals and project characteristics
3. **Extracts keywords** using LLM analysis (OpenAI API)
4. **Scores everything** using the overlooked algorithm
5. **Outputs a unified table** with all discovered projects

---

## Data Sources

### 1. Hacker News (Firebase API)

**Base URL:** `https://hacker-news.firebaseio.com/v0`

**Scanning Strategy:**
```python
def scan_hn(days):
    # Fetch from ALL story sources for maximum coverage
    sources = ["topstories", "newstories", "beststories", "showstories", "askstories"]
    all_stories = []
    
    for source in sources:
        story_ids = fetch(f"{BASE_URL}/{source}.json")[:500]
        for story_id in story_ids:
            story = fetch(f"{BASE_URL}/item/{story_id}.json")
            if is_within_days(story['time'], days):
                all_stories.append(story)
    
    return all_stories
```

**What we're looking for:**
- Show HN posts (strong founder signal)
- Ask HN posts about launching/building
- Stories with founder-related keywords
- Low points but decent engagement (overlooked gems)

### 2. GitHub (REST API)

**Base URL:** `https://api.github.com`

**Scanning Strategy:**
```python
def scan_github(days):
    # Search for repos created in the time window
    date_filter = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Multiple searches to catch different types of projects
    queries = [
        f"created:>{date_filter}",  # All new repos
        f"pushed:>{date_filter} stars:<50",  # Recently active, low visibility
    ]
    
    repos = []
    for query in queries:
        results = fetch(f"/search/repositories?q={query}&sort=updated&per_page=100")
        repos.extend(results['items'])
    
    return repos
```

**What we're looking for:**
- Newly created repositories
- Recent commit activity
- Meaningful descriptions/READMEs
- Low star count (under-the-radar)

---

## Intelligent Analysis

### Keyword Extraction (GPT-5 Powered)

Using the latest GPT-5 model (released August 2025) to intelligently extract:
- **Technology stack** used
- **Problem domain** being addressed
- **Stage indicators** (MVP, beta, launching, etc.)
- **Unique/weird aspects** that make it interesting
- **Founder signals** with enhanced reasoning

**GPT-5 Advantages:**
- 256,000 token context window for analyzing longer descriptions
- Improved reasoning capabilities (74.9% on SWE-bench vs 69.1% for previous models)
- 22% fewer output tokens and 45% fewer tool calls for efficiency
- Safe-completion approach for better helpfulness within safety constraints

```python
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def extract_insights(item, use_mini=False):
    """Use GPT-5 to extract keywords and classify the project
    
    Args:
        item: The HN story or GitHub repo to analyze
        use_mini: Use gpt-5-mini for faster/cheaper analysis
    """
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Choose model based on speed/cost requirements
    model = "gpt-5-mini" if use_mini else "gpt-5"
    
    prompt = f"""
    Analyze this project and extract:
    1. Key technologies used
    2. Problem domain/industry
    3. Stage (idea/MVP/beta/launched)
    4. What makes it unique or interesting
    5. Founder signals (is this someone building something?)
    
    Title: {item.get('title', '')}
    Description: {item.get('text', item.get('description', ''))}
    URL: {item.get('url', '')}
    
    Return as JSON with keys: technologies, domain, stage, unique_aspects, founder_score (0-1)
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},  # Structured output
        reasoning_effort="medium",  # GPT-5 specific parameter
        verbosity="concise"  # GPT-5 specific parameter
    )
    
    return json.loads(response.choices[0].message.content)
```

**GPT-5 Model Options:**
- `gpt-5` - Main model ($1.25/1M input, $10/1M output tokens)
- `gpt-5-mini` - Faster, cheaper variant ($0.25/1M input, $2/1M output tokens)
- `gpt-5-nano` - Ultra-fast for simple tasks ($0.05/1M input, $0.40/1M output tokens)

### Founder Signal Detection

Automatically detect founder indicators:
```python
FOUNDER_SIGNALS = {
    'strong': [
        'Show HN:', 'just launched', 'my startup', 'we built', 'I built',
        'seeking feedback', 'early access', 'beta users wanted'
    ],
    'medium': [
        'MVP', 'prototype', 'side project', 'weekend project',
        'first version', 'v0.1', 'alpha', 'proof of concept'
    ],
    'weak': [
        'working on', 'building', 'creating', 'developing',
        'feedback welcome', 'open source', 'contributors wanted'
    ]
}
```

---

## Scoring Algorithm

### Overlooked Score Calculation

```python
def calculate_overlooked_score(item, insights):
    """
    Score 0-100 where higher = more overlooked but promising
    """
    
    # 1. Low Visibility Score (40% weight)
    # Lower points/stars = higher score
    if item['source'] == 'hn':
        visibility = item.get('score', 0)
        max_visibility = 100
    else:  # github
        visibility = item.get('stargazers_count', 0)
        max_visibility = 50
    
    low_visibility_score = max(0, 1 - (visibility / max_visibility)) * 40
    
    # 2. Momentum Score (30% weight)
    # Recent activity relative to age
    age_hours = (time.time() - item['created_at']) / 3600
    if item['source'] == 'hn':
        activity = item.get('descendants', 0)  # comments
    else:
        activity = item.get('commits_last_week', 0)
    
    momentum = min(activity / max(age_hours, 1), 1.0)
    momentum_score = momentum * 30
    
    # 3. Founder Signal Score (20% weight)
    # Based on detected founder signals
    founder_score = insights.get('founder_score', 0) * 20
    
    # 4. Uniqueness Score (10% weight)
    # Weird tech, niche domain, unusual approach
    unique_keywords = ['quantum', 'biotech', 'spatial', 'local-first', 
                       'decentralized', 'embedded', 'real-time']
    unique_languages = ['Rust', 'Zig', 'Erlang', 'Haskell', 'Nim']
    
    uniqueness = 0
    if any(kw in str(insights.get('technologies', [])).lower() for kw in unique_keywords):
        uniqueness += 0.5
    if any(lang in str(insights.get('technologies', [])) for lang in unique_languages):
        uniqueness += 0.5
    uniqueness_score = uniqueness * 10
    
    total_score = low_visibility_score + momentum_score + founder_score + uniqueness_score
    
    return {
        'total': round(total_score, 1),
        'visibility': round(low_visibility_score, 1),
        'momentum': round(momentum_score, 1),
        'founder': round(founder_score, 1),
        'uniqueness': round(uniqueness_score, 1)
    }
```

---

## Output Format

### Unified Table Output

All results in a single, readable table sorted by overlooked score:

```
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                          FOUNDERSCOUT - OVERLOOKED BUILDERS                               ║
║                          Scanning last 7 days | Found 23 candidates                       ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝

Rank | Score | Source | Title/Project                        | Author      | Stage  | Tech Stack           | Domain        | Signals
-----|-------|--------|--------------------------------------|-------------|--------|---------------------|---------------|----------
1    | 82.5  | HN     | Show HN: Local-first note app       | builder123  | Beta   | Rust, WASM, CRDTs  | Productivity  | 15 pts, 8 comments
2    | 79.2  | GitHub | quantum-sdk                         | qdev        | MVP    | Python, Q#          | Quantum       | 3★, 12 commits/week
3    | 76.8  | HN     | Ask HN: Validating my biotech tool  | biohacker   | Idea   | Python, BioPython   | Biotech       | 8 pts, 12 comments
4    | 74.3  | GitHub | distributed-game-engine             | gamedev99   | Alpha  | Rust, WebRTC        | Gaming        | 7★, 8 commits/week
5    | 71.5  | HN     | Show HN: CLI for managing k8s       | devops_guy  | v0.1   | Go, Kubernetes      | DevOps        | 22 pts, 5 comments

Score Breakdown (hover for details):
V = Visibility (low is good) | M = Momentum | F = Founder Signals | U = Uniqueness

Extracted Insights:
• Strong founder activity in local-first and quantum computing spaces
• Several single-founder projects with consistent commit velocity
• Biotech emerging as an overlooked domain with technical builders
```

### Additional Output Options

```bash
# Save to JSON for further processing
founderscout --days 7 --output json > founders.json

# Get more detailed analysis
founderscout --days 7 --verbose

# Limit results to top N
founderscout --days 7 --top 10

# Skip LLM analysis (faster, less detailed)
founderscout --days 7 --no-llm
```

---

## Implementation Details

### Configuration

```python
# Environment variables (.env file in project root)
OPENAI_API_KEY=sk-proj-...  # Required for GPT-5 keyword extraction
OPENAI_MODEL=gpt-5          # Default model (can be gpt-5, gpt-5-mini, or gpt-5-nano)

# Optional configuration (~/.founderscout/config.json)
{
    "github_token": "ghp_...",    # Optional: for higher GitHub API rate limits
    "cache_ttl": 3600,            # Cache results for 1 hour
    "max_workers": 10,            # Parallel fetching threads
    "gpt5_model": "gpt-5",        # Override default model
    "reasoning_effort": "medium", # GPT-5 reasoning depth (low/medium/high)
    "verbosity": "concise"        # GPT-5 response detail level
}
```

**Security Note:** The `.env` file is automatically excluded from version control via `.gitignore`

### Core Script Structure

```python
#!/usr/bin/env python3
"""
FounderScout - Automated discovery of overlooked builders
"""

import argparse
import concurrent.futures
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any

import requests
from rich.console import Console
from rich.table import Table
from openai import OpenAI
from dotenv import load_dotenv
import os

class FounderScout:
    def __init__(self, days: int, use_llm: bool = True):
        load_dotenv()  # Load environment variables
        self.days = days
        self.use_llm = use_llm
        self.console = Console()
        self.candidates = []
        
        # Initialize OpenAI client if using LLM
        if self.use_llm:
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = os.getenv('OPENAI_MODEL', 'gpt-5')
        
    def run(self):
        """Main execution flow"""
        with self.console.status("Scanning for overlooked builders..."):
            # Parallel fetch from both sources
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                hn_future = executor.submit(self.scan_hackernews)
                gh_future = executor.submit(self.scan_github)
                
                hn_items = hn_future.result()
                gh_items = gh_future.result()
            
            # Process all items
            all_items = hn_items + gh_items
            
            # Extract insights (with progress bar)
            if self.use_llm:
                all_items = self.extract_insights_batch(all_items)
            
            # Score and rank
            scored_items = [self.score_item(item) for item in all_items]
            self.candidates = sorted(scored_items, 
                                    key=lambda x: x['score']['total'], 
                                    reverse=True)
            
        # Output results
        self.display_results()
    
    def scan_hackernews(self) -> List[Dict]:
        """Fetch recent HN stories"""
        # Implementation from spec above
        pass
    
    def scan_github(self) -> List[Dict]:
        """Fetch recent GitHub repos"""
        # Implementation from spec above
        pass
    
    def extract_insights_batch(self, items: List[Dict]) -> List[Dict]:
        """Extract insights using LLM for all items"""
        # Batch process with progress indicator
        pass
    
    def score_item(self, item: Dict) -> Dict:
        """Calculate overlooked score"""
        # Implementation from spec above
        pass
    
    def display_results(self):
        """Display formatted table of results"""
        table = Table(title="FounderScout - Overlooked Builders")
        
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Score", style="magenta")
        table.add_column("Source", style="yellow")
        table.add_column("Title/Project", style="green")
        table.add_column("Author", style="blue")
        table.add_column("Stage")
        table.add_column("Tech Stack")
        table.add_column("Signals")
        
        for i, item in enumerate(self.candidates[:20], 1):
            table.add_row(
                str(i),
                str(item['score']['total']),
                item['source'],
                item['title'][:40],
                item['author'],
                item.get('stage', 'Unknown'),
                ', '.join(item.get('technologies', [])[:3]),
                item['signals_summary']
            )
        
        self.console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description='Automatically discover overlooked builders on HN and GitHub'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=7,
        help='Number of days to look back (default: 7)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM analysis (faster but less detailed)'
    )
    parser.add_argument(
        '--output',
        choices=['table', 'json', 'csv'],
        default='table',
        help='Output format'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of results to show'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed scoring breakdown'
    )
    
    args = parser.parse_args()
    
    scout = FounderScout(days=args.days, use_llm=not args.no_llm)
    scout.run()

if __name__ == "__main__":
    main()
```

---

## Key Features

### What Makes This Different

1. **Zero Configuration**: Just specify days, everything else is automatic
2. **Intelligent Detection**: LLM-powered understanding of what makes a "founder"
3. **Unified View**: HN and GitHub results in one ranked list
4. **Overlooked Focus**: Specifically optimized to find low-visibility, high-potential builders
5. **Actionable Output**: Every result is someone you could reach out to today

### Performance Considerations

- **Caching**: Results cached for 1 hour to avoid repeated API calls
- **Parallel Fetching**: HN and GitHub scanned simultaneously
- **Rate Limiting**: Respects API limits, uses exponential backoff
- **LLM Optimization**: 
  - Batches requests for efficiency
  - Uses GPT-5 for best quality analysis
  - Can use GPT-5-mini for 5x cost savings with minimal quality loss
  - GPT-5-nano available for ultra-fast simple classification
- **Progressive Loading**: Shows results as they're processed
- **Token Efficiency**: GPT-5 uses 22% fewer tokens than previous models

---

## Installation & Setup

```bash
# Install
pip install founderscout

# Configure (optional)
founderscout --configure

# Run
founderscout --days 7
```

### Requirements

- Python 3.8+
- **Required:** OpenAI API key with GPT-5 access (for keyword extraction)
- Optional: GitHub token for higher rate limits

### Dependencies

```bash
pip install requests python-dotenv openai rich
```

### API Keys Setup

1. **OpenAI API Key (Required):**
   - Create a `.env` file in the project root
   - Add your GPT-5 enabled API key:
   ```
   OPENAI_API_KEY=sk-proj-...
   OPENAI_MODEL=gpt-5
   ```

2. **GitHub Token (Optional, for higher rate limits):**
   - Add to `.env` file:
   ```
   GITHUB_TOKEN=ghp_...
   ```

---

## Example Use Cases

1. **Weekly Founder Radar**
   ```bash
   # Run every Monday morning
   founderscout --days 7 > weekly-founders.md
   ```

2. **Find Very Early Projects**
   ```bash
   # Look at just the last 2 days for the newest stuff
   founderscout --days 2
   ```

3. **Research Mode**
   ```bash
   # Get detailed analysis of last 2 weeks
   founderscout --days 14 --verbose --output json
   ```

4. **Quick Scan**
   ```bash
   # Fast mode without LLM analysis
   founderscout --days 3 --no-llm --top 5
   ```

---

## Why This Approach Works

1. **Founders are everywhere** - They post on HN, create repos, ask for feedback
2. **The best are overlooked** - High-quality builders often have low initial visibility
3. **Timing matters** - Catching projects in their first week is crucial
4. **Patterns are detectable** - Founder language and behavior follows patterns
5. **Simplicity wins** - One parameter makes it actually get used

The tool is designed to be run regularly (daily/weekly) to maintain awareness of emerging builders without any manual searching or configuration.