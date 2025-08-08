#!/usr/bin/env python3
"""
FounderScout - Discover overlooked builders and interesting projects on HN and GitHub
"""

import argparse
import concurrent.futures
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any, Optional
import sys
import warnings
import re
import os

import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from openai import OpenAI
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Constants
HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
GITHUB_BASE_URL = "https://api.github.com"

class FounderScout:
    def __init__(self, days: int, use_llm: bool = True, top_n: int = 20, verbose: bool = False):
        self.days = days
        self.use_llm = use_llm
        self.top_n = top_n
        self.verbose = verbose
        self.console = Console()
        self.candidates = []
        
        # Initialize OpenAI client if using LLM
        if self.use_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.console.print("[red]Error: OPENAI_API_KEY not found in .env file[/red]")
                self.console.print("Please add your OpenAI API key to the .env file")
                sys.exit(1)
            
            self.openai_client = OpenAI(api_key=api_key)
            self.model = os.getenv('OPENAI_MODEL', 'gpt-5')
            
            # Validate model name
            if self.model not in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']:
                self.console.print(f"[yellow]Warning: Unknown model '{self.model}', defaulting to gpt-5[/yellow]")
                self.model = 'gpt-5'
        
        # GitHub token (optional)
        self.github_token = os.getenv('GITHUB_TOKEN')
        
    def run(self):
        """Main execution flow"""
        start_time = time.time()
        
        self.console.print(f"\n[bold cyan]🔍 FounderScout - Discovering Overlooked Builders (Last {self.days} days)[/bold cyan]")
        self.console.print(f"[dim]Summary Generation: {'Enabled with ' + self.model if self.use_llm else 'Disabled'}[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Parallel fetch from both sources
            task = progress.add_task("Searching for founders on HN and GitHub...", total=None)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                hn_future = executor.submit(self.scan_hackernews_for_founders)
                gh_future = executor.submit(self.scan_github_for_builders)
                
                hn_items = hn_future.result()
                gh_items = gh_future.result()
            
            progress.update(task, completed=True)
            
            # Combine all items
            all_items = hn_items + gh_items
            
            self.console.print(f"[green]✓[/green] Found {len(hn_items)} HN projects and {len(gh_items)} GitHub repositories")
            
            if not all_items:
                self.console.print("[yellow]No interesting projects found in the specified time range[/yellow]")
                return
            
            # Score and rank by overlooked/interesting factors
            task = progress.add_task("Scoring for overlooked gems...", total=None)
            self.candidates = self.score_and_rank_projects(all_items)
            progress.update(task, completed=True)
            
            # Post-process with LLM for better display (top N only)
            if self.use_llm and self.candidates:
                task = progress.add_task(f"Generating summaries with {self.model}...", total=1)
                self.enrich_with_llm(self.candidates[:self.top_n])
                progress.update(task, completed=True)
        
        # Display results
        self.display_founders()
        
        elapsed = time.time() - start_time
        self.console.print(f"\n[dim]Completed in {elapsed:.1f} seconds[/dim]")
    
    def fetch_hn_item(self, item_id: int) -> Optional[Dict]:
        """Fetch a single HN item"""
        try:
            response = requests.get(f"{HN_BASE_URL}/item/{item_id}.json", timeout=5)
            return response.json() if response.ok else None
        except:
            return None
    
    def scan_hackernews_for_founders(self) -> List[Dict]:
        """Scout for interesting builders and projects on HN"""
        builders = []
        cutoff_time = time.time() - (self.days * 86400)
        
        # Cast a wider net for interesting projects
        sources = {
            "showstories": 1.0,  # Show HN - people building things
            "askstories": 1.0,   # Ask HN - seeking advice/validation
            "newstories": 1.0,   # New stories - fresh content
            "topstories": 0.5,   # Top stories - may be too popular
        }
        
        for source, weight in sources.items():
            try:
                response = requests.get(f"{HN_BASE_URL}/{source}.json", timeout=5)
                if not response.ok:
                    continue
                    
                story_ids = response.json()[:200]  # Check more stories
                
                # Fetch stories in batches
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    stories = list(executor.map(self.fetch_hn_item, story_ids))
                
                # Look for any builder signals (more inclusive)
                for story in stories:
                    if not story or story.get('time', 0) <= cutoff_time:
                        continue
                    
                    title = story.get('title', '').lower()
                    text = story.get('text', '').lower()
                    combined = title + ' ' + text
                    
                    # Builder indicators (more inclusive)
                    builder_signal = ""
                    
                    # Priority signals
                    if 'show hn:' in title:
                        builder_signal = "Show HN - presenting project"
                    elif any(phrase in combined for phrase in [
                        'i built', 'i made', 'i created', 'we built', 'we made',
                        'my project', 'our project', 'side project', 'weekend project',
                        'open source', 'open-source', 'released', 'announcing'
                    ]):
                        builder_signal = "Building/releasing something"
                    elif any(phrase in combined for phrase in [
                        'working on', 'developing', 'creating', 'building',
                        'launched', 'launching', 'shipped', 'beta', 'alpha', 'mvp',
                        'prototype', 'proof of concept', 'experiment', 'tool', 'app'
                    ]):
                        builder_signal = "Active development"
                    elif 'ask hn:' in title and any(phrase in title for phrase in [
                        'feedback', 'validate', 'idea', 'startup', 'building', 
                        'project', 'launch', 'advice', 'thoughts'
                    ]):
                        builder_signal = "Seeking validation/feedback"
                    elif any(phrase in combined for phrase in [
                        'github.com', 'gitlab.com', 'demo', 'try it',
                        'check it out', 'live at', 'available at'
                    ]):
                        builder_signal = "Sharing project link"
                    
                    # Accept more posts, let scoring determine quality
                    if builder_signal:
                        story['source'] = 'hn'
                        story['source_type'] = source
                        story['source_weight'] = weight
                        story['builder_signal'] = builder_signal
                        builders.append(story)
                        
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]Failed to fetch {source}: {e}[/yellow]")
        
        return builders
    
    def scan_github_for_builders(self) -> List[Dict]:
        """Scout for interesting projects on GitHub"""
        builders = []
        date_filter = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        # Cast wider net for interesting projects
        queries = [
            f"created:>{date_filter} stars:<100",  # New projects, not yet viral
            f"pushed:>{date_filter} stars:<50",  # Recently active, overlooked
            f"created:>{date_filter} language:Rust",  # New Rust projects (often interesting)
            f"created:>{date_filter} language:Zig",  # New Zig projects (cutting edge)
            f"created:>{date_filter} topic:tool",  # Developer tools
            f"created:>{date_filter} topic:cli",  # CLI tools
        ]
        
        seen_repos = set()  # Avoid duplicates
        
        for query in queries:
            try:
                response = requests.get(
                    f"{GITHUB_BASE_URL}/search/repositories",
                    params={
                        'q': query,
                        'sort': 'updated',
                        'order': 'desc',
                        'per_page': 30
                    },
                    headers=headers,
                    timeout=10
                )
                
                if response.ok:
                    data = response.json()
                    for repo in data.get('items', []):
                        # Skip if we've already seen this repo
                        repo_id = repo.get('id')
                        if repo_id in seen_repos:
                            continue
                        seen_repos.add(repo_id)
                        
                        # Look for signs of an interesting project
                        description = (repo.get('description') or '')
                        has_description = len(description) > 10
                        
                        # More inclusive - any repo with description and content
                        if has_description and repo.get('size', 0) > 5:
                            repo['source'] = 'github'
                            repo['time'] = int(datetime.fromisoformat(
                                repo['created_at'].replace('Z', '+00:00')
                            ).timestamp())
                            
                            # Categorize the signal
                            desc_lower = description.lower()
                            if any(kw in desc_lower for kw in ['experiment', 'learning', 'toy', 'fun']):
                                repo['builder_signal'] = "Experimental project"
                            elif any(kw in desc_lower for kw in ['tool', 'cli', 'library', 'framework']):
                                repo['builder_signal'] = "Developer tool"
                            elif any(kw in desc_lower for kw in ['app', 'application', 'platform']):
                                repo['builder_signal'] = "Application/platform"
                            elif any(kw in desc_lower for kw in ['game', 'puzzle', 'simulator']):
                                repo['builder_signal'] = "Game/entertainment"
                            else:
                                repo['builder_signal'] = "Active project"
                            
                            builders.append(repo)
                            
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]GitHub search failed: {e}[/yellow]")
        
        return builders
    
    def enrich_with_llm(self, items: List[Dict]):
        """Post-process top results with LLM for keywords and summaries"""
        
        # Prepare batch request
        batch_context = []
        for i, item in enumerate(items):
            if item['source'] == 'hn':
                title = item.get('title', '')
                text = item.get('text', '')[:500]  # Limit text
                url = item.get('url', '')
                context = f"Item {i+1} (HN): {title}\n{text}\nURL: {url}"
            else:  # github
                name = item.get('name', '')
                description = item.get('description', '')
                url = item.get('html_url', '')
                lang = item.get('language', 'Unknown')
                context = f"Item {i+1} (GitHub): {name}\n{description}\nLanguage: {lang}\nURL: {url}"
            batch_context.append(context)
        
        # Create batch prompt
        prompt = f"""Analyze these {len(items)} projects and extract keywords and create a brief summary for each.

For each item, provide:
1. Keywords: 3-5 technical/domain keywords
2. Summary: One sentence about what makes this interesting or unique
3. Vibe: The creator's energy (passionate/exploratory/pragmatic/playful/serious)

{chr(10).join(batch_context[:self.top_n])}

Return a JSON array with one object per item:
[
  {{
    "item": 1,
    "keywords": ["keyword1", "keyword2"],
    "summary": "Brief summary of what makes this interesting",
    "vibe": "passionate/exploratory/pragmatic/playful/serious"
  }},
  ...
]"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You analyze projects to find what makes them interesting or unique. Focus on the unusual, overlooked, or passionate aspects."},
                    {"role": "user", "content": prompt}
                ]
                # Removed temperature parameter as it's not supported by all models
            )
            
            content = response.choices[0].message.content
            if content:
                if self.verbose:
                    self.console.print(f"[dim]LLM response received: {len(content)} chars[/dim]")
                results = json.loads(content)
                # Map results back to items
                for result in results:
                    idx = result['item'] - 1
                    if 0 <= idx < len(items):
                        items[idx]['keywords'] = result.get('keywords', [])
                        items[idx]['summary'] = result.get('summary', items[idx].get('builder_signal', ''))
                        items[idx]['vibe'] = result.get('vibe', 'unknown')
            else:
                if self.verbose:
                    self.console.print(f"[yellow]Empty LLM response[/yellow]")
        except json.JSONDecodeError as e:
            if self.verbose:
                self.console.print(f"[yellow]JSON decode error: {e}[/yellow]")
                self.console.print(f"[yellow]Response: {content[:500]}...[/yellow]")
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]LLM enrichment failed: {e}[/yellow]")
            # Fall back to basic enrichment
            for item in items:
                item['keywords'] = []
                item['summary'] = item.get('builder_signal', 'Interesting project')
                item['vibe'] = 'unknown'
    
    # Removed analyze_with_llm - no longer needed
    
    # Removed get_basic_founder_reason and calculate_basic_founder_score - no longer needed
    
    def score_and_rank_projects(self, items: List[Dict]) -> List[Dict]:
        """Score projects based on how overlooked and interesting they are"""
        
        scored_items = []
        for item in items:
            scores = {}
            
            if item['source'] == 'hn':
                points = item.get('score', 0)
                comments = item.get('descendants', 0)
                
                # Overlooked score (low visibility is good)
                if points < 5:
                    scores['overlooked'] = 1.0
                elif points < 20:
                    scores['overlooked'] = 0.7
                elif points < 50:
                    scores['overlooked'] = 0.4
                else:
                    scores['overlooked'] = max(0, 1 - (points / 200))
                
                # Engagement score (some discussion is good, too much means not overlooked)
                if 0 < comments <= 10:
                    scores['engagement'] = 0.8
                elif comments <= 30:
                    scores['engagement'] = 0.5
                else:
                    scores['engagement'] = 0.2
                
                # Passion indicators
                title_lower = item.get('title', '').lower()
                text_lower = item.get('text', '').lower()
                combined = title_lower + ' ' + text_lower
                
                passion_words = ['excited', 'love', 'passionate', 'obsessed', 'finally',
                                  'proud', 'happy', 'spent months', 'spent years', 'dream']
                scores['passion'] = min(sum(1 for word in passion_words if word in combined) * 0.3, 1.0)
                
                # Weird/unique factor
                weird_words = ['weird', 'unusual', 'strange', 'different', 'unique', 'niche',
                               'experimental', 'crazy', 'stupid', 'useless', 'fun', 'toy']
                scores['weird'] = min(sum(1 for word in weird_words if word in combined) * 0.25, 1.0)
                
                # Show HN bonus
                if 'show hn:' in title_lower:
                    scores['builder'] = 0.8
                elif 'ask hn:' in title_lower:
                    scores['builder'] = 0.3
                else:
                    scores['builder'] = 0.5
                    
            else:  # GitHub
                stars = item.get('stargazers_count', 0)
                forks = item.get('forks_count', 0)
                size = item.get('size', 0)
                
                # Overlooked score
                if stars < 10:
                    scores['overlooked'] = 1.0
                elif stars < 50:
                    scores['overlooked'] = 0.7
                elif stars < 200:
                    scores['overlooked'] = 0.4
                else:
                    scores['overlooked'] = max(0, 1 - (stars / 1000))
                
                # Completeness (bigger repos might be more complete)
                if size > 1000:
                    scores['completeness'] = 0.8
                elif size > 100:
                    scores['completeness'] = 0.5
                else:
                    scores['completeness'] = 0.2
                
                # Activity (forks show interest)
                scores['engagement'] = min(forks * 0.2, 1.0)
                
                # Language bonus for interesting techs
                interesting_langs = ['Rust', 'Zig', 'Nim', 'Haskell', 'Erlang', 'Elixir',
                                     'Crystal', 'V', 'Odin', 'Julia', 'OCaml']
                lang = item.get('language', '')
                if lang in interesting_langs:
                    scores['weird'] = 0.7
                else:
                    scores['weird'] = 0.2
                
                scores['builder'] = 0.6  # GitHub repos show building
                scores['passion'] = 0.3  # Hard to detect passion from repo alone
            
            # Calculate weighted total
            weights = {
                'overlooked': 0.35,  # Most important - finding hidden gems
                'weird': 0.20,       # Unique/interesting projects
                'passion': 0.15,     # Creator enthusiasm
                'builder': 0.15,     # Actually building something
                'engagement': 0.10,  # Some activity/interest
                'completeness': 0.05 # For GitHub repos
            }
            
            total = sum(scores.get(factor, 0) * weight for factor, weight in weights.items())
            item['total_score'] = round(total, 2)
            item['score_breakdown'] = scores
            scored_items.append(item)
        
        # Sort by score
        scored_items.sort(key=lambda x: x['total_score'], reverse=True)
        return scored_items
    
    def display_founders(self):
        """Display builders in a clear, readable table"""
        
        if not self.candidates:
            self.console.print("[yellow]No interesting projects found. Try increasing the time range.[/yellow]")
            return
        
        # Limit display to top N
        display_items = self.candidates[:self.top_n]
        
        # Create table
        table = Table(
            title=f"\n[bold]FounderScout - Overlooked Builders & Projects[/bold]\n[dim]Last {self.days} days | Top {len(display_items)} of {len(self.candidates)} found[/dim]",
            show_header=True,
            header_style="bold cyan",
            title_justify="center",
            expand=True
        )
        
        # Add columns
        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Score", style="magenta", no_wrap=True)
        table.add_column("Source", style="yellow", no_wrap=True)
        table.add_column("Creator", style="bright_blue", overflow="ellipsis")
        table.add_column("Project", style="green", overflow="ellipsis")
        table.add_column("What's Interesting", style="white", overflow="ellipsis")
        table.add_column("Stats", style="dim", no_wrap=True)
        if self.use_llm:
            table.add_column("Keywords", style="cyan", overflow="ellipsis")
        
        # Add rows
        for i, item in enumerate(display_items, 1):
            if item['source'] == 'hn':
                who = item.get('by', 'Unknown')
                what = item.get('title', 'No title')
                if 'show hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                elif 'ask hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                stats = f"{item.get('score', 0)}pts {item.get('descendants', 0)}cmt"
            else:
                who = item.get('owner', {}).get('login', 'Unknown')
                what = item.get('name', 'No name')
                if item.get('description'):
                    desc = item.get('description', '')[:40]
                    what = f"{what}: {desc}"
                lang = item.get('language', '')
                if lang:
                    stats = f"{item.get('stargazers_count', 0)}★ {lang[:8]}"
                else:
                    stats = f"{item.get('stargazers_count', 0)}★"
            
            # Use LLM summary if available, otherwise builder signal
            interesting = item.get('summary', item.get('builder_signal', 'Interesting project'))[:50]
            score = f"{item.get('total_score', 0):.2f}"
            
            # Truncate for display
            what = what[:45] if len(what) > 45 else what
            
            # Build row
            row = [
                str(i),
                score,
                item['source'].upper(),
                who[:12],
                what,
                interesting,
                stats
            ]
            
            # Add keywords if available
            if self.use_llm:
                keywords = item.get('keywords', [])
                if keywords:
                    row.append(', '.join(keywords[:3]))
                else:
                    row.append('')
            
            table.add_row(*row)
        
        self.console.print(table)
        
        # Score explanation
        self.console.print("\n[dim]Scoring: Overlooked (35%) + Unique/Weird (20%) + Passion (15%) + Builder (15%) + Engagement (10%)[/dim]")
        self.console.print("[dim]Higher scores = more likely to be an overlooked gem worth exploring[/dim]")
        
        # Show details for top 3 if verbose
        if self.verbose and len(display_items) > 0:
            self.console.print("\n[bold]Top Project Details:[/bold]")
            for item in display_items[:3]:
                self.console.print(f"\n[green]→ {item.get('by', item.get('owner', {}).get('login', 'Unknown'))}[/green]")
                
                # Show score breakdown
                if 'score_breakdown' in item:
                    scores = item['score_breakdown']
                    self.console.print(f"  Scores: Overlooked={scores.get('overlooked', 0):.1f}, "
                                      f"Weird={scores.get('weird', 0):.1f}, "
                                      f"Passion={scores.get('passion', 0):.1f}")
                
                # Show vibe if available
                if 'vibe' in item:
                    self.console.print(f"  Vibe: {item['vibe']}")
                
                # Show keywords
                if item.get('keywords'):
                    self.console.print(f"  Keywords: {', '.join(item['keywords'])}")
                
                # Show link
                if item['source'] == 'hn':
                    self.console.print(f"  Link: https://news.ycombinator.com/item?id={item['id']}")
                else:
                    self.console.print(f"  Link: {item.get('html_url', 'N/A')}")
    
    def export_json(self, filename: str = "founders.json"):
        """Export results to JSON"""
        output = {
            "generated_at": datetime.now().isoformat(),
            "search_params": {
                "days": self.days,
                "max_results": self.top_n
            },
            "projects": []
        }
        
        for i, item in enumerate(self.candidates[:self.top_n], 1):
            project = {
                "rank": i,
                "score": item.get('total_score', 0),
                "platform": item['source'],
                "summary": item.get('summary', item.get('builder_signal', '')),
                "keywords": item.get('keywords', []),
                "vibe": item.get('vibe', 'unknown'),
                "score_breakdown": item.get('score_breakdown', {})
            }
            
            if item['source'] == 'hn':
                project.update({
                    "author": item.get('by'),
                    "title": item.get('title'),
                    "url": f"https://news.ycombinator.com/item?id={item['id']}",
                    "hn_points": item.get('score', 0),
                    "comments": item.get('descendants', 0)
                })
            else:
                project.update({
                    "author": item.get('owner', {}).get('login'),
                    "repo": item.get('name'),
                    "description": item.get('description'),
                    "url": item.get('html_url'),
                    "language": item.get('language'),
                    "stars": item.get('stargazers_count', 0),
                    "forks": item.get('forks_count', 0)
                })
            
            output['projects'].append(project)
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.console.print(f"\n[green]✓ Results exported to {filename}[/green]")

def main():
    parser = argparse.ArgumentParser(
        description='Discover overlooked builders and interesting projects on HN and GitHub'
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
        help='Skip LLM keyword/summary generation (faster but less detailed)'
    )
    parser.add_argument(
        '--output',
        choices=['table', 'json'],
        default='table',
        help='Output format (default: table)'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of results to show (default: 20)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    args = parser.parse_args()
    
    # Run scout
    scout = FounderScout(
        days=args.days, 
        use_llm=not args.no_llm,
        top_n=args.top,
        verbose=args.verbose
    )
    scout.run()
    
    # Export if requested
    if args.output == 'json':
        scout.export_json()

if __name__ == "__main__":
    main()