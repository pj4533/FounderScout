#!/usr/bin/env python3
"""
FounderScout - Find actual founders building things on HN and GitHub
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
        
        self.console.print(f"\n[bold cyan]🔍 FounderScout - Finding Real Founders (Last {self.days} days)[/bold cyan]")
        self.console.print(f"[dim]LLM Analysis: {'Enabled with ' + self.model if self.use_llm else 'Disabled'}[/dim]\n")
        
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
            
            self.console.print(f"[green]✓[/green] Found {len(hn_items)} potential HN founders and {len(gh_items)} GitHub builders")
            
            if not all_items:
                self.console.print("[yellow]No potential founders found in the specified time range[/yellow]")
                return
            
            # Analyze with LLM to determine founder likelihood
            if self.use_llm:
                task = progress.add_task(f"Analyzing {len(all_items)} candidates with {self.model}...", total=len(all_items))
                all_items = self.analyze_founder_signals(all_items, progress, task)
                progress.update(task, completed=True)
            else:
                # Basic analysis without LLM
                for item in all_items:
                    item['why_listed'] = self.get_basic_founder_reason(item)
                    item['founder_confidence'] = self.calculate_basic_founder_score(item)
            
            # Score and rank by founder likelihood
            task = progress.add_task("Ranking by founder probability...", total=None)
            self.candidates = self.score_and_filter_founders(all_items)
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
        """Specifically look for founders on HN"""
        founders = []
        cutoff_time = time.time() - (self.days * 86400)
        
        # Prioritize Show HN and Ask HN posts
        sources = {
            "showstories": 1.5,  # Show HN gets a boost
            "askstories": 1.0,   # Ask HN might have founders
            "newstories": 0.7,   # New stories
            "topstories": 0.5,   # Top stories less likely
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
                
                # Filter for founder signals
                for story in stories:
                    if not story or story.get('time', 0) <= cutoff_time:
                        continue
                    
                    title = story.get('title', '').lower()
                    text = story.get('text', '').lower()
                    
                    # Strong founder indicators
                    is_founder = False
                    founder_signal = ""
                    
                    if 'show hn:' in title:
                        is_founder = True
                        founder_signal = "Show HN post"
                    elif any(phrase in title + text for phrase in [
                        'i built', 'i made', 'i created', 'we built', 'we made',
                        'my startup', 'my project', 'my app', 'our startup',
                        'launching', 'just launched', 'soft launch', 'beta launch',
                        'seeking feedback', 'looking for users', 'early access'
                    ]):
                        is_founder = True
                        founder_signal = "Founder language detected"
                    elif 'ask hn:' in title and any(phrase in title for phrase in [
                        'feedback', 'validate', 'idea', 'startup', 'building', 'launch'
                    ]):
                        is_founder = True
                        founder_signal = "Asking for startup advice"
                    
                    if is_founder:
                        story['source'] = 'hn'
                        story['source_type'] = source
                        story['weight'] = weight
                        story['founder_signal'] = founder_signal
                        founders.append(story)
                        
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]Failed to fetch {source}: {e}[/yellow]")
        
        return founders
    
    def scan_github_for_builders(self) -> List[Dict]:
        """Find people actively building on GitHub"""
        builders = []
        date_filter = (datetime.now() - timedelta(days=self.days)).strftime('%Y-%m-%d')
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        # Look for new projects with good indicators
        queries = [
            f"created:>{date_filter} stars:<20",  # New projects, not yet popular
            f"pushed:>{date_filter} created:>{date_filter}",  # Brand new and active
        ]
        
        for query in queries:
            try:
                response = requests.get(
                    f"{GITHUB_BASE_URL}/search/repositories",
                    params={
                        'q': query,
                        'sort': 'updated',
                        'order': 'desc',
                        'per_page': 50
                    },
                    headers=headers,
                    timeout=10
                )
                
                if response.ok:
                    data = response.json()
                    for repo in data.get('items', []):
                        # Look for signs of a real project
                        description = (repo.get('description') or '').lower()
                        has_description = len(description) > 20
                        has_readme = repo.get('size', 0) > 10  # Proxy for having content
                        
                        # Check for founder-like descriptions
                        founder_keywords = [
                            'my first', 'learning', 'experiment', 'building',
                            'wip', 'work in progress', 'side project', 'hobby',
                            'mvp', 'prototype', 'beta', 'alpha', 'v0'
                        ]
                        
                        is_likely_founder = (
                            has_description and 
                            has_readme and
                            (any(kw in description for kw in founder_keywords) or
                             repo.get('stargazers_count', 0) < 10)
                        )
                        
                        if is_likely_founder:
                            repo['source'] = 'github'
                            repo['time'] = int(datetime.fromisoformat(
                                repo['created_at'].replace('Z', '+00:00')
                            ).timestamp())
                            repo['founder_signal'] = "New active repository"
                            builders.append(repo)
                            
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]GitHub search failed: {e}[/yellow]")
        
        return builders
    
    def analyze_founder_signals(self, items: List[Dict], progress: Progress, task_id) -> List[Dict]:
        """Use LLM to analyze founder likelihood"""
        
        def analyze_item(item):
            try:
                analysis = self.analyze_with_llm(item)
                item['why_listed'] = analysis['why_listed']
                item['founder_confidence'] = analysis['founder_confidence']
                item['tech_stack'] = analysis.get('tech_stack', [])
                item['project_stage'] = analysis.get('stage', 'Unknown')
            except Exception as e:
                if self.verbose:
                    self.console.print(f"[yellow]Analysis failed: {e}[/yellow]")
                item['why_listed'] = item.get('founder_signal', 'Potential founder activity')
                item['founder_confidence'] = 0.5
                item['tech_stack'] = []
                item['project_stage'] = 'Unknown'
            
            progress.update(task_id, advance=1)
            return item
        
        # Process items sequentially for GPT-5 (to avoid rate limits)
        # Can change back to parallel with ThreadPoolExecutor if needed
        for i, item in enumerate(items):
            items[i] = analyze_item(item)
        
        return items
    
    def analyze_with_llm(self, item: Dict) -> Dict:
        """Use GPT-5 to determine if this is a real founder"""
        
        # Prepare context
        if item['source'] == 'hn':
            title = item.get('title', '')
            description = item.get('text', '')
            url = item.get('url', '')
            author = item.get('by', '')
            context = f"HN Post by {author}: {title}\n{description}\nURL: {url}"
        else:  # github
            name = item.get('name', '')
            description = item.get('description', '')
            url = item.get('html_url', '')
            author = item.get('owner', {}).get('login', '')
            context = f"GitHub Repo by {author}: {name}\n{description}\nURL: {url}"
        
        # Remove HTML tags
        context = re.sub('<[^<]+?>', '', context)
        
        prompt = f"""
        Analyze if this is a REAL FOUNDER building something. Be skeptical.
        
        Context: {context[:1500]}
        
        Determine:
        1. Is this actually someone building/launching something? (not just sharing an article)
        2. What are they building? (one sentence)
        3. What stage? (idea/building/launched/established)
        4. Tech stack if mentioned
        5. Confidence score (0-1) that this is a real founder
        
        Return JSON:
        {{
            "is_founder": true/false,
            "why_listed": "One sentence explaining why this person appears to be building something",
            "founder_confidence": 0.0-1.0,
            "tech_stack": ["tech1", "tech2"],
            "stage": "idea/building/launched/established",
            "what_building": "One sentence description"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying real founders and builders. Be skeptical - most HN posts are NOT founders. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ]
                # Removed all optional parameters - GPT-5 works best with defaults
            )
            
            # Debug: show what we got
            content = response.choices[0].message.content
            if not content:
                self.console.print(f"[yellow]Empty response for {author}[/yellow]")
                return {
                    'why_listed': 'Empty LLM response',
                    'founder_confidence': 0.3,
                    'tech_stack': [],
                    'stage': 'Unknown'
                }
            
            # Don't print content in normal mode, too verbose
            
            result = json.loads(content)
            
            # If not a founder, mark it clearly
            if not result.get('is_founder', False):
                result['founder_confidence'] = 0.0
                result['why_listed'] = "Not a founder - " + result.get('why_listed', 'Just sharing content')
            
            return result
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[yellow]LLM error for {author}: {str(e)[:100]}[/yellow]")
            return {
                'why_listed': 'Analysis error',
                'founder_confidence': 0.3,
                'tech_stack': [],
                'stage': 'Unknown'
            }
    
    def get_basic_founder_reason(self, item: Dict) -> str:
        """Get founder reason without LLM"""
        if item['source'] == 'hn':
            title = item.get('title', '').lower()
            if 'show hn:' in title:
                return "Show HN - presenting their project"
            elif 'i built' in title or 'i made' in title:
                return "Built something and sharing it"
            elif 'ask hn:' in title:
                return "Asking for feedback on their idea"
            else:
                return item.get('founder_signal', 'Possible founder activity')
        else:
            return "New GitHub project with activity"
    
    def calculate_basic_founder_score(self, item: Dict) -> float:
        """Calculate founder confidence without LLM"""
        score = 0.3  # Base score
        
        if item['source'] == 'hn':
            title = item.get('title', '').lower()
            text = item.get('text', '').lower()
            combined = title + ' ' + text
            
            # Strong signals
            if 'show hn:' in title:
                score += 0.4
            if any(x in combined for x in ['i built', 'i made', 'we built', 'we made']):
                score += 0.3
            if any(x in combined for x in ['launch', 'beta', 'mvp', 'prototype']):
                score += 0.2
                
            # Weak signals
            if item.get('descendants', 0) > 0:  # Has comments
                score += 0.1
                
        else:  # GitHub
            if item.get('stargazers_count', 0) < 10:
                score += 0.2  # Low visibility
            if item.get('description'):
                score += 0.2
        
        return min(score, 1.0)
    
    def score_and_filter_founders(self, items: List[Dict]) -> List[Dict]:
        """Score and filter to only show likely founders"""
        
        scored_items = []
        for item in items:
            # Skip if definitely not a founder
            if item.get('founder_confidence', 0) < 0.2:
                continue
            
            # Calculate overlooked score
            if item['source'] == 'hn':
                visibility = item.get('score', 0)  # HN points
                activity = item.get('descendants', 0)  # Comments
            else:
                visibility = item.get('stargazers_count', 0)
                activity = item.get('forks_count', 0) + item.get('watchers_count', 0)
            
            # Lower visibility is better for finding overlooked founders
            visibility_score = max(0, 1 - (visibility / 50))
            
            # Higher founder confidence is better
            founder_score = item.get('founder_confidence', 0.5)
            
            # Activity shows engagement
            activity_score = min(activity / 10, 1.0)
            
            # Weight towards actual founders
            total_score = (
                founder_score * 0.6 +  # Most important: are they a founder?
                visibility_score * 0.3 +  # Overlooked is good
                activity_score * 0.1  # Some activity is good
            )
            
            item['total_score'] = round(total_score, 2)
            scored_items.append(item)
        
        # Sort by score and take top N
        scored_items.sort(key=lambda x: x['total_score'], reverse=True)
        return scored_items[:self.top_n]
    
    def display_founders(self):
        """Display founders in a clear, readable table"""
        
        if not self.candidates:
            self.console.print("[yellow]No founders found. Try increasing the time range or adjusting filters.[/yellow]")
            return
        
        # Create table with better column sizing
        table = Table(
            title=f"\n[bold]FounderScout - Real Builders & Founders[/bold]\n[dim]Last {self.days} days | {len(self.candidates)} founders found[/dim]",
            show_header=True,
            header_style="bold cyan",
            title_justify="center",
            expand=True  # Allow table to expand to terminal width
        )
        
        # Add columns without fixed widths to let Rich handle sizing
        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Score", style="magenta", no_wrap=True)
        table.add_column("Platform", style="yellow", no_wrap=True)
        table.add_column("Who", style="bright_blue", overflow="ellipsis")
        table.add_column("What They're Building", style="green", overflow="ellipsis")
        table.add_column("Why Listed", style="white", overflow="ellipsis")
        table.add_column("Activity", style="dim", no_wrap=True)
        
        # Add rows
        for i, item in enumerate(self.candidates, 1):
            if item['source'] == 'hn':
                who = item.get('by', 'Unknown')
                what = item.get('title', 'No title')
                if 'show hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()  # Remove "Show HN:" prefix
                activity = f"{item.get('score', 0)}pts {item.get('descendants', 0)}cmt"
            else:
                who = item.get('owner', {}).get('login', 'Unknown')
                what = item.get('name', 'No name')
                if item.get('description'):
                    what = f"{what}: {item.get('description', '')[:50]}"
                activity = f"{item.get('stargazers_count', 0)}★ {item.get('forks_count', 0)}forks"
            
            why = item.get('why_listed', 'Potential founder')[:45]
            score = f"{item.get('total_score', 0):.1f}"
            
            # Truncate long strings for better display
            what = what[:50] if len(what) > 50 else what
            
            # Add row
            table.add_row(
                str(i),
                score,
                item['source'].upper(),
                who[:12],
                what,
                why,
                activity
            )
        
        self.console.print(table)
        
        # Add helpful context
        self.console.print("\n[dim]Score: Founder Confidence (0-1) × Low Visibility Bonus × Activity[/dim]")
        self.console.print("[dim]Higher scores = more likely to be overlooked founders building something interesting[/dim]")
        
        # Show details for top 3 if verbose
        if self.verbose and len(self.candidates) > 0:
            self.console.print("\n[bold]Top Founder Details:[/bold]")
            for item in self.candidates[:3]:
                self.console.print(f"\n[green]→ {item.get('by', item.get('owner', {}).get('login', 'Unknown'))}[/green]")
                self.console.print(f"  [white]{item.get('why_listed', 'No details')}[/white]")
                if item.get('tech_stack'):
                    self.console.print(f"  Tech: {', '.join(item['tech_stack'][:5])}")
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
            "founders": []
        }
        
        for i, item in enumerate(self.candidates, 1):
            founder = {
                "rank": i,
                "score": item.get('total_score', 0),
                "platform": item['source'],
                "why_listed": item.get('why_listed', ''),
                "founder_confidence": item.get('founder_confidence', 0)
            }
            
            if item['source'] == 'hn':
                founder.update({
                    "author": item.get('by'),
                    "title": item.get('title'),
                    "url": f"https://news.ycombinator.com/item?id={item['id']}",
                    "hn_points": item.get('score', 0),
                    "comments": item.get('descendants', 0)
                })
            else:
                founder.update({
                    "author": item.get('owner', {}).get('login'),
                    "repo": item.get('name'),
                    "description": item.get('description'),
                    "url": item.get('html_url'),
                    "stars": item.get('stargazers_count', 0),
                    "forks": item.get('forks_count', 0)
                })
            
            output['founders'].append(founder)
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.console.print(f"\n[green]✓ Results exported to {filename}[/green]")

def main():
    parser = argparse.ArgumentParser(
        description='Find real founders and builders on HN and GitHub'
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
        help='Skip GPT-5 analysis (faster but less accurate)'
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