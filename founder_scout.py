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
    def __init__(self, days: int, use_llm: bool = True, top_n: int = 20, verbose: bool = False, output_format: str = 'table'):
        self.days = days
        self.use_llm = use_llm
        self.top_n = top_n
        self.verbose = verbose
        self.output_format = output_format
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
        
        # Cast wider net without language bias
        queries = [
            f"created:>{date_filter} stars:<100",  # New projects, not yet viral
            f"pushed:>{date_filter} stars:<50",  # Recently active, overlooked
            f"created:>{date_filter} stars:0..5",  # Brand new with no traction
            f"created:>{date_filter} size:>1000",  # Substantial new projects
            f"pushed:>{date_filter} size:>5000 stars:<20",  # Big effort, low recognition
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
        """Score projects based on balance between engagement and passion"""
        
        scored_items = []
        for item in items:
            scores = {}
            
            if item['source'] == 'hn':
                points = item.get('score', 0)
                comments = item.get('descendants', 0)
                text = item.get('text', '')
                title = item.get('title', '')
                
                # Engagement score (normalized from 0-1)
                if points == 0:
                    engagement_norm = 0.05
                elif points < 10:
                    engagement_norm = 0.2
                elif points < 50:
                    engagement_norm = 0.4
                elif points < 100:
                    engagement_norm = 0.6
                elif points < 500:
                    engagement_norm = 0.8
                else:
                    engagement_norm = 1.0
                
                scores['engagement'] = engagement_norm
                
                # Passion score based on effort/substance
                passion_score = 0
                
                # Text length indicates effort in writing
                text_len = len(text)
                if text_len > 2000:
                    passion_score += 1.0  # Very long, detailed post
                elif text_len > 1000:
                    passion_score += 0.8
                elif text_len > 500:
                    passion_score += 0.6
                elif text_len > 200:
                    passion_score += 0.4
                elif text_len > 50:
                    passion_score += 0.2
                else:
                    passion_score += 0.05  # Just a link or very short
                
                # Show HN gets passion bonus (they built something)
                if 'show hn:' in title.lower():
                    passion_score = min(passion_score + 0.3, 1.0)
                
                # Has URL but also text = explaining their work
                if item.get('url') and text_len > 100:
                    passion_score = min(passion_score + 0.2, 1.0)
                
                scores['passion'] = passion_score
                    
            else:  # GitHub
                stars = item.get('stargazers_count', 0)
                forks = item.get('forks_count', 0)
                size = item.get('size', 0)
                description = item.get('description', '')
                
                # Engagement score (normalized)
                if stars == 0:
                    engagement_norm = 0.05
                elif stars < 10:
                    engagement_norm = 0.2
                elif stars < 50:
                    engagement_norm = 0.4
                elif stars < 200:
                    engagement_norm = 0.6
                elif stars < 1000:
                    engagement_norm = 0.8
                else:
                    engagement_norm = 1.0
                
                scores['engagement'] = engagement_norm
                
                # Passion score based on repo substance
                passion_score = 0
                
                # Repository size indicates amount of work
                if size > 10000:
                    passion_score += 0.8  # Large, substantial project
                elif size > 1000:
                    passion_score += 0.6
                elif size > 100:
                    passion_score += 0.4
                elif size > 10:
                    passion_score += 0.2
                else:
                    passion_score += 0.05  # Nearly empty repo
                
                # Good description shows care
                if len(description) > 100:
                    passion_score = min(passion_score + 0.3, 1.0)
                elif len(description) > 30:
                    passion_score = min(passion_score + 0.1, 1.0)
                
                # Has README (we can infer from size usually)
                if size > 50:  # Likely has documentation
                    passion_score = min(passion_score + 0.1, 1.0)
                
                scores['passion'] = min(passion_score, 1.0)
            
            # Calculate overlooked score (inverse of engagement)
            scores['overlooked'] = 1.0 - scores['engagement']
            
            # Balance score: reward high passion with low engagement
            passion_weight = scores['passion']
            overlooked_weight = scores['overlooked']
            
            # The ideal is high passion + high overlooked
            balance_score = (passion_weight * 0.6) + (overlooked_weight * 0.4)
            
            # Bonus for extreme cases: very high passion with very low engagement
            if scores['passion'] > 0.7 and scores['overlooked'] > 0.7:
                balance_score = min(balance_score * 1.2, 1.0)
            
            item['total_score'] = round(balance_score, 2)
            item['score_breakdown'] = scores
            scored_items.append(item)
        
        # Sort by score
        scored_items.sort(key=lambda x: x['total_score'], reverse=True)
        return scored_items
    
    def display_founders(self):
        """Display builders in a clear, readable format"""
        
        if not self.candidates:
            self.console.print("[yellow]No interesting projects found. Try increasing the time range.[/yellow]")
            return
        
        # Limit display to top N
        display_items = self.candidates[:self.top_n]
        
        if self.output_format == 'compact':
            self.display_compact(display_items)
        else:
            self.display_cards(display_items)
    
    def display_compact(self, display_items):
        """Display in compact one-line format"""
        # Header
        self.console.print(f"\n[bold cyan]FounderScout - Overlooked Builders[/bold cyan] (Last {self.days} days)\n")
        
        for i, item in enumerate(display_items, 1):
            if item['source'] == 'hn':
                who = item.get('by', 'Unknown')
                what = item.get('title', 'No title')
                if 'show hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                elif 'ask hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                url = f"https://news.ycombinator.com/item?id={item['id']}"
            else:
                who = item.get('owner', {}).get('login', 'Unknown')
                what = item.get('name', 'No name')
                url = item.get('html_url', 'N/A')
            
            score_color = "bright_magenta" if item.get('total_score', 0) > 0.7 else "magenta"
            source = item['source'].upper()
            
            # One-line format
            self.console.print(f"{i:2}. [{score_color}]{item.get('total_score', 0):.2f}[/{score_color}] [{source:6}] [bright_blue]{who:15}[/bright_blue] [green]{what[:60]:60}[/green] [dim]{url}[/dim]")
        
        self.console.print("\n[dim]Higher scores = more overlooked gems[/dim]")
    
    def display_cards(self, display_items):
        """Display in card format (default)"""
        # Header
        self.console.print(f"\n[bold cyan]🔍 FounderScout - Overlooked Builders & Projects[/bold cyan]")
        self.console.print(f"[dim]Last {self.days} days | Top {len(display_items)} of {len(self.candidates)} found[/dim]\n")
        
        # Display each item in a card-like format
        for i, item in enumerate(display_items, 1):
            # Rank and Score
            score_color = "bright_magenta" if item.get('total_score', 0) > 0.7 else "magenta"
            self.console.print(f"[bold]{i}.[/bold] [bold {score_color}]Score: {item.get('total_score', 0):.2f}[/bold {score_color}]", end="  ")
            
            # Source badge
            source_badge = f"[yellow][{item['source'].upper()}][/yellow]"
            self.console.print(source_badge)
            
            # Project title and creator
            if item['source'] == 'hn':
                who = item.get('by', 'Unknown')
                what = item.get('title', 'No title')
                if 'show hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                elif 'ask hn:' in what.lower():
                    what = what.split(':', 1)[1].strip()
                stats = f"{item.get('score', 0)}pts, {item.get('descendants', 0)}cmt"
                url = f"https://news.ycombinator.com/item?id={item['id']}"
            else:
                who = item.get('owner', {}).get('login', 'Unknown')
                what = item.get('name', 'No name')
                lang = item.get('language', '')
                stats = f"{item.get('stargazers_count', 0)}★"
                if lang:
                    stats += f", {lang}"
                url = item.get('html_url', 'N/A')
            
            # Title line
            self.console.print(f"   [bold green]{what}[/bold green] by [bright_blue]{who}[/bright_blue]")
            
            # Description/Summary
            if item.get('source') == 'github' and item.get('description'):
                self.console.print(f"   [white]{item.get('description')}[/white]")
            
            # What's interesting (from LLM or builder signal)
            interesting = item.get('summary', item.get('builder_signal', 'Interesting project'))
            if interesting:
                self.console.print(f"   [italic]💡 {interesting}[/italic]")
            
            # Keywords if available
            if self.use_llm and item.get('keywords'):
                keywords = item.get('keywords', [])
                if keywords:
                    self.console.print(f"   [cyan]Tags: {', '.join(keywords)}[/cyan]")
            
            # Stats and link
            self.console.print(f"   [dim]{stats} | {url}[/dim]")
            
            # Separator between items (except last)
            if i < len(display_items):
                self.console.print()
        
        # Score explanation
        self.console.print("\n[dim]Scoring: Balance of Passion (effort/substance) vs Engagement (stars/points)[/dim]")
        self.console.print("[dim]Higher scores = passionate projects with low visibility (overlooked gems)[/dim]")
        
        # Show details for top 3 if verbose
        if self.verbose and len(display_items) > 0:
            self.console.print("\n[bold]Top Project Details:[/bold]")
            for item in display_items[:3]:
                self.console.print(f"\n[green]→ {item.get('by', item.get('owner', {}).get('login', 'Unknown'))}[/green]")
                
                # Show score breakdown
                if 'score_breakdown' in item:
                    scores = item['score_breakdown']
                    self.console.print(f"  Scores: Passion={scores.get('passion', 0):.2f}, "
                                      f"Engagement={scores.get('engagement', 0):.2f}, "
                                      f"Overlooked={scores.get('overlooked', 0):.2f}")
                
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
        choices=['table', 'json', 'compact'],
        default='table',
        help='Output format: table (default card view), compact (one-line), json'
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
        verbose=args.verbose,
        output_format=args.output
    )
    scout.run()
    
    # Export if requested
    if args.output == 'json':
        scout.export_json()

if __name__ == "__main__":
    main()