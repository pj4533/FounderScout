# 🔍 FounderScout

**Discover overlooked projects and prolific builders on Hacker News and GitHub**

FounderScout contains two powerful tools for finding hidden gems in the developer community:
- **Builder Scout** - Discovers prolific builders working on overlooked projects
- **Project Scout** - Finds passionate projects with low engagement (overlooked gems)

---

## 🔨 Builder Scout

Identifies talented developers who are actively building but haven't gained recognition yet. It finds builders through overlooked projects, then analyzes their overall GitHub activity.

### Usage

```bash
python3 builder_scout.py [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 7 | Days to look back for overlooked projects |
| `--top` | 20 | Number of top builders to show |
| `--no-llm` | False | Skip AI enrichment for faster results |
| `--verbose` | False | Show detailed information |
| `--output` | card | Output format: `card`, `compact`, or `json` |

### Example Output

```
🔨 Builder Scout - Overlooked Prolific Builders
Analyzing projects from last 7 days | GitHub activity from recent events (up to 90 days)

1. Score: 75.0  [AI-ASSISTED]
   Sarah Martinez (@smartz_dev)
   Full-stack developer | Building tools for developers | Open source enthusiast
   Recent Activity: 📦 8 repos active (30d), 23 total active (90d), 💾 67 push events
   🤖 AI Usage: 12 commits with AI assistance detected
   Discovered via: HN post (2024-01-15): Show HN: QuantumCSS – A CSS-in-Rust compiler
   Profile: https://github.com/smartz_dev
```

### What It Analyzes

- **Prolificness** - Commits, new projects, pull requests
- **Overlooked factor** - High activity but low stars/forks
- **AI adoption** - Detects AI-assisted development in commit messages
- **Consistency** - Regular contribution patterns
- **Time context** - Shows activity over 30-day and 90-day windows

### AI Detection

Builder Scout can identify developers using AI tools by analyzing commit messages for patterns like:
- "Generated with Claude"
- "Co-authored-by: Copilot"
- AI-related emojis (🤖)
- Other AI tool signatures

---

## 🎯 Project Scout

Surfaces interesting projects that haven't gotten the attention they deserve by analyzing the balance between **passion** (effort/substance) and **engagement** (stars/points).

### Usage

```bash
python3 project_scout.py [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 7 | Number of days to look back |
| `--top` | 20 | Number of top projects to show |
| `--no-llm` | False | Skip AI enrichment for faster results |
| `--verbose` | False | Show detailed score breakdowns |
| `--output` | card | Output format: `card`, `compact`, or `json` |

### Example Output

```
🔍 ProjectScout - Overlooked Projects & Builders
Last 7 days | Top 20 of 156 found

1. Score: 0.82  [HN]
   Show HN: QuantumCSS – A CSS-in-Rust compiler for type-safe styles by alexchen
   💡 A Rust-based CSS compiler that brings type safety and compile-time validation to stylesheets
   Tags: rust, css, web-development, compiler
   15pts, 4cmt | https://news.ycombinator.com/item?id=41234567
```

### What It Finds

- **High passion, low engagement** - Long detailed posts with few upvotes
- **Show HN posts** - Builders sharing their projects that haven't gone viral
- **Substantial GitHub repos** - Projects with significant code but few stars
- **Recent activity** - Focuses on projects from the specified time period

---

## 🔨 Builder Scout

Identifies talented developers who are actively building but haven't gained recognition yet. It finds builders through overlooked projects, then analyzes their overall GitHub activity.

### Usage

```bash
python3 builder_scout.py [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 7 | Days to look back for overlooked projects |
| `--top` | 20 | Number of top builders to show |
| `--no-llm` | False | Skip AI enrichment for faster results |
| `--verbose` | False | Show detailed information |
| `--output` | card | Output format: `card`, `compact`, or `json` |

### Example Output

```
🔨 Builder Scout - Overlooked Prolific Builders
Analyzing projects from last 7 days | GitHub activity from recent events (up to 90 days)

1. Score: 75.0  [AI-ASSISTED]
   Sarah Martinez (@smartz_dev)
   Full-stack developer | Building tools for developers | Open source enthusiast
   Recent Activity: 📦 8 repos active (30d), 23 total active (90d), 💾 67 push events
   🤖 AI Usage: 12 commits with AI assistance detected
   Discovered via: HN post (2024-01-15): Show HN: QuantumCSS – A CSS-in-Rust compiler
   Profile: https://github.com/smartz_dev
```

### What It Analyzes

- **Prolificness** - Commits, new projects, pull requests
- **Overlooked factor** - High activity but low stars/forks
- **AI adoption** - Detects AI-assisted development in commit messages
- **Consistency** - Regular contribution patterns
- **Time context** - Shows activity over 30-day and 90-day windows

### AI Detection

Builder Scout can identify developers using AI tools by analyzing commit messages for patterns like:
- "Generated with Claude"
- "Co-authored-by: Copilot"
- AI-related emojis (🤖)
- Other AI tool signatures

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required for LLM enrichment (optional)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-5 if available

# Required for GitHub API (highly recommended)
GITHUB_TOKEN=your_github_token_here
```

> **Note**: Without a GitHub token, API rate limits are very restrictive (60 requests/hour). With a token, you get 5,000 requests/hour.

---

## 📊 Scoring System

### Project Scout Scoring

Balances **passion** (effort/substance) vs **engagement** (popularity):
- High passion + Low engagement = **High score** (overlooked gem) ✨
- Low passion + High engagement = **Low score** (viral but shallow) 
- Uses advanced mathematical models (BM25-inspired saturation, geometric means)

### Builder Scout Scoring

Combines multiple factors to identify prolific yet overlooked builders:
- **Prolificness** (0-30 pts): Push events, new projects, contributions
- **Overlooked** (0-30 pts): Activity vs recognition ratio
- **AI adoption** (0-20 pts): Use of AI development tools
- **Consistency** (0-10 pts): Regular contribution patterns
- **Project quality** (0-10 pts): Quality of discovered projects

---

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Find overlooked projects**:
   ```bash
   python3 project_scout.py --days 7 --top 10
   ```

3. **Discover hidden builders**:
   ```bash
   python3 builder_scout.py --days 7 --top 10
   ```

4. **Export to JSON**:
   ```bash
   python3 project_scout.py --output json > projects.json
   python3 builder_scout.py --output json > builders.json
   ```

---

## 💡 Philosophy

Both tools share a common philosophy: **Surface the overlooked gems**. In a world dominated by viral content and popularity metrics, these scripts help you find:

- Projects where someone poured their heart in but didn't get noticed
- Builders who ship consistently but fly under the radar
- Fresh ideas that haven't hit the mainstream yet
- Developers embracing new tools (like AI) to build faster

The goal is to discover **substance over hype** and **effort over virality**.