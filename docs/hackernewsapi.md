# Hacker News API Documentation

This document contains comprehensive information about the Hacker News APIs available for building a command-line application.

## Overview

There are two main APIs available for accessing Hacker News data:

1. **Official Hacker News API** (Firebase-based)
2. **Algolia Search API** (Enhanced search capabilities)

## 1. Official Hacker News API

### Base URL
```
https://hacker-news.firebaseio.com/v0/
```

### Authentication
- **No authentication required**
- No API keys needed
- Direct public access

### Rate Limiting
- Currently no rate limit imposed

### Available Endpoints

#### Item Endpoints
- **Get Item**: `/item/<id>.json`
  - Returns story, comment, job, poll, or pollopt
  - Example: `https://hacker-news.firebaseio.com/v0/item/8863.json`

#### User Endpoints
- **Get User**: `/user/<username>.json`
  - Only users with public activity are accessible
  - Example: `https://hacker-news.firebaseio.com/v0/user/jl.json`

#### Live Data Endpoints
- **Max Item**: `/maxitem.json` - Current largest item ID
- **Top Stories**: `/topstories.json` - Up to 500 top stories
- **New Stories**: `/newstories.json` - Up to 500 newest stories
- **Best Stories**: `/beststories.json` - Up to 500 best stories
- **Ask Stories**: `/askstories.json` - Up to 200 Ask HN stories
- **Show Stories**: `/showstories.json` - Up to 200 Show HN stories
- **Job Stories**: `/jobstories.json` - Up to 200 job postings

### Data Structures

#### Story
```json
{
  "by": "dhouston",
  "descendants": 71,
  "id": 8863,
  "kids": [8952, 9224, 8917, ...],
  "score": 111,
  "time": 1175714200,
  "title": "My YC app: Dropbox - Throw away your USB drive",
  "type": "story",
  "url": "http://www.getdropbox.com/u/2/screencast.html"
}
```

#### Comment
```json
{
  "by": "norvig",
  "id": 2921983,
  "kids": [2922097, 2922429, ...],
  "parent": 2921506,
  "text": "Aw shucks, guys ... you make me blush with your compliments.",
  "time": 1314211127,
  "type": "comment"
}
```

#### Ask
```json
{
  "by": "tel",
  "descendants": 16,
  "id": 121003,
  "kids": [121016, 121109, ...],
  "score": 25,
  "text": "<i>or</i> HN: the Next Iteration<p>I get the impression...",
  "time": 1203647620,
  "title": "Ask HN: The Arc Effect",
  "type": "story"
}
```

#### Job
```json
{
  "by": "justin",
  "id": 192327,
  "score": 6,
  "text": "Justin.tv is the biggest...",
  "time": 1210981217,
  "title": "Justin.tv is looking for a Lead Flash Engineer!",
  "type": "job",
  "url": ""
}
```

#### Poll
```json
{
  "by": "pg",
  "descendants": 54,
  "id": 126809,
  "kids": [126822, 126823, ...],
  "parts": [126810, 126811, ...],
  "score": 46,
  "text": "",
  "time": 1204403652,
  "title": "Poll: What would happen if News.YC had explicit support for polls?",
  "type": "poll"
}
```

#### User
```json
{
  "about": "This is a test",
  "created": 1173923446,
  "id": "jl",
  "karma": 2937,
  "submitted": [8265435, 8168423, ...]
}
```

### Field Descriptions
- **id**: Unique item ID
- **deleted**: `true` if deleted
- **type**: "job", "story", "comment", "poll", or "pollopt"
- **by**: Username of submitter
- **time**: Unix timestamp
- **text**: Comment, story, or poll text (HTML)
- **dead**: `true` if dead
- **parent**: Parent comment ID
- **poll**: Poll associated with pollopt
- **kids**: IDs of child comments
- **url**: URL of story
- **score**: Story points or poll option votes
- **title**: Story, poll, or job title
- **parts**: Poll option IDs
- **descendants**: Total comment count

## 2. Algolia Search API

### Base URL
```
https://hn.algolia.com/api/v1/
```

### Authentication
- **No authentication required**
- Public API with CORS support

### Main Endpoints

1. **Search**: `/search`
   - General search endpoint
   - Example: `https://hn.algolia.com/api/v1/search?query=javascript`

2. **Search by Date**: `/search_by_date`
   - Search ordered by date
   - Example: `https://hn.algolia.com/api/v1/search_by_date?query=javascript`

3. **Items**: `/items/{id}`
   - Get specific item by ID
   - Example: `https://hn.algolia.com/api/v1/items/8863`

### Search Parameters

#### Basic Parameters
- **query**: Search query string
- **tags**: Filter by tags (can be combined)
  - `story` - Stories only
  - `comment` - Comments only
  - `poll` - Polls only
  - `pollopt` - Poll options only
  - `show_hn` - Show HN posts
  - `ask_hn` - Ask HN posts
  - `front_page` - Front page stories
  - `author_{username}` - Posts by specific author
  - `story_{id}` - Comments on specific story

#### Pagination
- **page**: Page number (0-based)
- **hitsPerPage**: Results per page (default: 20, max: 1000)

#### Filtering
- **numericFilters**: Numeric filters
  - `created_at_i>{timestamp}` - Items after timestamp
  - `created_at_i<{timestamp}` - Items before timestamp
  - `points>{n}` - Items with more than n points
  - `num_comments>{n}` - Items with more than n comments

#### Advanced Parameters
- **restrictSearchableAttributes**: Limit search to specific fields
  - `title` - Search in titles only
  - `url` - Search in URLs only
  - `author` - Search in author names only

### Example Queries

1. **Search for JavaScript stories**:
   ```
   GET https://hn.algolia.com/api/v1/search?query=javascript&tags=story
   ```

2. **Get recent comments by user**:
   ```
   GET https://hn.algolia.com/api/v1/search_by_date?tags=comment,author_pg
   ```

3. **Search for stories with specific URL**:
   ```
   GET https://hn.algolia.com/api/v1/search?tags=story&restrictSearchableAttributes=url&query=github.com
   ```

4. **Get popular stories from last week**:
   ```
   GET https://hn.algolia.com/api/v1/search?tags=story&numericFilters=created_at_i>1640995200,points>100
   ```

5. **Get all comments for a story**:
   ```
   GET https://hn.algolia.com/api/v1/search_by_date?tags=comment,story_8863&hitsPerPage=500
   ```

### Response Format

```json
{
  "hits": [
    {
      "created_at": "2007-02-19T19:15:00.000Z",
      "title": "My YC app: Dropbox",
      "url": "http://www.getdropbox.com/u/2/screencast.html",
      "author": "dhouston",
      "points": 111,
      "story_text": null,
      "comment_text": null,
      "num_comments": 71,
      "story_id": null,
      "story_title": null,
      "story_url": null,
      "parent_id": null,
      "created_at_i": 1171913700,
      "objectID": "8863"
    }
  ],
  "nbHits": 46434,
  "page": 0,
  "nbPages": 50,
  "hitsPerPage": 20,
  "processingTimeMS": 3,
  "exhaustiveNbHits": false,
  "query": "dropbox",
  "params": "query=dropbox"
}
```

## Implementation Recommendations

### For Your CLI Application

1. **Use Both APIs**:
   - Use Algolia API for searching (better search capabilities)
   - Use Firebase API for fetching full item details
   - Use Firebase API for real-time feeds (top, new, best stories)

2. **Search Strategy**:
   - Start with Algolia API for search functionality
   - Support various filters (author, date range, points, comments)
   - Implement pagination for large result sets

3. **No Authentication Needed**:
   - Neither API requires authentication
   - Your HN account is not needed for API access
   - Both APIs are read-only (no posting capabilities)

4. **Performance Considerations**:
   - Algolia API is faster for search operations
   - Firebase API requires recursive calls for comment threads
   - Cache results when appropriate
   - Respect the 1000 hit limit on Algolia searches

5. **Error Handling**:
   - Handle network timeouts
   - Handle malformed JSON responses
   - Handle deleted/dead items gracefully
   - Handle rate limiting (though currently none exists)

## Example CLI Commands

Your CLI application could support commands like:

```bash
# Search for stories about Python
hackerfind search --query "python" --type story

# Get top stories
hackerfind top --limit 10

# Search by author
hackerfind search --author "pg"

# Search with filters
hackerfind search --query "AI" --min-points 100 --date-from "2024-01-01"

# Get comments for a story
hackerfind comments --story-id 8863

# Get user profile
hackerfind user --username "dhouston"
```

## Support and Resources

- **Firebase API Issues**: api@ycombinator.com
- **Official Firebase API Repo**: https://github.com/HackerNews/API
- **Algolia HN Search**: https://hn.algolia.com/
- **Community**: https://news.ycombinator.com/

## Notes

- The APIs are read-only; posting/voting requires web interface
- Data is near real-time but may have slight delays
- Consider implementing local caching for better performance
- Be mindful of the recursive nature when fetching comment threads