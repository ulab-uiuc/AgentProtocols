"""Simplified web search tool."""
import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseTool, ToolResult


class WebSearch(BaseTool):
    """Search the web for information using a simple mock implementation."""

    name: str = "web_search"
    description: str = """Search the web for real-time information about any topic.
    This is a simplified mock implementation for testing purposes."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """
        Execute a web search (mock implementation).
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            ToolResult with mock search results
        """
        # Mock search results
        results = []
        for i in range(min(num_results, 3)):  # Limit to 3 results
            results.append({
                "title": f"Search Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "description": f"This is a mock description for search result {i+1} about {query}."
            })
        
        # Format output
        output_lines = [f"Search results for '{query}':"]
        for i, result in enumerate(results, 1):
            output_lines.append(f"\n{i}. {result['title']}")
            output_lines.append(f"   URL: {result['url']}")
            output_lines.append(f"   Description: {result['description']}")
        
        return ToolResult(
            output="\n".join(output_lines)
        )
