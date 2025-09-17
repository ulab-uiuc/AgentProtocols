from typing import List
import random
import time

from duckduckgo_search import DDGS

from .base import SearchItem, WebSearchEngine
from ..utils.logger import logger

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class DuckDuckGoSearchEngine(WebSearchEngine):
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, **data):
        super().__init__(**data)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        DuckDuckGo search engine with robust retry mechanism.

        Returns results formatted according to SearchItem model.
        """
        if not query:
            return []

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"DuckDuckGo search attempt {attempt + 1}/{self.max_retries}: '{query}'")
                
                # Use random user agent for each attempt
                ua = random.choice(USER_AGENTS)
                
                # Add delay between attempts
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                    delay = min(delay, 10.0)  # Cap at 10 seconds
                    time.sleep(delay)
                
                # Try DDG search with headers
                with DDGS(headers={"User-Agent": ua}) as ddgs:
                    raw_results = ddgs.text(query, max_results=num_results)

                results = []
                for i, item in enumerate(raw_results):
                    try:
                        if isinstance(item, str):
                            # If it's just a URL
                            results.append(
                                SearchItem(
                                    title=f"DuckDuckGo Result {i + 1}", url=item, description=None
                                )
                            )
                        elif isinstance(item, dict):
                            # Extract data from the dictionary
                            results.append(
                                SearchItem(
                                    title=item.get("title", f"DuckDuckGo Result {i + 1}"),
                                    url=item.get("href", ""),
                                    description=item.get("body", None),
                                )
                            )
                        else:
                            # Try to extract attributes directly
                            try:
                                results.append(
                                    SearchItem(
                                        title=getattr(item, "title", f"DuckDuckGo Result {i + 1}"),
                                        url=getattr(item, "href", ""),
                                        description=getattr(item, "body", None),
                                    )
                                )
                            except Exception:
                                # Fallback
                                results.append(
                                    SearchItem(
                                        title=f"DuckDuckGo Result {i + 1}",
                                        url=str(item),
                                        description=None,
                                    )
                                )
                    except Exception as parse_error:
                        logger.debug(f"Error parsing DDG result item {i}: {parse_error}")
                        continue

                if results:
                    logger.info(f"DuckDuckGo search successful, returned {len(results)} results")
                    return results[:num_results]
                else:
                    logger.warning("DuckDuckGo search returned 0 results")

            except Exception as e:
                last_exception = e
                logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
                
                # If this is a rate limit or DDG-specific error, add longer delay
                if "ratelimit" in str(e).lower() or "202" in str(e):
                    if attempt < self.max_retries - 1:
                        extra_delay = random.uniform(5.0, 10.0)
                        logger.info(f"Rate limit detected, adding extra delay of {extra_delay:.1f}s")
                        time.sleep(extra_delay)

        logger.error("All DuckDuckGo search attempts failed")
        if last_exception:
            logger.error(f"Last exception: {last_exception}")
        
        return []

