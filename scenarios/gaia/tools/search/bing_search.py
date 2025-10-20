from typing import List, Optional, Tuple
import time
import random

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode

from ..utils.logger import logger
from .base import SearchItem, WebSearchEngine


ABSTRACT_MAX_LENGTH = 300

# Updated user-agent list using more modern browser versions
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0"
}

BING_HOST_URL = "https://www.bing.com"


class BingSearchEngine(WebSearchEngine):
    session: Optional[requests.Session] = None
    max_retries: int = 3
    retry_delay: float = 2.0

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, **data):
        """Initialize the BingSearch tool with a requests session and enhanced anti-detection."""
        super().__init__(**data)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.timeout = 15

    def _get_random_user_agent(self) -> str:
        """Get a random user agent to avoid detection."""
        return random.choice(USER_AGENTS)

    def _build_search_url(self, query: str, page: int = 1, count: int = 10) -> str:
        """Build search URL with proper parameters."""
        params = {
            'q': query,
            'count': min(count, 50),  # Bing maximum is 50
            'first': (page - 1) * count + 1 if page > 1 else 1,
            'FORM': 'QBRE',  # Source identifier
        }
        return f"https://www.bing.com/search?{urlencode(params)}"

    def _is_captcha_page(self, soup: BeautifulSoup) -> bool:
        """Check if the page contains a CAPTCHA challenge."""
        captcha_indicators = [
            'challenge',
            'captcha', 
            'robot',
            'solve the challenge',
            'one last step',
            'verify you are human'
        ]
        
        page_text = soup.get_text().lower()
        return any(indicator in page_text for indicator in captcha_indicators)

    def _search_sync(self, query: str, num_results: int = 10) -> List[SearchItem]:
        """
        Synchronous Bing search implementation with enhanced anti-detection.

        Args:
            query (str): The search query to submit to Bing.
            num_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            List[SearchItem]: A list of search items with title, URL, and description.
        """
        if not query:
            return []

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Bing search attempt {attempt + 1}/{self.max_retries}: '{query}'")
                
                # Set random user agent for each attempt
                self.session.headers['User-Agent'] = self._get_random_user_agent()
                
                # Add random delay to avoid detection
                if attempt > 0:
                    delay = self.retry_delay * attempt + random.uniform(1.0, 3.0)
                    time.sleep(delay)
                else:
                    time.sleep(random.uniform(1.0, 2.0))
                
                list_result = []
                page = 1
                first = 1
                
                while len(list_result) < num_results:
                    next_url = self._build_search_url(query, page=page, count=min(10, num_results - len(list_result)))
                    
                    data, next_url = self._parse_html(
                        next_url, rank_start=len(list_result), first=first
                    )
                    
                    if data:
                        list_result.extend(data)
                    
                    if not next_url:
                        break
                        
                    page += 1
                    first += 10

                if list_result:
                    logger.info(f"Bing search successful, returned {len(list_result)} results")
                    return list_result[:num_results]
                else:
                    logger.warning("Bing search returned 0 results")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Bing search attempt {attempt + 1} failed: {e}")
                
                # If we encounter CAPTCHA, increase delay significantly
                if 'captcha' in str(e).lower() or 'challenge' in str(e).lower():
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (attempt + 1) + random.uniform(5.0, 10.0)
                        logger.info(f"CAPTCHA detected, waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                
        logger.error("All Bing search attempts failed")
        if last_exception:
            logger.error(f"Last exception: {last_exception}")
        
        # Return empty list like the original implementation
        return []

    def _parse_html(
        self, url: str, rank_start: int = 0, first: int = 1
    ) -> Tuple[List[SearchItem], str]:
        """
        Parse Bing search result HTML with enhanced error handling and CAPTCHA detection.

        Returns:
            tuple: (List of SearchItem objects, next page URL or None)
        """
        try:
            res = self.session.get(url=url)
            res.encoding = "utf-8"
            root = BeautifulSoup(res.text, "lxml")
            
            # Check for CAPTCHA
            if self._is_captcha_page(root):
                raise Exception("CAPTCHA page detected, need manual verification or proxy")

            list_data = []
            
            # Try multiple result selectors for better compatibility
            ol_results = root.find("ol", id="b_results")
            if not ol_results:
                logger.warning("No results container found")
                return [], None

            result_elements = ol_results.find_all("li", class_="b_algo")
            
            # If no standard results, try alternative selectors
            if not result_elements:
                result_elements = ol_results.find_all("li", class_=lambda x: x and "algo" in x)
            
            for li in result_elements:
                title = ""
                url = ""
                abstract = ""
                
                try:
                    # Enhanced title and URL extraction
                    h2 = li.find("h2")
                    if h2:
                        a_tag = h2.find("a")
                        if a_tag:
                            title = a_tag.get_text(strip=True)
                            url = a_tag.get("href", "").strip()
                    
                    # If no h2, try h3 or direct link
                    if not title:
                        h3 = li.find("h3")
                        if h3:
                            a_tag = h3.find("a")
                            if a_tag:
                                title = a_tag.get_text(strip=True)
                                url = a_tag.get("href", "").strip()

                    # Enhanced description extraction
                    p_elements = li.find_all("p")
                    for p in p_elements:
                        text = p.get_text(strip=True)
                        if text and len(text) > 10:  # Avoid short snippets
                            abstract = text
                            break
                    
                    # Alternative description selectors
                    if not abstract:
                        desc_elem = li.find(class_=lambda x: x and ("caption" in x or "snippet" in x))
                        if desc_elem:
                            abstract = desc_elem.get_text(strip=True)

                    if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                        abstract = abstract[:ABSTRACT_MAX_LENGTH]

                    rank_start += 1

                    # Only add if we have both title and URL
                    if title and url:
                        list_data.append(
                            SearchItem(
                                title=title,
                                url=url,
                                description=abstract or None,
                            )
                        )
                except Exception as e:
                    logger.debug(f"Error parsing result element: {e}")
                    continue

            # Find next page link with better selector
            next_btn = root.find("a", title="Next page") or root.find("a", {"aria-label": "Next page"})
            next_url = None
            
            if next_btn and next_btn.get("href"):
                next_url = BING_HOST_URL + next_btn["href"]
            
            return list_data, next_url
            
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {e}")
            return [], None

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Bing search engine.

        Returns results formatted according to SearchItem model.
        """
        return self._search_sync(query, num_results=num_results)
