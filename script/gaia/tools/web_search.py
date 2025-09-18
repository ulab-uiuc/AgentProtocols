import asyncio
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from .utils.config import config
from .utils.logger import logger

from .base import BaseTool, ToolResult
from .exceptions import ToolError
from .search import (
    BaiduSearchEngine,
    BingSearchEngine,
    # DuckDuckGoSearchEngine,  # Disabled due to timeout issues
    GoogleSearchEngine,
    WebSearchEngine,
)
from .search.base import SearchItem

class SearchResult(BaseModel):
    """Represents a single search result returned by a search engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="Position in search results")
    url: str = Field(description="URL of the search result")
    title: str = Field(default="", description="Title of the search result")
    description: str = Field(
        default="", description="Description or snippet of the search result"
    )
    source: str = Field(description="The search engine that provided this result")
    raw_content: Optional[str] = Field(
        default=None, description="Raw content from the search result page if available"
    )

    def __str__(self) -> str:
        """String representation of a search result."""
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    """Metadata about the search operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="Total number of results found")
    language: str = Field(description="Language code used for the search")
    country: str = Field(description="Country code used for the search")


class SearchResponse(ToolResult):
    """Structured response from the web search tool, inheriting ToolResult."""

    query: str = Field(description="The search query that was executed")
    results: List[SearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    metadata: Optional[SearchMetadata] = Field(
        default=None, description="Metadata about the search"
    )

    @model_validator(mode="after")
    def populate_output(self) -> "SearchResponse":
        """Populate output or error fields based on search results."""
        if self.error:
            return self

        result_text = [f"Search results for '{self.query}':"]

        for i, result in enumerate(self.results, 1):
            # Add title with position number
            title = result.title.strip() or "No title"
            result_text.append(f"\n{i}. {title}")

            # Add URL with proper indentation
            result_text.append(f"   URL: {result.url}")

            # Add description if available
            if result.description.strip():
                result_text.append(f"   Description: {result.description}")

            # Add content preview if available
            if result.raw_content:
                content_preview = result.raw_content[:1000].replace("\n", " ").strip()
                if len(result.raw_content) > 1000:
                    content_preview += "..."
                result_text.append(f"   Content: {content_preview}")

        # Add metadata at the bottom if available
        if self.metadata:
            result_text.extend(
                [
                    f"\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )

        self.output = "\n".join(result_text)
        return self


class WebContentFetcher:
    """Utility class for fetching web content."""

    @staticmethod
    async def fetch_content(url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch and extract the main content from a webpage or PDF.

        Args:
            url: The URL to fetch content from
            timeout: Request timeout in seconds

        Returns:
            Extracted text content or None if fetching fails
        """
        headers = {
            "WebSearch": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            # Use asyncio to run requests in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=timeout)
            )

            if response.status_code != 200:
                logger.warning(
                    f"Failed to fetch content from {url}: HTTP {response.status_code}"
                )
                return None

            # Check if content is PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                # Handle PDF content
                try:
                    pdf_text = WebContentFetcher._extract_text_from_pdf(response.content)
                    return f"[PDF Content Extracted]\n{pdf_text[:10000]}" if pdf_text else None
                except Exception as e:
                    logger.warning(f"Failed to extract PDF content from {url}: {e}")
                    return f"[PDF File Detected - Content extraction failed: {str(e)}]"
            else:
                # Handle regular HTML content
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.extract()

                # Get text content
                text = soup.get_text(separator="\n", strip=True)

                # Clean up whitespace and limit size (100KB max)
                text = " ".join(text.split())
                return text[:10000] if text else None

        except Exception as e:
            logger.warning(f"Error fetching content from {url}: {e}")
            return None

    @staticmethod
    def _extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                text_content = []
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                return "\n".join(text_content)
            except ImportError:
                pass
            
            # Try pdfplumber as fallback
            try:
                import pdfplumber
                import io
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    text_content = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                    return "\n".join(text_content)
            except ImportError:
                pass
            
            # Try pymupdf as another fallback
            try:
                import fitz  # PyMuPDF
                pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
                text_content = []
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    text_content.append(page.get_text())
                pdf_document.close()
                return "\n".join(text_content)
            except ImportError:
                pass
            
            # If no PDF library is available, return error message
            return "Error: No PDF parsing library available. Please install PyPDF2, pdfplumber, or PyMuPDF."
            
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"


class WebSearch(BaseTool):
    """Search the web for information using various search engines."""

    name: str = "web_search"
    description: str = """Search the web for real-time information about any topic.
    This tool returns comprehensive search results with relevant information, URLs, titles, and descriptions.
    If the primary search engine fails, it automatically falls back to alternative engines."""
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
            "lang": {
                "type": "string",
                "description": "(optional) Language code for search results (default: en).",
                "default": "en",
            },
            "country": {
                "type": "string",
                "description": "(optional) Country code for search results (default: us).",
                "default": "us",
            },
            "fetch_content": {
                "type": "boolean",
                "description": "(optional) Whether to fetch full content from result pages. Default is false.",
                "default": False,
            },
        },
        "required": ["query"],
    }
    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "bing": BingSearchEngine(),
        # "duckduckgo": DuckDuckGoSearchEngine(),  # Disabled due to timeout issues
    }
    content_fetcher: WebContentFetcher = WebContentFetcher()

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        lang: Optional[str] = None,
        country: Optional[str] = None,
        fetch_content: bool = False,
    ) -> SearchResponse:
        """
        Execute a Web search and return detailed search results.

        Args:
            query: The search query to submit to the search engine
            num_results: The number of search results to return (default: 5)
            lang: Language code for search results (default from config)
            country: Country code for search results (default from config)
            fetch_content: Whether to fetch content from result pages (default: False)

        Returns:
            A structured response containing search results and metadata
        """
        # Get settings from config
        retry_delay = (
            getattr(config.search_config, "retry_delay", 60)
            if config.search_config
            else 60
        )
        max_retries = (
            getattr(config.search_config, "max_retries", 3)
            if config.search_config
            else 3
        )

        # Use config values for lang and country if not specified
        if lang is None:
            lang = (
                getattr(config.search_config, "lang", "en")
                if config.search_config
                else "en"
            )

        if country is None:
            country = (
                getattr(config.search_config, "country", "us")
                if config.search_config
                else "us"
            )

        search_params = {"lang": lang, "country": country}

        # Try searching with retries when all engines fail
        for retry_count in range(max_retries + 1):
            results = await self._try_all_engines(query, num_results, search_params)

            if results:
                # Fetch content if requested
                if fetch_content:
                    results = await self._fetch_content_for_results(results)

                # Return a successful structured response
                return SearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=lang,
                        country=country,
                    ),
                )

            if retry_count < max_retries:
                # All engines failed, wait and retry
                logger.warning(
                    f"All search engines failed. Waiting {retry_delay} seconds before retry {retry_count + 1}/{max_retries}..."
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"All search engines failed after {max_retries} retries. Giving up."
                )

        # Return an error response
        return SearchResponse(
            query=query,
            error="All search engines failed to return results after multiple retries.",
            results=[],
        )

    async def _try_specialized_apis(
        self, query: str, num_results: int
    ) -> Optional[List[SearchResult]]:
        """Try specialized APIs for GAIA-specific information sources."""
        
        # Check if query is asking for arXiv papers
        if any(keyword in query.lower() for keyword in ['arxiv', 'arxiv.org', 'preprint', 'submitted to arxiv']):
            logger.info("🔬 Detected arXiv query, using arXiv API...")
            return await self._search_arxiv_api(query, num_results)
        
        # Check if query is asking for Wikipedia information
        if any(keyword in query.lower() for keyword in ['wikipedia', 'wiki', 'encyclopedia']):
            logger.info("📚 Detected Wikipedia query, using Wikipedia API...")
            return await self._search_wikipedia_api(query, num_results)
        
        # Check if query mentions specific academic domains
        if any(keyword in query.lower() for keyword in ['physics and society', 'nature', 'science', 'journal']):
            logger.info("📄 Detected academic query, trying arXiv API first...")
            arxiv_results = await self._search_arxiv_api(query, num_results)
            if arxiv_results:
                return arxiv_results
        
        return None

    async def _parse_arxiv_query_with_llm(self, query: str) -> dict:
        """
        LLM parser that also identifies and removes low-value tokens ("negatives").
        Output schema (all arrays):
          - time_windows: [{"start":"YYYYMMDD","end":"YYYYMMDD"}, ...]
          - terms_required: ["regulation","governance", ...]         # stems/keywords for ti/abs
          - terms_optional: ["policy","compliance", ...]
          - negatives: ["axis","axes","figure","caption", ...]       # to be removed from terms
          - categories_generic: ["computers-society","physics-society", ...]
          - keywords: ["broad fallback tokens"]
        The caller will ignore categories in API queries (global-first), but use them in scoring.
        """
        import aiohttp
        import json
        import os
        import re
        import datetime as dt
        
        # Default seeds of low-yield words for scholarly abstracts; extend with LLM output
        NEGATIVE_SEEDS = [
            "axis","axes","figure","fig","caption","table","supplementary",
            "image","diagram","plot","graph","chart","dataset-overview",
            "appendix","toc","contents","license","three","both","ends"
        ]

        def rule_fallback(q: str) -> dict:
            """Minimal rule-based extraction in case LLM is not available."""
            ql = q.lower()
            # Time window: try to catch "June 2022" or "Aug 11, 2016" with ±3/±7 buffers
            time_windows = []
            # month window
            m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', ql)
            month_map = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
            if m:
                import calendar
                y = int(m.group(2)); mon = month_map[m.group(1)]
                time_windows.append({"start":f"{y:04d}{mon:02d}01",
                                     "end":f"{y:04d}{mon:02d}{calendar.monthrange(y,mon)[1]:02d}"})
            # exact day + buffers
            m2 = re.search(r'(aug|august)\s+(\d{1,2}),?\s*(\d{4})', ql)
            if m2:
                y = int(m2.group(3)); d = int(m2.group(2)); base = dt.datetime(y, 8, d)
                for days in (0,3,7):
                    lo = (base - dt.timedelta(days=days)).strftime("%Y%m%d")
                    hi = (base + dt.timedelta(days=days)).strftime("%Y%m%d")
                    time_windows.append({"start":lo,"end":hi})

            terms_required, terms_optional = [], []
            if any(k in ql for k in ["regulation","governance","policy","regulatory"]):
                terms_required += ["regulation","governance","policy","regulatory"]
            if "society" in ql:
                terms_required += ["society"]

            return {
                "time_windows": time_windows,
                "terms_required": list(dict.fromkeys(terms_required)),
                "terms_optional": terms_optional,
                "negatives": NEGATIVE_SEEDS[:],
                "categories_generic": [],
                "keywords": []
            }

        # Get API key from config/env
        api_key = None
        try:
            if hasattr(config, 'app_config') and config.app_config:
                llm_cfgs = getattr(config.app_config, 'llm', {})
                if isinstance(llm_cfgs, dict) and llm_cfgs:
                    if 'default' in llm_cfgs and getattr(llm_cfgs['default'], 'api_key', None):
                        api_key = llm_cfgs['default'].api_key
                    else:
                        any_cfg = next(iter(llm_cfgs.values()))
                        api_key = getattr(any_cfg, 'api_key', None)
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            return rule_fallback(query)

        # LLM prompt: ask for negatives explicitly and to exclude them from terms
        system_prompt = (
            "You convert research-style natural-language queries into relaxed, provider-agnostic constraints. "
            "Return ONLY a JSON object with arrays: time_windows, terms_required, terms_optional, negatives, "
            "categories_generic, keywords.\n"
            "- time_windows: array of objects {start:'YYYYMMDD',end:'YYYYMMDD'}; include both precise and relaxed buffers.\n"
            "- terms_required/optional: SHORT stems/keywords for title/abstract search. DO NOT include low-yield tokens.\n"
            "- negatives: tokens to IGNORE in search (e.g., figure, axis, axes, caption, table, appendix, image...). "
            "Add other clearly low-value tokens you see.\n"
            "- categories_generic: broad conceptual areas (e.g., 'computers-society','physics-society','ai','economics'). "
            "These are NOT for query syntax; they will be used only for post-filter scoring.\n"
            "- keywords: optional broad fallback tokens.\n"
            "Do not wrap JSON in markdown fences."
        )
        user_prompt = f"Query: {query}"

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 600
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as sess:
                async with sess.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        return rule_fallback(query)
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if content.startswith("```"):
                        content = content.strip("`").strip()
                        if content.lower().startswith("json"):
                            content = content[4:].strip()
                    parsed = json.loads(content)
        except Exception:
            return rule_fallback(query)

        # Normalize + inject default negatives
        out = {
            "time_windows": parsed.get("time_windows", []) or [],
            "terms_required": parsed.get("terms_required", []) or [],
            "terms_optional": parsed.get("terms_optional", []) or [],
            "negatives": list(dict.fromkeys((parsed.get("negatives", []) or []) + NEGATIVE_SEEDS)),
            "categories_generic": parsed.get("categories_generic", []) or [],
            "keywords": parsed.get("keywords", []) or [],
        }

        # Ensure negatives are NOT present in terms
        def clean_terms(arr, negs):
            negset = {n.lower() for n in negs}
            cleaned = []
            for t in arr:
                t2 = str(t).strip()
                if not t2: 
                    continue
                # If any negative token appears as a standalone word or exact token, skip
                low = t2.lower()
                if any(neg in low.split() for neg in negset):
                    continue
                cleaned.append(t2)
            # dedup
            return list(dict.fromkeys(cleaned))
        
        out["terms_required"] = clean_terms(out["terms_required"], out["negatives"])
        out["terms_optional"] = clean_terms(out["terms_optional"], out["negatives"])

        logger.info(f"LLM parsed query with negatives filtering: {out}")
        return out

    def _parse_arxiv_query_simple(self, query: str) -> dict:
        """Simple fallback parsing when LLM is not available."""
        time_window = {}
        category_candidates = []
        core_terms = []
        
        # Basic date extraction
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        # Extract month year patterns
        date_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', query.lower())
        if date_match:
            month_name = date_match.group(1)
            year = date_match.group(2)
            if month_name in month_map:
                time_window['month_range'] = f"{year}{month_map[month_name]}"
                time_window['flex_days'] = 7
        
        # Basic category detection
        query_lower = query.lower()
        if 'physics and society' in query_lower:
            category_candidates.append({"category": "physics.soc-ph", "confidence": 0.8})
        elif 'ai' in query_lower or 'artificial intelligence' in query_lower:
            category_candidates.append({"category": "cs.AI", "confidence": 0.8})
        elif 'machine learning' in query_lower:
            category_candidates.append({"category": "cs.LG", "confidence": 0.8})
        
        # Extract core terms (avoid low-value terms)
        if 'regulation' in query_lower:
            core_terms.append({"term": "regulation", "synonyms": ["governance", "policy", "regulatory"]})
        if 'society' in query_lower:
            core_terms.append({"term": "society", "synonyms": ["social", "community"]})
        if 'democracy' in query_lower:
            core_terms.append({"term": "democracy", "synonyms": ["democratic"]})
        
        return {
            'time_window': time_window,
            'category_candidates': category_candidates,
            'core_terms': core_terms,
            'confidence': 0.6,  # Lower confidence for simple parsing
            'query_intent': f"Simple parsing of query: {query[:50]}..."
        }

    def _generate_progressive_queries(self, parsed: dict) -> list:
        """Generate progressive search strategies from strict to relaxed."""
        from datetime import datetime, timedelta
        
        strategies = []
        time_window = parsed.get('time_window', {})
        categories = parsed.get('category_candidates', [])
        core_terms = parsed.get('core_terms', [])
        
        # Helper function to create date ranges
        def create_date_filter(exact_date=None, month_range=None, flex_days=7):
            if exact_date:
                # Exact date with flexibility
                exact_date_str = str(exact_date)
                date_obj = datetime.strptime(exact_date_str, '%Y%m%d')
                start_date = (date_obj - timedelta(days=flex_days)).strftime('%Y%m%d')
                end_date = (date_obj + timedelta(days=flex_days)).strftime('%Y%m%d')
                return f"submittedDate:[{start_date} TO {end_date}]"
            elif month_range:
                # Entire month
                month_range_str = str(month_range)
                year = month_range_str[:4]
                month = month_range_str[4:]
                if month == '02':
                    end_day = '29'  # Handle leap years
                elif month in ['04', '06', '09', '11']:
                    end_day = '30'
                else:
                    end_day = '31'
                return f"submittedDate:[{year}{month}01 TO {year}{month}{end_day}]"
            return None
        
        # Strategy 1: Exact date + best category + core terms
        if time_window.get('exact_date') and categories and core_terms:
            date_filter = create_date_filter(exact_date=time_window['exact_date'], flex_days=0)  # Exact
            best_category = max(categories, key=lambda x: x['confidence'])['category']
            term_queries = []
            for term_obj in core_terms:
                term = term_obj['term']
                synonyms = term_obj.get('synonyms', [])
                all_terms = [term] + synonyms
                term_query = ' OR '.join([f"(ti:{t} OR abs:{t})" for t in all_terms])
                term_queries.append(f"({term_query})")
            
            if term_queries:
                strategy = f"({date_filter}) AND (cat:{best_category}) AND ({' OR '.join(term_queries)})"
                strategies.append(("Exact date + category + terms", strategy))
        
        # Strategy 2: Month range + best category + core terms
        if time_window.get('month_range') and categories and core_terms:
            date_filter = create_date_filter(month_range=time_window['month_range'])
            best_category = max(categories, key=lambda x: x['confidence'])['category']
            # Use only the most important term
            main_term = core_terms[0]
            term = main_term['term']
            synonyms = main_term.get('synonyms', [])
            all_terms = [term] + synonyms
            term_query = ' OR '.join([f"(ti:{t} OR abs:{t})" for t in all_terms])
            
            strategy = f"({date_filter}) AND (cat:{best_category}) AND ({term_query})"
            strategies.append(("Month + category + main term", strategy))
        
        # Strategy 3: Month range + best category only
        if time_window.get('month_range') and categories:
            date_filter = create_date_filter(month_range=time_window['month_range'])
            best_category = max(categories, key=lambda x: x['confidence'])['category']
            
            strategy = f"({date_filter}) AND (cat:{best_category})"
            strategies.append(("Month + category only", strategy))
        
        # Strategy 4: Extended date range + category
        if time_window.get('exact_date') and categories:
            date_filter = create_date_filter(exact_date=time_window['exact_date'], flex_days=14)  # ±2 weeks
            best_category = max(categories, key=lambda x: x['confidence'])['category']
            
            strategy = f"({date_filter}) AND (cat:{best_category})"
            strategies.append(("Extended date + category", strategy))
        
        # Strategy 5: Category + main terms only (no date)
        if categories and core_terms:
            best_category = max(categories, key=lambda x: x['confidence'])['category']
            main_term = core_terms[0]
            term = main_term['term']
            synonyms = main_term.get('synonyms', [])
            all_terms = [term] + synonyms
            term_query = ' OR '.join([f"(ti:{t} OR abs:{t})" for t in all_terms])
            
            strategy = f"(cat:{best_category}) AND ({term_query})"
            strategies.append(("Category + terms only", strategy))
        
        return strategies

    def _filter_results_by_relevance(self, results: list, parsed: dict) -> list:
        """Filter results based on core terms and confidence."""
        if not results or not parsed.get('core_terms'):
            return results
        
        core_terms = parsed.get('core_terms', [])
        filtered_results = []
        
        # Extract all relevant terms and synonyms
        all_relevant_terms = set()
        for term_obj in core_terms:
            all_relevant_terms.add(term_obj['term'].lower())
            for synonym in term_obj.get('synonyms', []):
                all_relevant_terms.add(synonym.lower())
        
        # Add common word stems
        term_stems = set()
        for term in all_relevant_terms:
            if 'regulat' in term:
                term_stems.update(['regulat', 'govern', 'policy'])
            if 'societ' in term:
                term_stems.update(['societ', 'social', 'community'])
            if 'democra' in term:
                term_stems.update(['democra', 'democratic'])
        
        all_relevant_terms.update(term_stems)
        
        for result in results:
            # Check title and description for relevance
            title_text = result.title.lower()
            desc_text = result.description.lower()
            combined_text = f"{title_text} {desc_text}"
            
            # Calculate relevance score
            relevance_score = 0
            for term in all_relevant_terms:
                if term in combined_text:
                    relevance_score += 1
                    if term in title_text:  # Title matches are more important
                        relevance_score += 1
            
            # Only include if there's some relevance
            if relevance_score > 0:
                filtered_results.append(result)
        
        # Sort by relevance (could add more sophisticated scoring)
        logger.info(f"Filtered {len(results)} results down to {len(filtered_results)} relevant ones")
        return filtered_results

    async def _search_arxiv_api(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Global-first arXiv search (no category constraint) + post-filter scoring (fixed):
        1) Merge multiple time_windows into ONE widest range (avoid ANDing multiple ranges).
        2) Build API query from the single date range + cleaned content terms (omit cat:*).
        3) Score results by (date hit + term hit + optional category bonus mapped from generic labels)
           minus a small penalty for negative tokens.
        """
        import aiohttp
        import re
        import datetime as dt
        from typing import List, Tuple

        # ---------- helpers ----------
        def _norm_date_str(s: str) -> str:
            # 'YYYY-MM-DD' or 'YYYYMMDD' -> 'YYYYMMDD'
            return s.replace('-', '') if '-' in s else s

        def _within(pub_ymd: str, lo_ymd: str, hi_ymd: str) -> bool:
            return (not lo_ymd or pub_ymd >= lo_ymd) and (not hi_ymd or pub_ymd <= hi_ymd)

        def _parse_meta_from_raw(raw: str) -> dict:
            # Expect raw_content produced by _parse_arxiv_response:
            # [ARXIV_ID]=xxxx\n[PUB]=YYYY-MM-DD\n[PRIMARY]=cs.CY\n[CATS]=cs.CY,cs.AI
            meta = {}
            if not raw:
                return meta
            for line in raw.splitlines():
                if '=' in line:
                    k, v = line.split('=', 1)
                    meta[k.strip('[]')] = v.strip()
            return meta

        def _any_stem_hit(text: str, stems) -> bool:
            t = text.lower()
            return any(st in t for st in stems)

        # Map generic categories (from LLM) to arXiv category codes for a soft bonus in scoring
        GENERIC_TO_ARXIV = {
            # Computers & Society
            "computers-society": ["cs.CY", "cs.SI", "econ.GN"],
            "ai": ["cs.AI", "cs.LG", "cs.CV", "cs.CL"],  # expand as needed
            # Physics & Society
            "physics-society": ["physics.soc-ph"],
            # Economics or related
            "economics": ["econ.GN"],
        }

        def _expand_generic_categories(generic_labels) -> List[str]:
            out = []
            for g in generic_labels:
                out.extend(GENERIC_TO_ARXIV.get(str(g).lower(), []))
            # dedup while preserving order
            seen = set()
            uniq = []
            for c in out:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            return uniq

        def _score_item(sr: SearchResult,
                        merged_window: dict,
                        required_stems,
                        optional_stems,
                        mapped_arxiv_cats,
                        negatives) -> float:
            """
            Score = date_match (0/1) + term_match (0/1 or 0.5 for optional)
                  + category_bonus (0/0.5 if any mapped cat appears)
                  - negative_penalty (0..0.3)
            """
            meta = _parse_meta_from_raw(sr.raw_content or "")
            pub = _norm_date_str(meta.get("PUB", "")[:10])
            title_desc = f"{sr.title} {sr.description}"
            cats_str = meta.get("CATS", "")

            # date score
            date_hit = 0.0
            if merged_window:
                lo = merged_window.get('start', '')
                hi = merged_window.get('end', '')
                if pub and _within(pub, lo, hi):
                    date_hit = 1.0

            # term score
            term_hit = 0.0
            if not required_stems or _any_stem_hit(title_desc, required_stems):
                term_hit = 1.0
            elif optional_stems and _any_stem_hit(title_desc, optional_stems):
                term_hit = 0.5

            # category bonus (soft)
            cat_bonus = 0.0
            if mapped_arxiv_cats and cats_str:
                if any(cat in cats_str for cat in mapped_arxiv_cats):
                    cat_bonus = 0.5

            # negative penalty (very small, only for obvious low-value terms)
            neg_penalty = 0.0
            if negatives:
                # Only penalize obvious diagram/layout terms, not common words like "three"
                diagram_negs = [n for n in negatives if n.lower() in ['figure', 'fig', 'axis', 'axes', 'diagram', 'plot', 'chart']]
                hits = sum(1 for n in diagram_negs if n.lower() in title_desc.lower())
                neg_penalty = 0.1 * min(hits, 3)  # cap at 0.3

            return date_hit + term_hit + cat_bonus - neg_penalty

        # ---------- LLM parse ----------
        parsed = await self._parse_arxiv_query_with_llm(query)
        time_windows = parsed.get("time_windows", []) or []
        terms_required = parsed.get("terms_required", []) or []
        terms_optional = parsed.get("terms_optional", []) or []
        negatives = parsed.get("negatives", []) or []
        categories_generic = parsed.get("categories_generic", []) or []
        keywords = parsed.get("keywords", []) or []

        # 1) Progressive time window strategy (tight to loose)
        # Sort time windows by range size (narrowest first)
        sorted_windows = []
        if time_windows:
            for tw in time_windows:
                start = tw.get('start', '')
                end = tw.get('end', '')
                if start and end:
                    # Calculate range size for sorting
                    range_size = int(end) - int(start) if start.isdigit() and end.isdigit() else 0
                    sorted_windows.append((range_size, tw))
            sorted_windows.sort(key=lambda x: x[0])  # narrowest first

        # 2) Build content terms (cleaned of negatives)
        negset = {n.lower() for n in negatives}
        content_terms = []
        for term in (terms_required + terms_optional):
            t = str(term or "").strip()
            if not t:
                continue
            if t.lower() in negset:
                continue
            content_terms.append(f"(ti:{t} OR abs:{t})")

        # Fallback to keywords if no specific terms
        if not content_terms and keywords:
            kws = [kw for kw in keywords if kw and str(kw).lower() not in negset]
            if kws:
                content_terms = [f"all:{' '.join(str(kw) for kw in kws)}"]

        # 3) Progressive search: try tight time windows first, then relax
        all_results = []
        mapped_arxiv_cats = _expand_generic_categories(categories_generic)
        
        # Strategy 1: Try each time window individually (tight first)
        for range_size, time_window in sorted_windows:
            query_parts = []
            query_parts.append(f"submittedDate:[{time_window['start']} TO {time_window['end']}]")
            
            if content_terms:
                query_parts.append(f"({' OR '.join(content_terms)})")
            
            search_query = ' AND '.join(f"({part})" for part in query_parts)
            logger.info(f"arXiv trying tight window (range={range_size}): {search_query}")
            
            results = await self._execute_arxiv_query(search_query, num_results * 2)
            if results:
                scored_results = self._score_and_filter_results(
                    results, time_window, terms_required, terms_optional, 
                    mapped_arxiv_cats, negatives, min_score=0.8  # Higher threshold for tight windows
                )
                if scored_results:
                    logger.info(f"Tight window strategy found {len(scored_results)} high-quality results")
                    return scored_results[:num_results]
                all_results.extend(results)

        # Strategy 2: Use broader time range if tight windows failed
        if sorted_windows and not all_results:
            # Use widest window
            widest_window = sorted_windows[-1][1] if sorted_windows else {}
            if widest_window:
                query_parts = []
                query_parts.append(f"submittedDate:[{widest_window['start']} TO {widest_window['end']}]")
                
                if content_terms:
                    query_parts.append(f"({' OR '.join(content_terms)})")
                
                search_query = ' AND '.join(f"({part})" for part in query_parts)
                logger.info(f"arXiv trying broad window: {search_query}")
                
                results = await self._execute_arxiv_query(search_query, num_results * 3)
                if results:
                    all_results.extend(results)

        # Strategy 3: Content terms only (no time constraint)
        if not all_results and content_terms:
            search_query = f"({' OR '.join(content_terms)})"
            logger.info(f"arXiv trying content-only: {search_query}")
            
            results = await self._execute_arxiv_query(search_query, num_results * 4)
            if results:
                all_results.extend(results)

        # Strategy 4: Ultimate fallback
        if not all_results:
            clean_query = re.sub(r'\b(site:arxiv\.org|arxiv\.org|arxiv)\b', '', query, flags=re.IGNORECASE)
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()
            search_query = f"all:{clean_query}"
            logger.info(f"arXiv fallback query: {search_query}")
            
            results = await self._execute_arxiv_query(search_query, num_results * 2)
            if results:
                all_results.extend(results)

        # Final scoring and filtering
        if all_results:
            # Use the best time window for scoring
            best_window = sorted_windows[0][1] if sorted_windows else {}
            scored_results = self._score_and_filter_results(
                all_results, best_window, terms_required, terms_optional,
                mapped_arxiv_cats, negatives, min_score=0.5  # Lower threshold for fallback
            )
            
            if scored_results:
                logger.info(f"arXiv final results: {len(all_results)} total -> {len(scored_results)} relevant")
                return scored_results[:num_results]

        logger.warning("All arXiv search strategies failed")
        return []

    async def _execute_arxiv_query(self, search_query: str, max_results: int) -> List[SearchResult]:
        """Execute a single arXiv API query."""
        try:
            import aiohttp
            
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": min(max_results, 100),
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get("http://export.arxiv.org/api/query", params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_arxiv_response(xml_content)
                    else:
                        logger.warning(f"arXiv API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"arXiv query execution failed: {e}")
            return []

    def _score_and_filter_results(self, results: List[SearchResult], time_window: dict, 
                                 required_stems, optional_stems, mapped_arxiv_cats, 
                                 negatives, min_score: float = 0.6) -> List[SearchResult]:
        """Score and filter results with soft/hard category filtering."""
        
        def _norm_date_str(s: str) -> str:
            return s.replace('-', '') if '-' in s else s

        def _within(pub_ymd: str, lo_ymd: str, hi_ymd: str) -> bool:
            return (not lo_ymd or pub_ymd >= lo_ymd) and (not hi_ymd or pub_ymd <= hi_ymd)

        def _parse_meta_from_raw(raw: str) -> dict:
            meta = {}
            if not raw:
                return meta
            for line in raw.splitlines():
                if '=' in line:
                    k, v = line.split('=', 1)
                    meta[k.strip('[]')] = v.strip()
            return meta

        def _any_stem_hit(text: str, stems) -> bool:
            t = text.lower()
            return any(str(st).lower() in t for st in stems)

        scored_results = []
        for sr in results:
            meta = _parse_meta_from_raw(sr.raw_content or "")
            pub = _norm_date_str(meta.get("PUB", "")[:10])
            title_desc = f"{sr.title} {sr.description}"
            cats_str = meta.get("CATS", "")

            # Date score
            date_hit = 0.0
            if time_window:
                lo = time_window.get('start', '')
                hi = time_window.get('end', '')
                if pub and _within(pub, lo, hi):
                    date_hit = 1.0

            # Term score
            term_hit = 0.0
            if not required_stems or _any_stem_hit(title_desc, required_stems):
                term_hit = 1.0
            elif optional_stems and _any_stem_hit(title_desc, optional_stems):
                term_hit = 0.5

            # Category score (strong bonus for exact matches)
            cat_score = 0.0
            if mapped_arxiv_cats and cats_str:
                # Strong bonus for exact category match
                if any(cat in cats_str for cat in mapped_arxiv_cats):
                    cat_score = 1.0  # Strong category match
                    
                    # Hard filter: if we have specific category requirements, 
                    # prioritize results from those categories
                    if "physics.soc-ph" in mapped_arxiv_cats and "physics.soc-ph" in cats_str:
                        cat_score = 2.0  # Very strong for Physics and Society
                    elif "cs.AI" in mapped_arxiv_cats and any(c in cats_str for c in ["cs.AI", "cs.LG"]):
                        cat_score = 1.5  # Strong for AI/ML

            # Negative penalty (small, only for obvious diagram terms)
            neg_penalty = 0.0
            if negatives:
                diagram_negs = [n for n in negatives if str(n).lower() in ['figure', 'fig', 'axis', 'axes', 'diagram', 'plot', 'chart']]
                hits = sum(1 for n in diagram_negs if str(n).lower() in title_desc.lower())
                neg_penalty = 0.1 * min(hits, 2)  # cap at 0.2

            total_score = date_hit + term_hit + cat_score - neg_penalty
            
            if total_score >= min_score:
                scored_results.append((total_score, sr))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [sr for _, sr in scored_results]

    def _parse_arxiv_response(self, xml_content: str) -> List[SearchResult]:
        """Parse arXiv API XML response with enhanced metadata for post-filtering."""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            results = []
            for i, entry in enumerate(root.findall('atom:entry', namespace)):
                title_elem = entry.find('atom:title', namespace)
                summary_elem = entry.find('atom:summary', namespace)
                id_elem = entry.find('atom:id', namespace)
                published_elem = entry.find('atom:published', namespace)
                
                title = title_elem.text.strip() if title_elem is not None else "No title"
                summary = summary_elem.text.strip() if summary_elem is not None else "No summary"
                arxiv_url = id_elem.text.strip() if id_elem is not None else ""
                published = published_elem.text.strip() if published_elem is not None else ""
                
                # Extract arXiv ID and categories
                arxiv_id = ""
                if arxiv_url:
                    id_match = re.search(r'abs/([^/]+)$', arxiv_url)
                    if id_match:
                        arxiv_id = id_match.group(1)
                
                # Extract categories from entry
                categories = []
                primary_category = ""
                for category_elem in entry.findall('arxiv:primary_category', namespace):
                    primary_category = category_elem.get('term', '')
                    if primary_category:
                        categories.append(primary_category)
                
                for category_elem in entry.findall('category', namespace):
                    cat_term = category_elem.get('term', '')
                    if cat_term and cat_term not in categories:
                        categories.append(cat_term)
                
                # Clean up title and summary
                title = re.sub(r'\s+', ' ', title)
                summary = re.sub(r'\s+', ' ', summary)
                
                # Create enhanced raw_content for metadata parsing
                raw_content = f"[ARXIV_ID]={arxiv_id}\n[PUB]={published[:10]}\n[PRIMARY]={primary_category}\n[CATS]={','.join(categories)}"
                
                results.append(SearchResult(
                    position=i + 1,
                    url=arxiv_url,
                    title=title,
                    description=f"Published: {published[:10]} | {summary[:200]}...",
                    source="arxiv_api",
                    raw_content=raw_content
                ))
            
            logger.info(f"arXiv API returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse arXiv response: {e}")
            return []

    async def _search_wikipedia_api(self, query: str, num_results: int) -> List[SearchResult]:
        """Search Wikipedia using official API."""
        try:
            import aiohttp
            
            # Clean query for Wikipedia search
            clean_query = re.sub(r'\b(wikipedia|wiki)\b', '', query, flags=re.IGNORECASE)
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()
            
            # Wikipedia MediaWiki API endpoint
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": clean_query,
                "srlimit": min(num_results, 10)
            }
            
            headers = {
                "User-Agent": "GAIA-Agent/1.0 (https://github.com/gaia-research/agent-network; contact@example.com)"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), headers=headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        search_results = data.get('query', {}).get('search', [])
                        for i, page in enumerate(search_results):
                            title = page.get('title', 'No title')
                            snippet = page.get('snippet', '')
                            # Remove HTML tags from snippet
                            snippet = re.sub(r'<[^>]+>', '', snippet)
                            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                            
                            results.append(SearchResult(
                                position=i + 1,
                                url=page_url,
                                title=title,
                                description=snippet,
                                source="wikipedia_api"
                            ))
                        
                        logger.info(f"Wikipedia API returned {len(results)} results")
                        return results
                    else:
                        logger.warning(f"Wikipedia API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Wikipedia API search failed: {e}")
        
        return []

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """Try specialized APIs first, then fallback to search engines."""
        
        # First try specialized APIs for GAIA information sources
        specialized_results = await self._try_specialized_apis(query, num_results)
        if specialized_results:
            return specialized_results
        
        # Fallback to traditional search engines
        logger.info("🔍 No specialized API matches, using traditional search engines...")
        
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            logger.info(f"🔍 \033[32mAttempting search with {engine_name.capitalize()}...\033[0m")
            try:
                search_items = await self._perform_search_with_engine(
                    engine, query, num_results, search_params
                )
            except Exception as e:
                logger.warning(f"Search with {engine_name} failed: {e}")
                failed_engines.append(f"{engine_name} ({e.__class__.__name__})")
                continue

            if not search_items:
                failed_engines.append(f"{engine_name} (no results)")
                continue

            if failed_engines:
                logger.info(
                    f"Search successful with {engine_name.capitalize()} after trying: {', '.join(failed_engines)}"
                )

            # Transform search items into structured results
            return [
                SearchResult(
                    position=i + 1,
                    url=item.url,
                    title=item.title
                    or f"Result {i+1}",  # Ensure we always have a title
                    description=item.description or "",
                    source=engine_name,
                )
                for i, item in enumerate(search_items)
            ]

        if failed_engines:
            logger.error(f"All search engines failed: {', '.join(failed_engines)}")
        return []

    async def _fetch_content_for_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fetch and add web content to search results."""
        if not results:
            return []

        # Create tasks for each result
        tasks = [self._fetch_single_result_content(result) for result in results]

        # Type annotation to help type checker
        fetched_results = await asyncio.gather(*tasks)

        # Explicit validation of return type
        return [
            (
                result
                if isinstance(result, SearchResult)
                else SearchResult(**result.dict())
            )
            for result in fetched_results
        ]

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        """Fetch content for a single search result."""
        if result.url:
            content = await self.content_fetcher.fetch_content(result.url)
            if content:
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        """Determines the order in which to try search engines."""
        preferred = (
            getattr(config.search_config, "engine", "google").lower()
            if config.search_config
            else "google"
        )
        fallbacks = (
            [engine.lower() for engine in config.search_config.fallback_engines]
            if config.search_config
            and hasattr(config.search_config, "fallback_engines")
            else []
        )

        # Start with preferred engine, then fallbacks, then remaining engines
        engine_order = [preferred] if preferred in self._search_engine else []
        engine_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self._search_engine and fb not in engine_order
            ]
        )
        engine_order.extend([e for e in self._search_engine if e not in engine_order])

        return engine_order

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Execute search with the given engine and parameters."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(
                engine.perform_search(
                    query,
                    num_results=num_results,
                    lang=search_params.get("lang"),
                    country=search_params.get("country"),
                )
            ),
        )


if __name__ == "__main__":
    web_search = WebSearch()
    search_response = asyncio.run(
        web_search.execute(
            query="Python programming", fetch_content=True, num_results=1
        )
    )
    # Print the ToolResult-friendly string (uses ToolResult.__str__)
    print(search_response)
