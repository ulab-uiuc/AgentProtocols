import asyncio
import base64
import json
import io
import re
from typing import Generic, Optional, TypeVar

import sys

# Check Python version requirement for browser_use
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"browser_use requires Python 3.11 or higher. "
        f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}. "
        f"Please upgrade to Python 3.11+ to use BrowserUseTool."
    )

try:
    from browser_use import Browser as BrowserUseBrowser
    from browser_use import BrowserConfig
    from browser_use.browser.context import BrowserContext, BrowserContextConfig
    from browser_use.dom.service import DomService
except ImportError as e:
    raise ImportError(
        f"[BROWSER_USE]: Failed to import browser_use - {e}. "
        f"Please install browser_use: pip install browser-use"
    )
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from .utils.config import config
from .utils.llm import LLM
from .base import BaseTool, ToolResult
from .web_search import WebSearch


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                    "parse_pdf",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
            "parse_pdf": ["url"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    def _has_display(self) -> bool:
        """Check if display is available for browser operations."""
        import os
        # Check if DISPLAY is set (for X11)
        if os.environ.get('DISPLAY'):
            return True
        # Check if WAYLAND_DISPLAY is set (for Wayland)
        if os.environ.get('WAYLAND_DISPLAY'):
            return True
        # Check if we're in a desktop environment
        if os.environ.get('XDG_SESSION_TYPE') in ['x11', 'wayland']:
            return True
        # For Windows/Mac, assume display is available
        import platform
        if platform.system() in ['Windows', 'Darwin']:
            return True
            return False

    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            # Try PyPDF2 first
            try:
                import PyPDF2
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

    def _is_pdf_content(self, content: str) -> bool:
        """Check if content appears to be PDF binary data."""
        # Check for PDF header or binary indicators
        return (
            content.startswith('%PDF') or 
            'stream' in content[:1000] or
            'obj' in content[:1000] or
            len([c for c in content[:1000] if ord(c) < 32 and c not in ['\n', '\r', '\t']]) > 50
        )

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            # Always run headless in server environments
            browser_config_kwargs = {"headless": True, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value
                            
            # Force headless mode in server environments
            browser_config_kwargs["headless"] = True

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def _fetch_url_content(self, url: str) -> str:
        """Fetch content from URL using requests/httpx for headless environments."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Check if content is PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type or url.lower().endswith('.pdf'):
                    # Handle PDF content
                    pdf_text = self._extract_text_from_pdf(response.content)
                    return f"[PDF Content Extracted]\n{pdf_text}"
                else:
                    # Handle regular text content
                    return response.text
                    
        except ImportError:
            # Fallback to requests if httpx not available
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if content is PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                # Handle PDF content
                pdf_text = self._extract_text_from_pdf(response.content)
                return f"[PDF Content Extracted]\n{pdf_text}"
            else:
                # Handle regular text content
                return response.text
                
        except Exception as e:
            raise Exception(f"Failed to fetch content from {url}: {str(e)}")

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:

                # Check if we have display for full browser functionality
                has_display = self._has_display()
                
                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 4000
                )

                # In headless server environments, use simplified web operations
                if not has_display:
                    return await self._execute_headless_action(
                        action, url, query, goal, max_content_length, seconds
                    )

                # Full browser functionality for environments with display
                context = await self._ensure_browser_initialized()

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    
                    search_engine = None
                    if getattr(config, "search_config", None):
                        search_engine = getattr(config.search_config, "engine", "google").lower()
                    # Green color printing
                    GREEN = '\033[92m'
                    RESET = '\033[0m'
                    if search_engine:
                        print(f"{GREEN}🔍 Using search engine: {search_engine.capitalize()}{RESET}")

                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    content = markdownify.markdownify(await page.content())

                    prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [{"role": "system", "content": prompt}]

                    # Define extraction function schema
                    extraction_function = {
                        "type": "function",
                        "function": {
                            "name": "extract_content",
                            "description": "Extract specific information from a webpage based on a goal",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "extracted_content": {
                                        "type": "object",
                                        "description": "The content extracted from the page according to the goal",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content extracted from the page",
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "description": "Additional metadata about the extracted content",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Source of the extracted content",
                                                    }
                                                },
                                            },
                                        },
                                    }
                                },
                                "required": ["extracted_content"],
                            },
                        },
                    }

                    # Use LLM to extract content with required function calling
                    response = await self.llm.ask_tool(
                        messages,
                        tools=[extraction_function],
                        tool_choice="required",
                    )

                    if response and response.tool_calls:
                        args = json.loads(response.tool_calls[0].function.arguments)
                        extracted_content = args.get("extracted_content", {})
                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )

                    return ToolResult(output="No content was extracted from the page.")

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                elif action == "parse_pdf":
                    if not url:
                        return ToolResult(error="URL is required for 'parse_pdf' action")
                    
                    try:
                        content = await self._fetch_url_content(url)
                        if content.startswith("[PDF Content Extracted]"):
                            return ToolResult(output=content)
                        else:
                            # Try to parse as PDF if it wasn't automatically detected
                            if self._is_pdf_content(content):
                                pdf_text = self._extract_text_from_pdf(content.encode('latin-1'))
                                return ToolResult(output=f"[PDF Content Extracted]\n{pdf_text}")
                            else:
                                return ToolResult(error=f"URL does not appear to contain PDF content: {url}")
                    except Exception as e:
                        return ToolResult(error=f"Failed to parse PDF from {url}: {str(e)}")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def _execute_headless_action(
        self, 
        action: str, 
        url: Optional[str], 
        query: Optional[str], 
        goal: Optional[str], 
        max_content_length: int,
        seconds: Optional[int] = None
    ) -> ToolResult:
        """Execute actions in headless server environment using only web search and content fetching."""
        
        if action == "web_search":
            if not query:
                return ToolResult(error="Query is required for 'web_search' action")
            
            # Execute web search and fetch content from first result
            search_response = await self.web_search_tool.execute(
                query=query, fetch_content=True, num_results=3
            )
            
            if not search_response.results:
                return ToolResult(error="No search results found")
            
            # Get the first result and fetch its content
            first_result = search_response.results[0]
            try:
                content = await self._fetch_url_content(first_result.url)
                
                # Check if it's PDF content that wasn't properly parsed
                if self._is_pdf_content(content) and not content.startswith("[PDF Content Extracted]"):
                    # Try to parse as PDF if it looks like binary PDF data
                    try:
                        pdf_text = self._extract_text_from_pdf(content.encode('latin-1'))
                        content = f"[PDF Content Extracted]\n{pdf_text}"
                    except:
                        pass  # Keep original content if PDF parsing fails
                
                # Convert HTML to markdown for better readability (skip for PDF content)
                if not content.startswith("[PDF Content Extracted]"):
                    import markdownify
                    markdown_content = markdownify.markdownify(content)
                else:
                    markdown_content = content
                
                # Truncate content if too long
                if len(markdown_content) > max_content_length:
                    markdown_content = markdown_content[:max_content_length] + "..."
                
                # Combine search result with fetched content
                result_output = f"Search Results:\n{search_response.output}\n\nFetched Content from {first_result.url}:\n{markdown_content}"
                
                return ToolResult(output=result_output)
                
            except Exception as e:
                # Return search results even if content fetching fails
                return ToolResult(
                    output=f"{search_response.output}\n\nNote: Could not fetch content from {first_result.url}: {str(e)}"
                )
        
        elif action == "go_to_url":
            if not url:
                return ToolResult(error="URL is required for 'go_to_url' action")
            
            try:
                content = await self._fetch_url_content(url)
                
                # Check if it's PDF content that wasn't properly parsed
                if self._is_pdf_content(content) and not content.startswith("[PDF Content Extracted]"):
                    try:
                        pdf_text = self._extract_text_from_pdf(content.encode('latin-1'))
                        content = f"[PDF Content Extracted]\n{pdf_text}"
                    except:
                        pass  # Keep original content if PDF parsing fails
                
                # Convert HTML to markdown (skip for PDF content)
                if not content.startswith("[PDF Content Extracted]"):
                    import markdownify
                    markdown_content = markdownify.markdownify(content)
                else:
                    markdown_content = content
                
                # Truncate content if too long
                if len(markdown_content) > max_content_length:
                    markdown_content = markdown_content[:max_content_length] + "..."
                
                return ToolResult(output=f"Fetched content from {url}:\n{markdown_content}")
                
            except Exception as e:
                return ToolResult(error=f"Failed to fetch content from {url}: {str(e)}")
        
        elif action == "extract_content":
            if not goal:
                return ToolResult(error="Goal is required for 'extract_content' action")
            
            # For headless mode, we need a URL to extract content from
            # This should be preceded by a go_to_url or web_search action
            return ToolResult(
                error="extract_content in headless mode requires first navigating to a URL with go_to_url or web_search"
            )
        
        elif action == "wait":
            seconds_to_wait = seconds if seconds is not None else 3
            await asyncio.sleep(seconds_to_wait)
            return ToolResult(output=f"Waited for {seconds_to_wait} seconds")
        
        elif action == "parse_pdf":
            if not url:
                return ToolResult(error="URL is required for 'parse_pdf' action")
            
            try:
                content = await self._fetch_url_content(url)
                if content.startswith("[PDF Content Extracted]"):
                    return ToolResult(output=content)
                else:
                    # Try to parse as PDF if it wasn't automatically detected
                    if self._is_pdf_content(content):
                        pdf_text = self._extract_text_from_pdf(content.encode('latin-1'))
                        return ToolResult(output=f"[PDF Content Extracted]\n{pdf_text}")
                    else:
                        return ToolResult(error=f"URL does not appear to contain PDF content: {url}")
            except Exception as e:
                return ToolResult(error=f"Failed to parse PDF from {url}: {str(e)}")
        
        else:
            # For other browser actions that require interaction, return appropriate message
            interactive_actions = [
                "click_element", "input_text", "scroll_down", "scroll_up", 
                "scroll_to_text", "send_keys", "get_dropdown_options", 
                "select_dropdown_option", "go_back", "switch_tab", 
                "open_tab", "close_tab"
            ]
            
            if action in interactive_actions:
                return ToolResult(
                    error=f"Action '{action}' requires browser interaction and is not available in headless server environment. Use 'web_search' or 'go_to_url' for content access."
                )
            else:
                return ToolResult(error=f"Unknown action: {action}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
