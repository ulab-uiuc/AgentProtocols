from typing import List

from googlesearch import search

from .base import SearchItem, WebSearchEngine


class GoogleSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        raw_results = search(query, num_results=num_results, advanced=True)

        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # If it's just a URL, wrap into SearchItem
                results.append(
                    SearchItem(title=f"Google Result {i+1}", url=item, description="")
                )
            else:
                results.append(
                    SearchItem(
                        title=getattr(item, "title", f"Google Result {i+1}"),
                        url=getattr(item, "url", str(item)),
                        description=getattr(item, "description", ""),
                    )
                )

        return results
