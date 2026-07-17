import wikipedia
from fastmcp import FastMCP
from wikipedia.exceptions import DisambiguationError, PageError, WikipediaException
from src.logger import logging
import json

mcp = FastMCP("wiki-search")

@mcp.tool()
async def wiki_search(query: str, sentences: int = 20, lang: str = "en") -> str:
    """
    Search Wikipedia for concise encyclopedic background, historical facts, definitions, and biographies.
    Args:
        query: Search term or phrase.
        sentences: Number of sentences to return from the summary (default 5).
        lang: Wikipedia language edition code (default "en").
    """
    wikipedia.set_lang(lang)

    try:
        search_hits = wikipedia.search(query)
        if not search_hits:
            logging.warning(f"wiki_search: no results for query='{query}'")
            return json.dumps({
                "wiki_search_error": f"No Wikipedia page found for '{query}'"
            })

        target_title = search_hits[0]

        page = wikipedia.page(target_title, auto_suggest=False)
        summary = wikipedia.summary(target_title, sentences=sentences, auto_suggest=False)

        logging.info(f"wiki_search: resolved '{query}' -> '{target_title}'")
        return json.dumps({
            "title": page.title,
            "summary": summary,
            "url": page.url,
            "query_used": query,
        })

    except DisambiguationError as e:
        logging.info(f"wiki_search: disambiguation for '{query}': {e.options[:10]}")
        return json.dumps({
            "wiki_search_error": f"'{query}' is ambiguous on Wikipedia.",
            "candidates": e.options[:10],
        })

    except PageError:
        logging.warning(f"wiki_search: page not found for '{query}'")
        return json.dumps({
            "wiki_search_error": f"No Wikipedia page found for '{query}'"
        })

    except WikipediaException as e:
        logging.error(f"wiki_search: Wikipedia API error for '{query}': {e}")
        return json.dumps({
            "wiki_search_error": "Wikipedia API error, try again later"
        })

    except Exception as e:
        logging.exception(f"wiki_search: unexpected failure for '{query}'")
        return json.dumps({
            "wiki_search_error": f"Unexpected error during wikipedia search: {e}"
        })


if __name__ == "__main__":
    mcp.run(transport="stdio")