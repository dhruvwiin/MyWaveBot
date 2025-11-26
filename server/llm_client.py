"""Light wrapper around the Perplexity SDK.

This module attempts to import the `perplexity` package (from `perplexityai`).
If not present, it provides a safe local fallback.
"""

from typing import AsyncGenerator, List, Dict, Any

from .settings import settings

# Try to import the official SDK.
HAS_PERPLEXITY = False
try:
    from perplexity import Perplexity
    HAS_PERPLEXITY = True
except ImportError:
    HAS_PERPLEXITY = False

# Initialize client if possible
client = None
if HAS_PERPLEXITY:
    try:
        client = Perplexity(api_key=settings.PERPLEXITY_API_KEY)
    except Exception as e:
        print(f"Error initializing Perplexity client: {e}")
        client = None

# Default search domains and filters
DOMAIN_FILTER = settings.SEARCH_DOMAINS.split(',')
DEFAULT_SEARCH_OPTIONS = {
    "search_domain_filter": DOMAIN_FILTER,
    "search_recency_filter": "year",
    # "search_context_size": "low", # Removed as it might not be supported in all versions/models
}
DEFAULT_MODEL = "sonar-pro" # Low-cost online model

# Core LLM Functions
async def chat_stream(user_message: str):
    """
    Streams a response from Perplexity AI using the configured model.
    Yields text chunks as they arrive.
    """
    if not client:
        yield "Perplexity client not configured. Please check your API key."
        return

    try:
        # Build the system prompt - emphasize brevity
        system_prompt = (
            f"You are a helpful assistant for {settings.UNIVERSITY_NAME}. "
            f"Search only: {settings.SEARCH_DOMAINS}. "
            "CRITICAL: Keep responses under 100 words. Be direct and concise. "
            "Provide only essential information with citations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Perplexity streaming call
        search_options = {}
        if settings.SEARCH_DOMAINS:
            search_options["search_domain_filter"] = settings.SEARCH_DOMAINS.split(',')

        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True,
            max_tokens=250,  # Reduced from 400 to 250 for shorter responses
            **search_options,
        )

        citations = []
        for chunk in stream:
            # Different SDK versions may use slightly different shapes; we
            # guard access to avoid AttributeErrors.
            try:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        print(f"Yielding content chunk: {content[:20]}...")
                        yield content
                
                # Collect citations if present
                if getattr(chunk, "citations", None):
                    citations = chunk.citations
                    print(f"Found citations in chunk: {citations}")
            except Exception as e:
                # Ignore malformed chunks
                print(f"Error processing chunk: {e}")
                continue
        
        # Also check if the final chunk has citations
        print(f"Final citations collected: {citations}")
        
        # Yield citations at the end as a special message
        if citations:
            import json
            citations_json = json.dumps(citations)
            print(f"Sending citations to frontend: {citations_json}")
            yield f"\n\n__CITATIONS__:{citations_json}"
        else:
            print("No citations found in response")

    except Exception as e:
        print(f"Perplexity API Error: {e}")
        yield f"Error: The chat service is currently unavailable. Please try again later. Details: {e}"

# Example non-streaming/structured function (optional, not used in initial app.py)
async def classify_topic(text: str) -> str:
    """
    Uses a cheap, offline model to classify a query's topic.
    """
    # Fallback: a very small rule-based classifier so the app remains
    # functional even without networked LLM access.
    if not client:
        txt = text.lower()
        if 'registrar' in txt or 'add' in txt or 'drop' in txt:
            return 'registration'
        if 'bursar' in txt or 'tuition' in txt or 'bill' in txt:
            return 'bursar'
        if 'wifi' in txt or 'eduroam' in txt or 'network' in txt:
            return 'wifi'
        return 'other'

    messages = [
        {
            "role": "system",
            "content": "Classify the user query into one word: 'registration', 'bursar', 'housing', 'wifi', 'other'.",
        },
        {"role": "user", "content": text},
    ]

    try:
        response = client.chat.completions.create(
            model='sonar',  # Lightweight model
            messages=messages,
            max_tokens=10,
            # disable_search=True,  # Crucial for cost control on classification - check SDK support
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Perplexity API Error (Classification): {e}")
        return 'unclassified'