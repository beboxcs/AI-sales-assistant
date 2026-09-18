import re

from app.ai_engine.arabic_search_normalizer import normalize_arabic_search_query
from app.ai_engine.arabizi_search_normalizer import normalize_arabizi_search_query
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_intent_interpreter import interpret_search_query


_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefc]")


def interpret_multilingual_search_query(text: str) -> ParsedSearchIntent:
    """Route known multilingual vocabulary through the existing normalizers."""
    # The public normalizer recognizes its own vocabulary without a second marker table.
    normalized = normalize_arabizi_search_query(text)
    if normalized != text or _ARABIC_SCRIPT.search(text):
        return interpret_search_query(normalize_arabic_search_query(normalized))
    return interpret_search_query(text)
