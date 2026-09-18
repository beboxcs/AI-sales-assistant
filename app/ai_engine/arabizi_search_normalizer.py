import re

from app.ai_engine.arabic_search_normalizer import normalize_arabic_search_query
from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_intent_interpreter import interpret_search_query


_PHRASES = {
    "bdwr 3la": "looking for",
    "manate2 tanya": "مناطق تانية",
    "mo2adam ta7t": "مقدم لحد",
    "3ayez": "want",
    "3awez": "want",
    "m7tag": "need",
    "sha2a": "شقة",
    "sh2a": "شقة",
    "zaied": "zayed",
    "ta2seet": "تقسيط",
    "ta2sit": "تقسيط",
    "mo2adam": "مقدم",
    "lazem": "لازم",
    "darory": "ضروري",
    "dorory": "ضروري",
    "yefadal": "يفضل",
    "yfdl": "يفضل",
    "mofadal": "مفضل",
    "3ady": "عادي",
    "7awaly": "حوالي",
    "7awli": "حوالي",
    "ta2reban": "تقريبا",
    "alf": "ألف",
    "sneen": "سنين",
    "senen": "سنين",
    "ta7t": "أقل من",
    "fe": "في",
    "bas": "بس",
    "3la": "على",
    "mesh": "مش",
    "mosh": "مش",
}
_PHRASE_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(phrase).replace(r"\ ", r"\s+")
        for phrase in sorted(_PHRASES, key=len, reverse=True)
    ) + r")(?!\w)",
    re.IGNORECASE,
)
_INSTALLMENT_DURATION = re.compile(
    r"(?<!\w)(?:ta2seet|ta2sit)\s+(\d+(?:[.٫]\d+)?)\s+(?:sneen|senen)(?!\w)",
    re.IGNORECASE,
)


def normalize_arabizi_search_query(text: str) -> str:
    """Replace known Arabizi phrases, leaving other tokens and numeric text intact."""
    # The existing parser accepts duration-before-installments word order.
    normalized = _INSTALLMENT_DURATION.sub(r"\1 سنين تقسيط", text)
    return _PHRASE_PATTERN.sub(
        lambda match: _PHRASES[" ".join(match[0].lower().split())], normalized,
    )


def interpret_arabizi_search_query(text: str) -> ParsedSearchIntent:
    return interpret_search_query(
        normalize_arabic_search_query(normalize_arabizi_search_query(text))
    )
