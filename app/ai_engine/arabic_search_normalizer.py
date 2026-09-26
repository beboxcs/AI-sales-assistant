import re
from decimal import Decimal

from app.ai_engine.search_intent import ParsedSearchIntent
from app.ai_engine.search_intent_interpreter import interpret_search_query


_CHARACTERS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹أإآىة٫،؛؟",
    "01234567890123456789ااايه.,;?",
)
_DIACRITICS = re.compile(r"[\u0640\u064b-\u065f\u0670]")

# Longer phrases precede their component words; unknown vocabulary is retained.
_PHRASES = (
    (r"التجمع الخامس", "Fifth Settlement"),
    (r"القاهره الجديده|التجمع", "New Cairo"),
    (r"الشيخ زايد|شيخ زايد|زايد", "Sheikh Zayed"),
    (r"6 اكتوبر|السادس من اكتوبر|اكتوبر", "6th of October"),
    (r"العاصمه الاداريه الجديده|العاصمه الاداريه", "New Capital"),
    (r"الساحل الشمالي", "North Coast"),
    (r"متر مربع", "sqm"),
    (r"مفروش|مفروشه", "furnished"),
    (r"مناطق (?:تانيه|اخري)", "other locations"),
    (r"انواع عقارات (?:تانيه|اخري)", "other property types"),
    (r"طرق دفع (?:تانيه|اخري)", "other payment methods"),
    (r"اطلاله علي البحر|فيو بحر|منظر البحر", "sea view"),
    (r"قريب من (?:المدرسه|مدرسه)|قرب (?:المدرسه|مدرسه)", "near school"),
    (r"قريب من (?:المترو|مترو)|قرب (?:المترو|مترو)", "near metro"),
    (r"منطقه هادئه|منطقه هاديه", "quiet area"),
    (r"تشطيب فاخر|تشطيب لوكس", "luxury finishing"),
    (r"استثمار جيد|استثمار كويس|فرصه استثماريه", "good investment"),
    (r"مساحه كبيره", "large unit"),
    (r"شقه كبيره|شقه واسعه", "large apartment"),
    (r"فيلا كبيره|فيلا واسعه", "large villa"),
    (r"سعر رخيص", "cheap unit"),
    (r"شقه رخيصه", "cheap apartment"),
    (r"مقدم قليل|مقدم منخفض", "low down payment"),
    (r"تقسيط طويل|اقساط طويله", "long installments"),
    (r"ميزانيه مستهدفه", "target budget"),
    (r"مقدم بحد اقصي|اقصي مقدم|حد اقصي للمقدم", "max down payment"),
    (r"مقدم (?:لحد|حتي)", "down payment up to"),
    (r"ميزانيه بحد اقصي|اقصي ميزانيه|حد اقصي للميزانيه", "max budget"),
    (r"(?:بالتقسيط|تقسيط|اقساط) علي", "installments over"),
    (r"غرف نوم|غرفه نوم|اوض نوم|اوضه نوم|غرف|اوض", "bedrooms"),
    (r"حمامات|حمام", "bathrooms"),
    (r"طريقه الدفع|طرق الدفع", "payment method"),
    (r"نوع العقار", "property type"),
    (r"مرن (?:في|بخصوص)", "flexible on"),
    (r"لازم|ضروري", "must"),
    (r"مطلوب|مطلوبه|الزامي", "required"),
    (r"يفضل|افضل", "prefer"),
    (r"مفضل|مفضله", "preferred"),
    (r"حوالي|تقريبا", "around"),
    (r"اقل من", "under"),
    (r"اكثر من|اكتر من", "above"),
    (r"علي الاقل|بحد ادني|حد ادني", "minimum"),
    (r"لحد|حتي", "up to"),
    (r"ميزانيه|الميزانيه", "budget"),
    (r"السعر|سعر", "price"),
    (r"مليون|ملايين", "million"),
    (r"الف|الاف", "thousand"),
    (r"جنيه مصري|جنيه", "EGP"),
    (r"شقه", "apartment"),
    (r"فيلا|فيله", "villa"),
    (r"كاش|نقدا|نقدي", "cash"),
    (r"بالتقسيط|تقسيط|اقساط", "installments"),
    (r"سنوات|سنين|سنه|عام|اعوام", "years"),
    (r"مقدم", "down payment"),
    (r"مناطق|منطقه|المكان", "location"),
    (r"(?:مش|ليس|لا) (?:عادي|مقبول|مقبوله)", "are not okay"),
    (r"عادي|مقبول|مقبوله", "are okay"),
    (r"مش|ليس|لا", "not"),
    (r"الا|باستثناء", "except"),
    (r"بس|لكن", "but"),
    (r"و", "and"),
    (r"في", "in"),
)
_PHRASE_PATTERNS = tuple(
    (re.compile(rf"(?<!\w)(?:{phrase})(?!\w)"), english)
    for phrase, english in _PHRASES
)
_THOUSANDS = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s+(?:الف|الاف)(?!\w)"
)


def _expand_thousands(match: re.Match[str]) -> str:
    # Tuple construction shifts the decimal point without context rounding.
    amount = Decimal(match[1].replace(",", "")).as_tuple()
    return format(Decimal((amount.sign, amount.digits, amount.exponent + 3)), "f")


def normalize_arabic_search_query(text: str) -> str:
    """Normalize known Arabic search phrases, preserving unknown/English text."""
    normalized = " ".join(_DIACRITICS.sub("", text).translate(_CHARACTERS).split())
    normalized = re.sub(r"(?<=\d)٬(?=\d)", "", normalized)
    normalized = _THOUSANDS.sub(_expand_thousands, normalized)
    for pattern, english in _PHRASE_PATTERNS:
        normalized = pattern.sub(english, normalized)
    return normalized


def interpret_arabic_search_query(text: str) -> ParsedSearchIntent:
    return interpret_search_query(normalize_arabic_search_query(text))
