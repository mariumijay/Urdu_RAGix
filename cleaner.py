"""
Urdu OCR Text Cleaner
Normalizes raw OCR Urdu text: fixes Unicode variants, removes noise,
preserves sentence boundaries, and handles Nastaliq-specific OCR errors.
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Unicode normalization maps
# ---------------------------------------------------------------------------

URDU_CHAR_MAP: dict[str, str] = {
    "\u064A": "\u06CC",  # ي → ی
    "\u0649": "\u06CC",  # ى → ی
    "\u0643": "\u06A9",  # ك → ک
    "\u0647": "\u06C1",  # ه → ہ
    "\u0640": "",        # tatweel (kashida) → removed
}

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

OCR_NOISE_CHARS = re.compile(r"[|\\/<>{}[\]#@$%^&*_=+`~]")
ENGLISH_CHARS   = re.compile(r"[a-zA-Z0-9]+")
MULTI_SPACE     = re.compile(r"[ \t]+")
MULTI_NEWLINE   = re.compile(r"\n{3,}")

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*[\u06F0-\u06F9\d]+\s*$", re.MULTILINE),
    re.compile(r"^\s*[-\u2013\u2014]\s*[\u06F0-\u06F9\d]+\s*[-\u2013\u2014]\s*$", re.MULTILINE),
    re.compile(r"\u0635\u0641\u062D\u06C1\s*[\u06F0-\u06F9\d]+"),
]

OCR_CORRECTIONS: dict[str, str] = {
    "\uFDF2": "\u0627\u0644\u0644\u06C1",          # ﷲ → اللہ
    "\uFDFA": "\u0635\u0644\u06CC \u0627\u0644\u0644\u06C1 \u0639\u0644\u06CC\u06C1 \u0648\u0633\u0644\u0645",
    "\uFDFB": "\u062C\u0644 \u062C\u0644\u0627\u0644\u06C1",
}

SENTENCE_END = re.compile(r"[\u06D4\u061F!]$")  # ۔ ؟ !


def clean_urdu_text(raw_text: str) -> str:
    """Full cleaning pipeline for raw Urdu OCR text."""
    text = raw_text

    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Known OCR ligature corrections
    for wrong, correct in OCR_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    # 3. Normalize Urdu character variants
    for src, dst in URDU_CHAR_MAP.items():
        text = text.replace(src, dst)

    # 4. Strip diacritics
    text = ARABIC_DIACRITICS.sub("", text)

    # 5. Remove OCR noise
    text = OCR_NOISE_CHARS.sub(" ", text)

    # 6. Remove stray English chars
    text = ENGLISH_CHARS.sub(" ", text)

    # 7. Remove page numbers / headers
    for pattern in PAGE_NUMBER_PATTERNS:
        text = pattern.sub("", text)

    # 8. Merge hyphenated line breaks
    text = re.sub(r"(\S)-\n(\S)", r"\1\2", text)

    # 9. Merge short broken lines
    lines = text.split("\n")
    merged: list[str] = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append("")
            continue
        buffer = (buffer + " " + line).strip() if buffer else line
        if SENTENCE_END.search(line):
            merged.append(buffer.strip())
            buffer = ""
    if buffer:
        merged.append(buffer.strip())

    text = "\n".join(merged)

    # 10. Normalize whitespace
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def normalize_for_search(text: str) -> str:
    """Lightweight normalization for query-time use (no English removal)."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in URDU_CHAR_MAP.items():
        text = text.replace(src, dst)
    text = ARABIC_DIACRITICS.sub("", text)
    text = MULTI_SPACE.sub(" ", text)
    return text.strip()
