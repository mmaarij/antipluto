"""
cleaners/unicode_normalizer.py
==============================
Unicode normalization, invisible character stripping, and homoglyph mapping.

Processes text strings to:
1. Normalize Unicode via NFKC (converts non-breaking spaces U+00A0 and fullwidth/math characters).
2. Strip invisible format control characters (category 'Cf', e.g. U+200F RTL mark, U+200B zero-width space, U+FEFF BOM).
3. Map Unicode homoglyphs (Armenian, Cyrillic, Greek, IPA, Cherokee lookalike characters) to standard ASCII equivalents.
"""

from __future__ import annotations

import unicodedata

# Exhaustive homoglyph translation table (Armenian, Cyrillic, Greek, IPA, Cherokee -> ASCII)
HOMOGLYPH_MAP: dict[int, int] = str.maketrans({
    # --- Complete Armenian Block (0x0531 - 0x058E) -> ASCII ---
    0x0531: 'A', 0x0561: 'a',
    0x0532: 'B', 0x0562: 'b',
    0x0533: 'G', 0x0563: 'g',
    0x0534: 'D', 0x0564: 'd',
    0x0535: 'E', 0x0565: 'e',
    0x0536: 'Z', 0x0566: 'z',
    0x0537: 'E', 0x0567: 'e',
    0x0538: 'E', 0x0568: 'e',
    0x0539: 'T', 0x0569: 't',
    0x053a: 'Z', 0x056a: 'z',
    0x053b: 'I', 0x056b: 'i',
    0x053c: 'L', 0x056c: 'l',
    0x053d: 'X', 0x056d: 'x',
    0x053e: 'C', 0x056e: 'c',
    0x053f: 'K', 0x056f: 'k',
    0x0540: 'H', 0x0570: 'h',
    0x0541: 'D', 0x0571: 'd',
    0x0542: 'G', 0x0572: 'g',
    0x0543: 'C', 0x0573: 'c',
    0x0544: 'M', 0x0574: 'm',
    0x0545: 'Y', 0x0575: 'y',
    0x0546: 'N', 0x0576: 'n',
    0x0547: 'S', 0x0577: 's',
    0x0548: 'O', 0x0578: 'n',
    0x0549: 'C', 0x0579: 'c',
    0x054a: 'P', 0x057a: 'u',
    0x054b: 'J', 0x057b: 'j',
    0x054c: 'R', 0x057c: 'r',
    0x054d: 'S', 0x057d: 'u',
    0x054e: 'V', 0x057e: 'v',
    0x054f: 'T', 0x057f: 't',
    0x0550: 'R', 0x0580: 'r',
    0x0551: 'C', 0x0581: 'c',
    0x0552: 'U', 0x0582: 'u',
    0x0553: 'P', 0x0583: 'p',
    0x0554: 'K', 0x0584: 'k',
    0x0555: 'O', 0x0585: 'o',
    0x0556: 'F', 0x0586: 'f',
    0x058c: 'l', 0x058e: 'u',

    # --- Complete Cyrillic Homoglyphs (Uppercase & Lowercase) ---
    '\u0410': 'A', '\u0430': 'a',
    '\u0411': 'B', '\u0431': 'b',
    '\u0412': 'B', '\u0432': 'v',
    '\u0413': 'F', '\u0433': 'r',
    '\u0414': 'D', '\u0434': 'd',
    '\u0415': 'E', '\u0435': 'e',
    '\u0416': 'ZH', '\u0436': 'zh',
    '\u0417': '3', '\u0437': 'z',
    '\u0418': 'N', '\u0438': 'u',
    '\u0419': 'Y', '\u0439': 'i',
    '\u041a': 'K', '\u043a': 'k',
    '\u041b': 'L', '\u043b': 'l',
    '\u041c': 'M', '\u043c': 'm',
    '\u041d': 'H', '\u043d': 'h',
    '\u041e': 'O', '\u043e': 'o',
    '\u041f': 'N', '\u043f': 'n',
    '\u0420': 'P', '\u0440': 'p',
    '\u0421': 'C', '\u0441': 'c',
    '\u0422': 'T', '\u0442': 't',
    '\u0423': 'Y', '\u0443': 'y',
    '\u0424': 'F', '\u0444': 'f',
    '\u0425': 'X', '\u0445': 'x',
    '\u0426': 'C', '\u0446': 'c',
    '\u0427': 'CH', '\u0447': 'h',
    '\u0428': 'W', '\u0448': 'w',
    '\u0429': 'W', '\u0449': 'w',
    '\u042a': 'b', '\u044a': 'b',
    '\u042b': 'bl', '\u044b': 'bl',
    '\u042c': 'b', '\u044c': 'b',
    '\u042d': 'E', '\u044d': 'e',
    '\u042e': 'IO', '\u044e': 'io',
    '\u042f': 'R', '\u044f': 'r',
    '\u0401': 'E', '\u0451': 'e',
    '\u0406': 'I', '\u0456': 'i',
    '\u0408': 'J', '\u0458': 'j',
    '\u0405': 'S', '\u0455': 's',
    '\u040e': 'U', '\u045e': 'u',
    '\u04bb': 'h',
    '\u0501': 'd',
    '\u051b': 'q',
    '\u051c': 'W', '\u051d': 'w',

    # --- Greek Homoglyphs (Uppercase & Lowercase) ---
    '\u0391': 'A', '\u03b1': 'a',
    '\u0392': 'B',
    '\u0395': 'E', '\u03b5': 'e',
    '\u0396': 'Z',
    '\u0397': 'H', '\u03b7': 'n',
    '\u0399': 'I', '\u03b9': 'i',
    '\u039a': 'K', '\u03ba': 'k',
    '\u039c': 'M',
    '\u039d': 'N', '\u03bd': 'v',
    '\u039f': 'O', '\u03bf': 'o',
    '\u03a1': 'P', '\u03c1': 'p',
    '\u03a4': 'T', '\u03c4': 't',
    '\u03a7': 'X', '\u03c7': 'x',
    '\u03a5': 'Y', '\u03c5': 'u',
    '\u03f2': 'c', '\u03f9': 'C',

    # --- IPA & Latin Extended Homoglyphs ---
    '\u0251': 'a', '\u0252': 'o', '\u0254': 'c', '\u0261': 'g',
    '\u0269': 'i', '\u0274': 'N', '\u0275': 'o', '\u0280': 'R', '\u028f': 'Y',

    # --- Cherokee Homoglyphs ---
    '\u13a0': 'T', '\u13a1': 'A', '\u13a4': 'y', '\u13a5': 'E',
    '\u13a9': 'H', '\u13ab': 'M', '\u13bd': 'Y', '\u13c7': 'Z',
    '\u13ce': 'S', '\u13cf': 'b', '\u13d5': 'W',
})


def normalize_unicode_text(text: str) -> str:
    """
    Normalize Unicode text, remove invisible control characters, and map homoglyphs.

    Parameters
    ----------
    text : str
        Input string (subject or body text).

    Returns
    -------
    str
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # 1. NFKC normalization (converts non-breaking spaces U+00A0 into standard spaces ' ')
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip Category 'Cf' (format control characters: zero-width spaces, RTL/LTR marks)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")

    # 3. Translate explicit homoglyph lookalikes to ASCII equivalents
    text = text.translate(HOMOGLYPH_MAP)

    # 4. Secondary fallback pass for any remaining Cyrillic (0x0400-0x052F) or Armenian (0x0530-0x058F)
    cleaned_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0x0400 <= code <= 0x052F or 0x0530 <= code <= 0x058F:
            name = unicodedata.name(ch, "")
            if "LETTER" in name:
                parts = name.split()
                last_part = parts[-1] if parts else "A"
                c = last_part.lower()[0] if "SMALL" in name else last_part.upper()[0]
                cleaned_chars.append(c)
            else:
                cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)

    return "".join(cleaned_chars)
