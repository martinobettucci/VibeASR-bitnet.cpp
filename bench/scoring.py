"""Text normalisation and WER/CER scoring, shared by the runner and the summariser.

Scoring is deliberately separate from inference: run_asr.py stores the raw hypothesis
next to the reference, so the normalisation policy can change without re-running a
multi-hour sweep. summarize.py re-scores from those stored records.

Two levels:

  basic    NFKC, lowercase, punctuation -> space, collapse whitespace.

  numbers  basic, plus every digit run spelled out in the language of the clip. This
           matters more than it sounds. FLEURS references write "le 35 mm"; the model
           says "trente-cinq millimètres". Unnormalised, one number costs several word
           errors, which is most of why French scores far worse than its place in the
           training mix suggests. Spelling out both sides is symmetric and is what
           Whisper-style evaluations do.

Languages num2words cannot spell fall back to `basic` and are marked in the output,
so a normalised number is never quietly claimed for a language that did not get one.
"""

import re
import sys
import unicodedata

try:
    from num2words import num2words
    _HAVE_N2W = True
except ImportError:  # scoring still works, just without number normalisation
    _HAVE_N2W = False

# Unicode punctuation -> space. Space rather than deletion so that "trente-cinq" and
# "trente cinq" normalise alike, instead of one becoming a single token.
_PUNCT = {i: " " for i in range(sys.maxunicode)
          if unicodedata.category(chr(i)).startswith("P")}

_DIGITS = re.compile(r"\d[\d.,]*")

_N2W_OK = {}


def can_spell_numbers(lang):
    """Whether num2words handles this ISO code."""
    if not _HAVE_N2W:
        return False
    if lang not in _N2W_OK:
        try:
            num2words(1, lang=lang)
            _N2W_OK[lang] = True
        except Exception:
            _N2W_OK[lang] = False
    return _N2W_OK[lang]


def _spell(match, lang):
    raw = match.group(0).rstrip(".,")
    # Strip thousands separators; give up on anything that is not a plain integer
    # (dates, versions, ranges) rather than guessing at its reading.
    cleaned = raw.replace(".", "").replace(",", "")
    if not cleaned.isdigit():
        return raw
    try:
        n = int(cleaned)
        # Integers in the plausible-year range are read as years: references write
        # "1940" and the model correctly says "nineteen forty", so the cardinal
        # spelling ("one thousand nine hundred forty") charged several word errors
        # for a transcription that was right. Measured on FLEURS English, the worst
        # clip's 35% WER was almost entirely this. Non-year quantities in the range
        # ("1500 copies") lose either way; years dominate in this corpus.
        if 1500 <= n <= 2099 and len(cleaned) == 4:
            try:
                return num2words(n, lang=lang, to="year")
            except Exception:
                pass
        return num2words(n, lang=lang)
    except Exception:
        return raw


def normalise(text, lang=None, numbers=False):
    text = unicodedata.normalize("NFKC", text or "").lower()
    if numbers and lang and can_spell_numbers(lang):
        text = _DIGITS.sub(lambda m: _spell(m, lang), text)
    text = text.translate(_PUNCT)
    return " ".join(text.split())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(ref, hyp, lang=None, numbers=False):
    """Returns (word_errors, ref_words, char_errors, ref_chars)."""
    r = normalise(ref, lang, numbers)
    h = normalise(hyp, lang, numbers)
    rw, hw = r.split(), h.split()
    return edit_distance(rw, hw), len(rw), edit_distance(r, h), len(r)
