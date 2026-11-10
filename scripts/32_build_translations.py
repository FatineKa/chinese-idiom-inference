"""32_build_translations.py — English meaning + pinyin for each idiom (demo page).

- Source: CC-CEDICT (free Chinese–English dictionary, CC BY-SA 4.0).
- Keeps only idioms that are in our dictionary.
- Idioms not in CC-CEDICT get no translation (nothing invented).
- Output: data/idiom_en.json  {idiom: {"pinyin": ..., "en": ...}}

Run:  python scripts/32_build_translations.py"""
import gzip
import json
import re
import urllib.request

from chengyu.evaluation import load_dictionary

URL = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
OUT = "data/idiom_en.json"

TONES = {"a": "āáǎà", "e": "ēéěè", "i": "īíǐì", "o": "ōóǒò", "u": "ūúǔù", "ü": "ǖǘǚǜ"}


def tone_marks(syllable: str) -> str:
    """'hua4' -> 'huà'. Tone 5 -> no mark."""
    s = syllable.replace("u:", "ü").replace("U:", "Ü")
    if not s[-1:].isdigit():
        return s
    tone, s = int(s[-1]), s[:-1]
    low = s.lower()
    if tone == 5:
        return s
    # rule: a, else e, else o of "ou", else last vowel
    if "a" in low:
        i = low.index("a")
    elif "e" in low:
        i = low.index("e")
    elif "ou" in low:
        i = low.index("o")
    else:
        i = max(low.rfind(v) for v in "iouü")
    if i < 0:
        return s
    mark = TONES[low[i]][tone - 1]
    return s[:i] + (mark.upper() if s[i].isupper() else mark) + s[i + 1:]


def clean(definitions: list) -> str:
    """Drop '(idiom)' tags. Keep the first 3 meanings."""
    defs = [re.sub(r"\s*\(idiom\)", "", d).strip() for d in definitions]
    return "; ".join([d for d in defs if d][:3])


dictionary, _ = load_dictionary()
raw = gzip.decompress(urllib.request.urlopen(URL, timeout=60).read()).decode("utf-8")

out = {}
line_re = re.compile(r"^\S+ (\S+) \[([^\]]+)\] /(.+)/$")
for line in raw.splitlines():
    m = line_re.match(line)
    if not m or m.group(1) not in dictionary or m.group(1) in out:
        continue
    simplified, pinyin, defs = m.groups()
    out[simplified] = {
        "pinyin": " ".join(tone_marks(s) for s in pinyin.split()).lower(),
        "en": clean(defs.split("/")),
    }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"{len(out)} / {len(dictionary)} idioms translated ({len(out) / len(dictionary):.0%}) -> {OUT}")
