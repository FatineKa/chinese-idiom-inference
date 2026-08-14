"""evaluation.py: text normalization and extraction of the target idiom
   from a CIP (src, dst) pair."""

def normalize(text: str) -> str:
    """Strip all whitespace. train.csv is space-segmented, but neither the
       test files nor natural Chinese text are -> normalize before Qwen."""
    return "".join(str(text).split())


def load_dictionary(path="data/raw/cip/idioms.txt"):
    """Return (set of idioms, list of lengths present).
       The lengths are used to spot idioms with a sliding window."""
    with open(path, encoding="utf-8") as f:
        dictionary = {l.strip() for l in f if l.strip()}
    lengths = sorted({len(i) for i in dictionary}, reverse=True)
    return dictionary, lengths


def idioms_of_length(dictionary: set, n: int) -> list:
    """Idioms with exactly n Chinese characters -- the candidate set I_n."""
    return [i for i in dictionary if len(i) == n]


def idioms_present(normalized_text: str, dictionary: set, lengths) -> set:
    """All dictionary idioms present in the (already normalized) text.
       Method: sliding window + O(1) set membership test.
       Much faster than testing all 31,113 idioms one by one."""
    found = set()
    n = len(normalized_text)
    for L in lengths:
        for k in range(n - L + 1):
            sub = normalized_text[k:k + L]
            if sub in dictionary:
                found.add(sub)
    return found


def find_idiom(src: str, dst: str, dictionary: set, lengths):
    """The target idiom is the one present in src but absent from dst (the
       one the paraphrase unfolded). Handles the false friend 众所周知
       (present on both sides -> discarded). Returns None if the case is
       ambiguous (0 or several candidates), to keep a clean evaluation set."""
    s = normalize(src)
    d = normalize(dst)
    candidates = idioms_present(s, dictionary, lengths) - idioms_present(d, dictionary, lengths)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None                      # 0 candidates or ambiguity -> skip the row
