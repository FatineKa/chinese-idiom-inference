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


def sample_length_matched_distractors(target: str, by_length: dict, k: int, rng) -> list:
    """K distractor idioms for `target`, sampled first from idioms of the
    SAME length as the target (the candidate set I_n) -- so a candidate
    can't be told apart from the target purely by having a different
    character count, a shortcut unrelated to whether it actually fits the
    text (the dictionary is 95.4% length-4 idioms, so a uniform sample would
    almost always make a non-length-4 target the odd one out by length
    alone). Falls back to idioms of OTHER lengths only if I_n itself has
    fewer than k members besides the target.
    `by_length`: {length: list of idioms}, built once from a SORTED idiom
    list (set iteration order depends on PYTHONHASHSEED -- see script 09),
    not rebuilt on every call."""
    same_length = [i for i in by_length[len(target)] if i != target]
    if len(same_length) >= k:
        return rng.sample(same_length, k)
    rest = [i for n, idioms in by_length.items() if n != len(target) for i in idioms if i != target]
    return same_length + rng.sample(rest, k - len(same_length))


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
