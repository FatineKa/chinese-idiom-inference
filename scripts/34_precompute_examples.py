"""34_precompute_examples.py — results for the portfolio's pre-computed demo.

- Example sentences written for the demo (not from CIP), each with:
  - an English translation
  - the idiom the sentence was written for ("intended")
- Same method as the live demo: shortlist B -> Qwen + prior -> exact posterior.
- GPU -> K = 1000 candidates (recall@1000 = 83%, script 31).
- Needs: data/examples_index.npz, data/idiom_en.json
- Output: results/outputs/34_examples.json
- Run: CHENGYU_MODEL=Qwen/Qwen2.5-0.5B-Instruct python scripts/34_precompute_examples.py"""
import json
import os
import time

from chengyu.argmax import exact_posterior, text_scores
from chengyu.evaluation import load_dictionary
from chengyu.neighbours import load_examples, neighbour_candidates
from chengyu.prior import log_prior

K = 1000
TOP = 8

# (sentence, English, intended idiom)
EXAMPLES = [
    ("画已经很好了，他又多加了几笔，反而把画弄坏了。", "The painting was already good, but he added a few more strokes and ruined it.", "画蛇添足"),
    ("他在专家面前卖弄自己的一点知识。", "He shows off his little bit of knowledge in front of experts.", "班门弄斧"),
    ("他一直等着好运自己送上门，却什么也不做。", "He keeps waiting for good luck to come to him, without doing anything.", "守株待兔"),
    ("他每天早起晚睡，非常努力地工作。", "He gets up early and goes to bed late every day, working very hard.", "起早贪黑"),
    ("他们俩说的话完全一样，好像事先商量好了。", "The two of them said exactly the same thing, as if they had agreed beforehand.", "异口同声"),
    ("他讲得很认真，可是听的人一点也不懂。", "He explained it carefully, but the listeners did not understand a thing.", "对牛弹琴"),
    ("羊丢了以后，他马上把羊圈修好了。", "After the sheep were lost, he immediately repaired the pen.", "亡羊补牢"),
    ("他担心天会塌下来，整天睡不着觉。", "He worries that the sky will fall, and cannot sleep.", "杞人忧天"),
    ("为了让庄稼长得快，他把禾苗一棵棵往上拔。", "To make the crops grow faster, he pulled the seedlings up one by one.", "拔苗助长"),
    ("他住在小村子里，以为外面的世界就这么大。", "He lives in a small village and thinks the world is only that big.", "井底之蛙"),
    ("她做事一会儿做这个，一会儿做那个，从来不专心。", "She does one thing, then another, and never concentrates.", "三心二意"),
    ("他这样做，一次就得到了两个好处。", "By doing this, he got two benefits at once.", "一举两得"),
    ("大家一起努力，很快就把困难的工作完成了。", "Everyone worked together and quickly finished the hard job.", "齐心协力"),
    ("这个故事在我们国家，每家每户都知道。", "In our country, every household knows this story.", "家喻户晓"),
    ("做决定以前，他把每一步都想得很清楚。", "Before deciding, he thought every step through very carefully.", "深思熟虑"),
    ("他根本不会吹竽，却混在乐队里假装吹。", "He cannot play the yu at all, but hides in the band and pretends to play.", "滥竽充数"),
    ("他说他的矛什么都能刺穿，又说他的盾什么都刺不穿。", "He says his spear can pierce anything, and that his shield cannot be pierced by anything.", "自相矛盾"),
    ("他听到一点声音，就以为到处都是敌人。", "He hears a small noise and thinks there are enemies everywhere.", "草木皆兵"),
    ("老师讲了一个例子，他马上就能想到别的例子。", "The teacher gave one example, and he could immediately think of others.", "举一反三"),
    ("他丢了那份工作，后来却找到了更好的。", "He lost that job, but later found a better one.", "塞翁失马"),
    ("他们两个关系非常好，什么话都可以说。", "The two of them are very close and can talk about anything.", "无话不谈"),
    ("他看书的时候，连饭都忘了吃，觉也忘了睡。", "When he reads, he forgets to eat and to sleep.", "废寝忘食"),
]

dictionary, _ = load_dictionary()
examples = load_examples()
with open("data/idiom_en.json", encoding="utf-8") as f:
    english = json.load(f)

out = []
for text, en, intended in EXAMPLES:
    t0 = time.time()
    idioms = neighbour_candidates(text, *examples, k=K)
    likelihood = text_scores(text, idioms)
    h = {i: likelihood[i] + log_prior(i) for i in idioms}
    posterior = exact_posterior(h)
    ranked = sorted(h, key=h.get, reverse=True)
    rank = ranked.index(intended) + 1 if intended in h else None   # None = not in the K candidates
    out.append({
        "text": text, "en": en,
        "intended": intended,
        "intended_in_dictionary": intended in dictionary,
        "intended_rank": rank,
        "intended_pinyin": english.get(intended, {}).get("pinyin"),
        "intended_en": english.get(intended, {}).get("en"),
        "top": [{"idiom": i, "posterior": round(posterior[i], 4),
                 "pinyin": english.get(i, {}).get("pinyin"), "en": english.get(i, {}).get("en")}
                for i in ranked[:TOP]],
    })
    print(f"{intended}: rank {rank} | top-1 {ranked[0]} | {time.time() - t0:.0f} s")

ranks = [e["intended_rank"] for e in out]
print(f"\nintended idiom ranked 1st: {sum(r == 1 for r in ranks)}/{len(ranks)}"
      f" | in top {TOP}: {sum(r is not None and r <= TOP for r in ranks)}/{len(ranks)}"
      f" | not in the {K} candidates: {sum(r is None for r in ranks)}")
os.makedirs("results/outputs", exist_ok=True)
with open("results/outputs/34_examples.json", "w", encoding="utf-8") as f:
    json.dump({"model": os.environ.get("CHENGYU_MODEL"), "k": K, "examples": out}, f, ensure_ascii=False, indent=1)
print("saved results/outputs/34_examples.json")
