"""BM25 (Okapi) с токенизатором под русский и коды моделей.

- лёгкий стемминг: отсекаются частые окончания («доставки», «доставку» -> «доставк»;
  «штрафах» -> «штраф»), основа не короче 4 букв и не длиннее 8;
- коды с цифрами сохраняются целиком и в латинице: «ТП-400» и «TP-400» -> «tp-400»,
  плюс отдельные части «tp», «400» (найдётся и «ТП 400»);
- ё -> е.
Своя реализация вместо rank_bm25: на 20 тыс. чанков считает за миллисекунды по обратному индексу.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"\w+(?:[-./]\w+)*")
_LOOKALIKE = str.maketrans({  # транслитерация кодов: «тп-400» -> «tp-400»
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch",
    "ш": "sh", "щ": "sch", "ы": "y", "э": "e", "ю": "yu", "я": "ya", "ь": "", "ъ": "",
})
MAX_STEM = 8
_ENDINGS = sorted("""ами ями ого его ому ему ыми ими иях ах ях ов ев ей ой ий ый ая яя ое ее ые ие
ую юю ом ем ам ям ою ею ия ии ию а я о е ы и у ю ь й""".split(), key=len, reverse=True)


def stem(word: str) -> str:
    for end in _ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 4:
            word = word[: -len(end)]
            break
    return word[:MAX_STEM]


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower().replace("ё", "е")):
        if any(ch.isdigit() for ch in tok):              # код модели, номер пункта, артикул
            code = tok.translate(_LOOKALIKE)
            out.append(code)
            parts = re.split(r"[-./]", code)
            if len(parts) > 1:
                out += [p for p in parts if p]
        else:
            for w in re.split(r"[-./]", tok):
                if w:
                    out.append(stem(w))
    return out


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len = np.zeros(0, dtype="float32")
        self.idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> "BM25":
        post = defaultdict(list)
        lens = []
        for i, toks in enumerate(docs):
            lens.append(len(toks))
            for t, c in Counter(toks).items():
                post[t].append((i, c))
        self.postings = dict(post)
        self.doc_len = np.asarray(lens, dtype="float32")
        n = len(docs)
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in post.items()}
        return self

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_len), dtype="float32")
        if not len(self.doc_len):
            return out
        avg = float(self.doc_len.mean()) or 1.0
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avg)
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out
