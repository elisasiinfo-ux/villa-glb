"""Parse lyrics.txt into sections -> cards -> lines -> words."""
import re
from dataclasses import dataclass, field

TAG_RE = re.compile(r"\{(\w+)\}")
PUNCT_RE = re.compile(r"[^֐-׿A-Za-z0-9]")


@dataclass
class Word:
    text: str            # as displayed
    key: str             # normalized for alignment
    line: int
    card: int
    index: int = 0
    start: float = 0.0
    end: float = 0.0


@dataclass
class Line:
    text: str
    tags: set
    card: int
    section: str
    index: int = 0
    words: list = field(default_factory=list)


@dataclass
class Card:
    index: int
    section: str
    lines: list = field(default_factory=list)


def normalize(w):
    return PUNCT_RE.sub("", w)


def parse(path):
    cards, lines, words = [], [], []
    section = ""
    cur = None
    for raw in open(path, encoding="utf-8"):
        raw = raw.rstrip("\n")
        s = raw.strip()
        if s.startswith("##"):
            section = s[2:].strip()
            cur = None
            continue
        if s.startswith("#"):
            continue
        if not s:
            cur = None
            continue
        if cur is None:
            cur = Card(len(cards), section)
            cards.append(cur)
        tags = set(TAG_RE.findall(s))
        text = TAG_RE.sub("", s).strip()
        ln = Line(text, tags, cur.index, section, index=len(lines))
        for w in text.split():
            wd = Word(w, normalize(w), ln.index, cur.index, index=len(words))
            ln.words.append(wd)
            words.append(wd)
        cur.lines.append(ln)
        lines.append(ln)
    return cards, lines, words
