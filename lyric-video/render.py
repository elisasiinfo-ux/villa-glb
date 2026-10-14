"""Hebrew lyric-video renderer.

Reads lyrics.txt (layout/effect tags) + timings.json (word times from align.py) and
writes each sung word on screen at the moment it is sung, over a calm, almost static
background. The video opens on the title cover and dissolves into the same picture
without the title (assets/background.webp, see make_clean_background.py) before the
singing starts, then stays on that picture to the end.

usage:
  python render.py --audio song.mp3 --timings timings.json --out video.mp4
  python render.py ... --stills 15.0,60.5     # preview PNGs instead of video
"""
import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from lyrics_parse import parse

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
W, H = 1920, 1080

FONT_HAND = os.path.join(ASSETS, "fonts", "KtavYadCLM-BoldItalic.otf")
FONT_HAND_LIGHT = os.path.join(ASSETS, "fonts", "KtavYadCLM-MediumItalic.otf")
FONT_GRUNGE = os.path.join(ASSETS, "fonts", "RubikDirt-Hebrew.ttf")
FONT_PRAYER = os.path.join(ASSETS, "fonts", "FrankRuhlLibre-Hebrew-500.ttf")

CREAM = (250, 241, 220)
GOLD = (255, 214, 140)
GOLD_GLOW = (255, 170, 60)

SIZE = {"normal": 112, "big": 158, "huge": 250}
SIZE_GRUNGE = {"normal": 96, "big": 132, "huge": 205}
SIZE_PRAYER = {"normal": 150, "big": 190, "huge": 250}   # closing prayer: gold serif, enlarged

EXIT_DUR = {"default": 0.35, "crash": 1.2, "wind": 1.1, "sink": 0.9, "fall": 0.6,
            "glitch": 0.5, "prayer": 0.6, "angel": 0.6, "devil": 0.5}
ENTRY_LEAD = {"write": 0.18}   # seconds an entry starts before the sung onset (others: 0.15);
                               # the write-on lead also covers the sprite's empty right margin

COVER_DISSOLVE = 3.0      # seconds of cross-dissolve from the title cover into the clip
ZOOM_END = 1.03           # background creeps from 1.00 to this zoom over the whole song


# ----------------------------------------------------------------------------- easing
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(p):
    p = clamp01(p)
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    p = clamp01(p)
    return p ** 3


def ease_in_out(p):
    p = clamp01(p)
    return p * p * (3 - 2 * p)


def hash01(*xs):
    h = 0
    for x in xs:
        h = (h * 1000003) ^ int(x * 1000 + 7919)
        h &= 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 0x5bd1e995 & 0xFFFFFFFF
    return (h ^ (h >> 15)) / 0xFFFFFFFF


# ----------------------------------------------------------------------------- sprites
def blur(arr, r):
    if r < 0.4:
        return arr
    return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(r)))


def make_sprite(word, font, size, fill, glow=None, angle=0.0):
    """RGBA sprite of one word with soft shadow (+optional glow).
    Returns (array, anchor_x, anchor_y, advance): anchor = centre of the advance on the baseline."""
    bbox = font.getbbox(word, anchor="ls", direction="rtl")
    adv = font.getlength(word, direction="rtl")
    pad = int(size * 0.55)
    w = int(bbox[2] - bbox[0]) + 2 * pad
    h = int(bbox[3] - bbox[1]) + 2 * pad
    ox, oy = pad - bbox[0], pad - bbox[1]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((ox, oy), word, font=font, fill=255, anchor="ls", direction="rtl")
    m = np.asarray(mask).astype(np.float32) / 255.0

    def shifted(a, dx, dy):
        out = np.zeros_like(a)
        out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
            a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
        return out

    def gblur(a, r):
        im = Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
        return np.asarray(im).astype(np.float32) / 255.0

    halo = gblur(m, size * 0.22) * 0.55
    shadow = gblur(shifted(m, int(size * 0.035), int(size * 0.05)), size * 0.06) * 0.85
    # premultiplied composition: halo (black) -> shadow (black) -> glow -> fill
    rgb = np.zeros((h, w, 3), np.float32)
    a = shadow + halo * (1 - shadow)
    if glow is not None:
        g = np.clip(gblur(m, size * 0.16) * 1.6, 0, 1) * 0.9
        rgb = np.asarray(glow, np.float32)[None, None] * g[..., None] + rgb * (1 - g[..., None])
        a = g + a * (1 - g)
    col = np.asarray(fill, np.float32)[None, None]
    rgb = col * m[..., None] + rgb * (1 - m[..., None])
    a = m + a * (1 - m)
    un = np.where(a[..., None] > 1e-4, rgb / np.maximum(a[..., None], 1e-4), 0)
    arr = np.dstack([np.clip(un, 0, 255), np.clip(a * 255, 0, 255)]).astype(np.uint8)
    ax, ay = ox + adv / 2.0, oy
    if abs(angle) > 0.05:
        im = Image.fromarray(arr, "RGBA")
        cx, cy = w / 2, h / 2
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
        th = math.radians(angle)
        dx, dy = ax - cx, ay - cy
        rx = dx * math.cos(th) + dy * math.sin(th)
        ry = -dx * math.sin(th) + dy * math.cos(th)
        ax, ay = im.width / 2 + rx, im.height / 2 + ry
        arr = np.asarray(im)
    return arr, ax, ay, adv


def scale_sprite(arr, ax, ay, s):
    """Scale exactly about the anchor, keeping the canvas (no rounding to whole pixels,
    so the size never snaps at the end of an animation)."""
    if s == 1.0:
        return arr, ax, ay
    h, w = arr.shape[:2]
    im = Image.fromarray(arr).transform((w, h), Image.AFFINE,
                                        (1 / s, 0, ax - ax / s, 0, 1 / s, ay - ay / s),
                                        resample=Image.BILINEAR)
    return np.asarray(im), ax, ay


def with_alpha(arr, mult):
    if isinstance(mult, (float, int)) and mult >= 0.999:
        return arr
    out = arr.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * mult).astype(np.uint8)
    return out


def wipe_rtl(arr, p, edge=48):
    """Reveal from the right edge to the left (Hebrew writing direction)."""
    if p >= 1:
        return arr
    w = arr.shape[1]
    xb = (w + edge) * (1 - p) - edge
    ramp = np.clip((np.arange(w, dtype=np.float32) - xb) / edge, 0, 1)
    return with_alpha(arr, ramp[None, :])


# ----------------------------------------------------------------------------- scene
@dataclass
class WordItem:
    text: str
    start: float
    end: float
    sprite: np.ndarray
    ax: float
    ay: float
    adv: float = 0.0
    x: float = 0.0          # target anchor position on screen
    y: float = 0.0
    k: int = 0              # index inside its line
    shards: list = None


@dataclass
class LineItem:
    text: str
    tags: set
    size_class: str
    font_kind: str
    words: list = field(default_factory=list)
    entry: str = "write"
    exit: str = "default"
    angle: float = 0.0
    seed: float = 0.0
    space: float = 0.0
    size: int = 0
    t_exit: float = 0.0     # this line's own exit time (never later than its card's)


@dataclass
class CardItem:
    index: int
    section: str
    lines: list
    t_in: float = 0.0
    t_exit: float = 0.0


FX_ENTRY = ["stamp", "slam", "crash", "fall", "glitch", "angel", "devil", "prayer", "pulse"]
FX_EXIT = ["crash", "wind", "sink", "fall", "glitch", "prayer", "angel", "devil"]


def build_scene(lyrics_path, timings_path):
    cards, lines, words = parse(lyrics_path)
    tm = json.load(open(timings_path))
    assert len(tm) == len(words), f"timings ({len(tm)}) do not match lyrics ({len(words)})"
    for w, t in zip(words, tm):
        w.start, w.end = t["start"], t["end"]

    fonts = {}

    def font(path, size):
        key = (path, size)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
        return fonts[key]

    scene = []
    for c in cards:
        items = []
        for ln in c.lines:
            tags = ln.tags
            sc = "huge" if "huge" in tags else "big" if "big" in tags else "normal"
            kind = "prayer" if "prayer" in tags else "grunge" if "grunge" in tags else "hand"
            li = LineItem(ln.text, tags, sc, kind, seed=hash01(ln.index, 3))
            li.entry = next((f for f in FX_ENTRY if f in tags), "pop" if sc != "normal" else "write")
            li.exit = next((f for f in FX_EXIT if f in tags), "default")
            tilt = 0.0 if kind == "prayer" else 5.0 if sc == "huge" else 7.0
            li.angle = (hash01(ln.index, 11) - 0.55) * tilt
            path = {"hand": FONT_HAND, "grunge": FONT_GRUNGE, "prayer": FONT_PRAYER}[kind]
            size = {"hand": SIZE, "grunge": SIZE_GRUNGE, "prayer": SIZE_PRAYER}[kind][sc]
            space_k = {"hand": 0.55, "grunge": 1.7, "prayer": 1.1}[kind]
            f = font(path, size)
            space = f.getlength(" ") * space_k
            total = sum(f.getlength(w.text.replace(",", ""), direction="rtl") for w in ln.words) \
                + space * (len(ln.words) - 1)
            if total > 1720:                       # auto-shrink very long lines
                size = int(size * 1720 / total)
                f = font(path, size)
                space = f.getlength(" ") * space_k
            fill, glow = CREAM, None
            if kind == "prayer":
                fill, glow = GOLD, GOLD_GLOW
            elif li.entry == "angel":
                fill, glow = (255, 255, 252), (255, 236, 200)
            elif li.entry == "devil":
                fill, glow = (235, 38, 24), (120, 0, 0)
            for k, w in enumerate(ln.words):
                spr, ax, ay, adv = make_sprite(w.text.replace(",", ""), f, size, fill, glow, li.angle)
                li.words.append(WordItem(w.text, w.start, w.end, spr, ax, ay, adv=adv, k=k))
            li.space = space
            li.size = size
            items.append(li)
        scene.append(CardItem(c.index, c.section, items))
    layout(scene)
    timing(scene)
    return scene


def layout(scene):
    for card in scene:
        n = len(card.lines)
        hs = [ln.size * 1.18 for ln in card.lines]
        top, bottom = 190, 1010
        gap = 0 if n == 1 else min(230, max(50, (bottom - top - 60 - sum(hs)) / (n - 1)))
        total = sum(hs) + gap * (n - 1)
        y = (top + bottom) / 2 - total / 2 + (hash01(card.index, 5) - 0.5) * 60
        for ln, hgt in zip(card.lines, hs):
            pos = "R" if "R" in ln.tags else "L" if "L" in ln.tags else "C" if "C" in ln.tags else "R"
            widths = [w.adv for w in ln.words]
            total_w = sum(widths) + ln.space * (len(widths) - 1)
            jit = hash01(card.index, ln.seed * 100)
            if pos == "R":
                right = W - 90 - jit * 70
            elif pos == "L":
                right = 90 + jit * 70 + total_w
            else:
                right = W / 2 + total_w / 2 + (jit - 0.5) * 80
            base_y = y + hgt * 0.80
            cx, cy = right - total_w / 2, base_y
            th = math.radians(ln.angle)
            x = right
            for w in ln.words:   # Hebrew: first word on the right
                lx = x - w.adv / 2 - cx
                w.x = round(cx + lx * math.cos(th) - w.ax) + w.ax
                w.y = round(cy - lx * math.sin(th) - w.ay) + w.ay
                x -= w.adv + ln.space
            ink = _line_box(ln, 0)            # flourishes can reach past the glyph advance
            dx = round(min(0, W - 60 - ink[2]) + max(0, 60 - ink[0]))
            for w in ln.words:
                w.x += dx
            y += hgt + gap


def timing(scene):
    for card in scene:
        card.t_in = min(w.start for ln in card.lines for w in ln.words)
    for i, card in enumerate(scene):
        ws = [w for ln in card.lines for w in ln.words]
        last = max(ws, key=lambda w: w.start)
        nxt = scene[i + 1].t_in if i + 1 < len(scene) else 1e9
        hold = 2.2 if card.lines[0].font_kind != "prayer" else 3.5
        t = min(nxt - 0.02, max(last.end, last.start + 0.45) + hold)
        card.t_exit = max(t, last.start + 0.40)
        for ln in card.lines:
            ln.t_exit = card.t_exit
            if i + 1 == len(scene):
                continue
            # a line that the next card writes over leaves early enough to be gone first,
            # but never before it has been sung and shown for a moment
            lw = max(ln.words, key=lambda w: w.start)
            floor = max(lw.end, lw.start + 0.45) + 0.4
            gone = EXIT_DUR.get(ln.exit, EXIT_DUR["default"]) + 0.03 * len(ln.words)
            box = _line_box(ln)
            for nl in scene[i + 1].lines:
                if _boxes_touch(box, _line_box(nl)):
                    t0 = min(w.start for w in nl.words) - ENTRY_LEAD.get(nl.entry, 0.15)
                    ln.t_exit = min(ln.t_exit, max(floor, t0 - 0.05 - gone))
        if card.lines[0].font_kind == "prayer":
            # the prayer screens change as a whole
            t = min(ln.t_exit for ln in card.lines)
            for ln in card.lines:
                ln.t_exit = t


def _line_box(ln, pad=24):
    """Screen bounding box of a line's ink at rest (x0, y0, x1, y1), padded."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for w in ln.words:
        ink = w.sprite[..., 3] > 24
        rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
        if not len(rows):
            continue
        x0, y0 = w.x - w.ax, w.y - w.ay
        xs0.append(x0 + cols[0])
        xs1.append(x0 + cols[-1])
        ys0.append(y0 + rows[0])
        ys1.append(y0 + rows[-1])
    return (min(xs0) - pad, min(ys0) - pad, max(xs1) + pad, max(ys1) + pad)


def _boxes_touch(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ----------------------------------------------------------------------------- word animation
def word_ops(ln, w, card, t):
    """Return a list of (sprite, x, y, alpha) draw ops for one word at time t.

    All motion is smooth and starts from rest: no bounce, overshoot, flicker or jitter,
    and words do not move at all while they sit on screen."""
    e = ln.entry
    t0 = w.start - ENTRY_LEAD.get(e, 0.15)
    if t < t0:
        return []
    te = ln.t_exit + 0.03 * w.k
    exit_dur = EXIT_DUR.get(ln.exit, EXIT_DUR["default"])
    if t > te + exit_dur:
        return []
    arr, ax, ay = w.sprite, w.ax, w.ay
    x, y, alpha, s = w.x, w.y, 1.0, 1.0
    dt = t - t0

    if e == "write":
        dur = min(0.5, max(0.3, 0.8 * (w.end - w.start)))
        arr = wipe_rtl(arr, ease_in_out(dt / dur))
        y += 6 * (1 - ease_in_out(dt / dur))
    elif e in ("pop", "pulse"):
        p = dt / 0.5
        s *= 0.95 + 0.05 * ease_in_out(p)
        alpha = ease_in_out(dt / 0.4)
        if p < 1:
            arr = blur(arr, 3 * (1 - ease_in_out(p)))
    elif e in ("stamp", "crash"):
        p = dt / 0.6
        s *= 1 + 0.06 * (1 - ease_in_out(p))
        alpha = ease_in_out(dt / 0.4)
        if p < 1:
            arr = blur(arr, 3 * (1 - ease_in_out(p)))
    elif e == "slam":
        p = dt / 0.55
        y -= 24 * (1 - ease_in_out(p))
        alpha = ease_in_out(dt / 0.4)
    elif e == "fall":
        p = dt / 0.6
        y -= 30 * (1 - ease_in_out(p))
        alpha = ease_in_out(dt / 0.4)
    elif e == "glitch":
        p = dt / 0.7
        alpha = ease_in_out(dt / 0.4)
        if p < 1:
            arr = chroma(arr, 1 - ease_in_out(p))
    elif e == "angel":
        p = dt / 0.7
        alpha = ease_out_cubic(p)
        y += 30 * (1 - ease_out_cubic(p))
        if p < 1:
            arr = blur(arr, 8 * (1 - p))
    elif e == "devil":
        p = dt / 0.6
        alpha = ease_out_cubic(p)
        if p < 1:
            arr = blur(arr, 6 * (1 - p))
    elif e == "prayer":
        p = dt / 0.75
        alpha = ease_in_out(p)
        y += 20 * (1 - ease_out_cubic(p))
        if p < 1:
            arr = blur(arr, 10 * (1 - ease_out_cubic(p)))

    # exits
    if t > te:
        q = (t - te) / exit_dur
        x_ = ln.exit
        if x_ == "crash":
            return crash_ops(w, x, y, s, q)
        if x_ == "wind":
            arr, shift = wind(arr, q, w)
            ax = ax + shift
            alpha *= 1 - ease_in_cubic(q)
        elif x_ == "sink":
            y += 120 * ease_in_out(q)
            arr = blur(arr, 14 * q)
            alpha *= 1 - ease_in_out(q)
        elif x_ == "fall":
            y += 90 * ease_in_cubic(q)
            alpha *= 1 - ease_in_out(q)
        elif x_ == "glitch":
            arr = chroma(arr, 0.7 * ease_in_out(q))
            alpha *= 1 - ease_in_out(q)
        elif x_ in ("prayer", "angel"):
            y -= 30 * ease_in_out(q)
            arr = blur(arr, 8 * q)
            alpha *= 1 - ease_in_out(q)
        else:   # default (also devil)
            y -= 20 * ease_in_out(q)
            arr = blur(arr, 6 * q)
            alpha *= 1 - ease_in_out(q)

    arr, ax2, ay2 = scale_sprite(arr, ax, ay, s)
    return [(arr, x - ax2, y - ay2, alpha)]


def chroma(arr, amount):
    """Static RGB split (red left / cyan right) whose offset and strength both scale
    smoothly with `amount`, so it fades in and out without any jump."""
    amount = max(0.0, min(1.0, amount))
    if amount < 0.02:
        return arr
    d = max(1, int(round(14 * amount)))
    a = arr[..., 3].astype(np.float32) / 255
    rgb = arr[..., :3].astype(np.float32)
    glyph = a * np.clip((rgb.mean(axis=2) - 60) / 150, 0, 1)      # the letters, not the shadow
    gl = np.roll(glyph, -d, axis=1)
    gr = np.roll(glyph, d, axis=1)
    k = 0.85 * amount
    wl = gl * (1 - glyph) * k
    wr = gr * (1 - glyph) * (1 - gl * k) * k
    red = np.array([255, 30, 60], np.float32)
    cyan = np.array([30, 230, 255], np.float32)
    base = rgb * a[..., None]
    col_p = base * (1 - wl - wr)[..., None] + red * wl[..., None] + cyan * wr[..., None]
    out_a = a + (wl + wr) * (1 - a)
    col = col_p / np.maximum(out_a, 1e-4)[..., None]
    return np.dstack([np.clip(col, 0, 255), np.clip(out_a * 255, 0, 255)]).astype(np.uint8)


def wind(arr, q, w):
    """Thin strips of the word drift away to the left and dissolve.
    Returns (array, shift): the original content now starts `shift` px to the right."""
    if q <= 0:
        return arr, 0
    h, wd = arr.shape[:2]
    extra = int(380 * q * q) + 2
    out = np.zeros((h, wd + extra, 4), np.uint8)
    strip = 4
    for i, y0 in enumerate(range(0, h, strip)):
        r = hash01(i, len(w.text))
        dx = int(extra * (0.3 + 0.7 * r) * clamp01(q * 1.5 - (y0 / h) * 0.5))
        seg = arr[y0:y0 + strip].copy()
        seg[..., 3] = (seg[..., 3].astype(np.float32) * clamp01(1 - q * (0.5 + r))).astype(np.uint8)
        x0 = extra - dx
        out[y0:y0 + strip, x0:x0 + wd] = seg
    return out, extra


def crash_ops(w, x, y, s, q):
    """The word breaks into pieces that slowly drift apart, sink and fade."""
    if w.shards is None:
        h, wd = w.sprite.shape[:2]
        cols, rows = 7, 3
        shards = []
        for i in range(cols):
            for j in range(rows):
                x0, x1 = int(wd * i / cols), int(wd * (i + 1) / cols)
                y0, y1 = int(h * j / rows), int(h * (j + 1) / rows)
                piece = w.sprite[y0:y1, x0:x1]
                if piece[..., 3].max() < 20:
                    continue
                cxp, cyp = (x0 + x1) / 2, (y0 + y1) / 2
                r1, r2, r3 = hash01(i, j, 1), hash01(i, j, 2), hash01(i, j, 3)
                shards.append(dict(img=Image.fromarray(np.ascontiguousarray(piece)), cx=cxp, cy=cyp,
                                   vx=(cxp - w.ax) * 0.5 + (r1 - 0.5) * 120,
                                   vy=-40 - r2 * 80, rot=(r3 - 0.5) * 120))
        w.shards = shards
    T = EXIT_DUR["crash"] * q * q          # starts from rest, then drifts
    ops = []
    for sh in w.shards:
        px = x + (sh["cx"] - w.ax) * s + sh["vx"] * T
        py = y + (sh["cy"] - w.ay) * s + sh["vy"] * T + 0.5 * 500 * T * T
        im = sh["img"].rotate(sh["rot"] * T, resample=Image.BILINEAR, expand=True)
        a = np.asarray(im)
        ops.append((a, px - a.shape[1] / 2, py - a.shape[0] / 2, 1 - ease_in_out(q)))
    return ops


# ----------------------------------------------------------------------------- background
class Background:
    """Calm background: fixed framing with an almost imperceptible zoom, light vignette
    and a fine static grain. `cover_mix` cross-dissolves to the title cover."""

    def __init__(self, bg_path, cover_path):
        self.S = 1.5

        def prep(path):
            src = Image.open(path).convert("RGB")
            return src.resize((int(src.width * self.S), int(src.height * self.S)), Image.LANCZOS)

        self.bg = prep(bg_path)
        self.cover = prep(cover_path)
        assert self.bg.size == self.cover.size, "cover and background must share a size"
        self.bw, self.bh = self.bg.size
        # 16:9 window at zoom 1 spans the full width; centre it a little above the middle
        # so the whole title on the cover stays in frame
        self.cy = min(max(self.bh * 0.44, self.bw * H / W / 2), self.bh - self.bw * H / W / 2)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = (0.95 * (1 - 0.42 * np.clip(r / 1.3, 0, 1) ** 2.4))[..., None].astype(np.float32)
        g = np.random.default_rng(7).normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        self.grain = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)[..., None] * 3.0   # static texture

    def render(self, t, zoom, cover_mix):
        vw = self.bw / zoom
        vh = vw * H / W
        cx, cy = self.bw / 2, min(max(self.cy, vh / 2), self.bh - vh / 2)
        sc = vw / W
        data = (sc, 0, cx - vw / 2, 0, sc, cy - vh / 2)
        arr = None
        if cover_mix < 0.999:
            arr = np.asarray(self.bg.transform((W, H), Image.AFFINE, data, resample=Image.BILINEAR)).astype(np.float32)
        if cover_mix > 0.001:
            c = np.asarray(self.cover.transform((W, H), Image.AFFINE, data, resample=Image.BILINEAR)).astype(np.float32)
            arr = c if arr is None else arr * (1 - cover_mix) + c * cover_mix
        arr *= self.vig
        return arr

    def finish(self, arr, t):
        arr += self.grain
        return np.clip(arr, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- frame
def subpixel(src, fx, fy):
    """Shift an RGBA sprite right/down by a fraction of a pixel (bilinear, premultiplied).
    The result keeps the same origin and is one pixel larger to the right and bottom."""
    h, w = src.shape[:2]
    a = src[..., 3:4].astype(np.float32) / 255.0
    p = np.zeros((h + 1, w + 1, 4), np.float32)
    p[:h, :w, :3] = src[..., :3] * a
    p[:h, :w, 3:] = a
    out = ((1 - fx) * (1 - fy)) * p
    out[:, 1:] += (fx * (1 - fy)) * p[:, :-1]
    out[1:, :] += ((1 - fx) * fy) * p[:-1, :]
    out[1:, 1:] += (fx * fy) * p[:-1, :-1]
    rgb = out[..., :3] / np.maximum(out[..., 3:], 1e-4)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(out[..., 3] * 255, 0, 255)]).astype(np.uint8)


def blend(dst, src, x, y, alpha):
    if alpha <= 0.003:
        return
    xi, yi = math.floor(x), math.floor(y)
    fx, fy = x - xi, y - yi
    if 0.02 < fx < 0.98 or 0.02 < fy < 0.98:
        src = subpixel(src, fx, fy)
    else:
        xi, yi = int(round(x)), int(round(y))
    x, y = xi, yi
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) * (alpha / 255.0)
    reg = dst[y0:y1, x0:x1]
    reg *= 1 - a
    reg += s[..., :3].astype(np.float32) * a


class Renderer:
    def __init__(self, args):
        self.scene = build_scene(args.lyrics, args.timings)
        self.duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.audio],
            capture_output=True, text=True, check=True).stdout.strip())
        self.bg = Background(args.cover, args.title_cover)
        self.offset = args.offset
        first = self.scene[0].t_in - self.offset     # real time of the first sung word
        # opening: title cover, dissolving into the clip well before the first sung word
        self.intro_out_end = max(COVER_DISSOLVE + 1.0, first - 2.5)
        self.intro_out_start = self.intro_out_end - COVER_DISSOLVE
        # corner label fades in once the cover's own title is gone and stays to the end
        self.label_in = (self.intro_out_end, max(self.intro_out_end + 0.5, first - 0.5))
        f = ImageFont.truetype(FONT_HAND_LIGHT, 52, layout_engine=ImageFont.Layout.RAQM)
        self.corner = []
        spr, ax, ay, adv = make_sprite(args.title, f, 52, CREAM)
        self.corner.append((spr, 80 + adv / 2 - ax, 88 - ay))
        if args.artist:
            spr, ax, ay, adv = make_sprite(args.artist, f, 52, CREAM)
            self.corner.append((spr, W - 80 - adv / 2 - ax, 88 - ay))

    def cover_mix(self, t):
        return 1 - ease_in_out((t - self.intro_out_start) / COVER_DISSOLVE)

    def zoom(self, t):
        a = self.intro_out_start
        return 1.0 + (ZOOM_END - 1.0) * ease_in_out((t - a) / max(1.0, self.duration - a))

    def frame(self, t):
        t = float(t)
        arr = self.bg.render(t, self.zoom(t), self.cover_mix(t))
        tt = t + self.offset
        la = clamp01((t - self.label_in[0]) / (self.label_in[1] - self.label_in[0]))
        for spr, x, y in self.corner:
            blend(arr, spr, x, y, 0.85 * ease_in_out(la))
        for card in self.scene:
            if tt < card.t_in - 0.2 or tt > card.t_exit + 2.0:
                continue
            for ln in card.lines:
                for w in ln.words:
                    for spr, x, y, a in word_ops(ln, w, card, tt):
                        blend(arr, spr, x, y, a)
        return self.bg.finish(arr, t)


# ----------------------------------------------------------------------------- main
def render_range(args, f0, f1, out_path):
    r = Renderer(args)
    fps = args.fps
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-maxrate", args.maxrate, "-bufsize", "18M", "-pix_fmt", "yuv420p", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(r.frame(f / fps).tobytes())
        if (f - f0) % 150 == 0:
            print(f"[{os.getpid()}] frame {f}/{f1}", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit(f"ffmpeg failed for {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--timings", default=os.path.join(HERE, "timings.json"))
    ap.add_argument("--lyrics", default=os.path.join(HERE, "lyrics.txt"))
    ap.add_argument("--cover", default=os.path.join(ASSETS, "background.webp"),
                    help="background under the lyrics: the title cover without its title")
    ap.add_argument("--title-cover", default=os.path.join(ASSETS, "cover_title.webp"),
                    help="cover with the song title, shown at the start")
    ap.add_argument("--title", default="מכות עם עצמי")
    ap.add_argument("--artist", default="")
    ap.add_argument("--out", default="lyric_video.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=21)
    ap.add_argument("--maxrate", default="9M")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--offset", type=float, default=0.0, help="shift lyrics (sec, + = earlier)")
    ap.add_argument("--stills", default="", help="comma separated times -> PNG previews")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    args = ap.parse_args()

    if args.stills:
        r = Renderer(args)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        for s in args.stills.split(","):
            t = float(s)
            Image.fromarray(r.frame(t)).save(f"{os.path.splitext(args.out)[0]}_{t:07.2f}.png")
        return

    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", args.audio], capture_output=True, text=True).stdout)
    end = args.end or dur
    f0, f1 = int(args.start * args.fps), int(math.ceil(end * args.fps))
    n = max(1, args.workers)
    bounds = [f0 + (f1 - f0) * i // n for i in range(n + 1)]
    tmpdir = os.path.splitext(os.path.abspath(args.out))[0] + "_parts"
    os.makedirs(tmpdir, exist_ok=True)
    parts = [os.path.join(tmpdir, f"part{i}.mp4") for i in range(n)]
    import multiprocessing as mp
    procs = [mp.Process(target=render_range, args=(args, bounds[i], bounds[i + 1], parts[i])) for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            sys.exit(f"worker failed with code {p.exitcode}")
    lst = os.path.join(tmpdir, "list.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{x}'\n" for x in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(args.start), "-t", str(end - args.start), "-i", args.audio,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-movflags", "+faststart", "-shortest", args.out], check=True)
    for x in parts:
        os.remove(x)
    os.remove(lst)
    os.rmdir(tmpdir)
    print("done:", args.out)


if __name__ == "__main__":
    main()
