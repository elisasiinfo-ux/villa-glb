"""Hebrew lyric-video renderer.

Reads lyrics.txt (layout/effect tags) + timings.json (word times from align.py),
draws the cover image with camera motion, grading, grain and beat punches, and
writes each sung word on screen at the moment it is sung.

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
RED = (255, 70, 40)

SIZE = {"normal": 112, "big": 158, "huge": 250}
SIZE_GRUNGE = {"normal": 96, "big": 132, "huge": 205}
SIZE_PRAYER = {"normal": 118, "big": 150, "huge": 200}

EXIT_DUR = {"default": 0.28, "crash": 1.1, "wind": 1.1, "sink": 0.9, "fall": 0.7,
            "glitch": 0.4, "prayer": 1.4, "angel": 0.6, "devil": 0.45}


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


def ease_out_back(p, s=1.9):
    p = clamp01(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def ease_out_bounce(p):
    p = clamp01(p)
    n, d = 7.5625, 2.75
    if p < 1 / d:
        return n * p * p
    if p < 2 / d:
        p -= 1.5 / d
        return n * p * p + 0.75
    if p < 2.5 / d:
        p -= 2.25 / d
        return n * p * p + 0.9375
    p -= 2.625 / d
    return n * p * p + 0.984375


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
    Returns (array, anchor_x, anchor_y): anchor = centre of the advance on the baseline."""
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
    a = halo
    rgb = np.zeros((h, w, 3), np.float32)
    a = shadow + a * (1 - shadow)
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
    if abs(s - 1) < 0.004:
        return arr, ax, ay
    h, w = arr.shape[:2]
    nw, nh = max(2, int(w * s)), max(2, int(h * s))
    im = Image.fromarray(arr).resize((nw, nh), Image.BILINEAR)
    return np.asarray(im), ax * nw / w, ay * nh / h


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
            if kind == "prayer":
                li.angle = 0.0
            else:
                li.angle = (hash01(ln.index, 11) - 0.55) * (5.0 if sc == "huge" else 7.0)
            path = {"hand": FONT_HAND, "grunge": FONT_GRUNGE, "prayer": FONT_PRAYER}[kind]
            size = {"hand": SIZE, "grunge": SIZE_GRUNGE, "prayer": SIZE_PRAYER}[kind][sc]
            f = font(path, size)
            space = f.getlength(" ") * {"hand": 0.55, "grunge": 1.7, "prayer": 1.1}[kind]
            total = sum(f.getlength(w.text.replace(",", ""), direction="rtl") for w in ln.words) + space * (len(ln.words) - 1)
            if total > 1720:                       # auto-shrink very long lines
                size = int(size * 1720 / total)
                f = font(path, size)
                space = f.getlength(" ") * {"hand": 0.55, "grunge": 1.7, "prayer": 1.1}[kind]
            fill = GOLD if kind == "prayer" else CREAM
            glow = {"angel": (255, 236, 200), "devil": (120, 0, 0)}.get(li.entry)
            if li.entry == "devil":
                fill = (235, 38, 24)
            elif li.entry == "angel":
                fill = (255, 255, 252)
            if kind == "prayer":
                glow = (255, 170, 60)
            for k, w in enumerate(ln.words):
                spr, ax, ay, adv = make_sprite(w.text.replace(",", ""), f, size, fill, glow, li.angle)
                li.words.append(WordItem(w.text, w.start, w.end, spr, ax, ay, k=k))
                li.words[-1].adv = adv
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
                w.x = cx + lx * math.cos(th)
                w.y = cy - lx * math.sin(th)
                x -= w.adv + ln.space
            ln.cx, ln.cy = cx, cy
            y += hgt + gap


def timing(scene):
    for card in scene:
        card.t_in = min(w.start for ln in card.lines for w in ln.words)
    for i, card in enumerate(scene):
        ws = [w for ln in card.lines for w in ln.words]
        last = max(ws, key=lambda w: w.start)
        nxt = scene[i + 1].t_in if i + 1 < len(scene) else 1e9
        hold = 2.2 if card.lines[0].font_kind != "prayer" else 3.5
        t = min(nxt - 0.12, max(last.end, last.start + 0.45) + hold)
        card.t_exit = max(t, last.start + 0.40)


# ----------------------------------------------------------------------------- word animation
def word_ops(ln, w, card, t, fx_state):
    """Return a list of (sprite, x, y, alpha) draw ops for one word at time t."""
    t0 = w.start - 0.05
    if t < t0:
        return []
    k = w.k
    te = card.t_exit + 0.03 * k
    exit_dur = EXIT_DUR.get(ln.exit, EXIT_DUR["default"])
    if t > te + exit_dur:
        return []
    arr, ax, ay = w.sprite, w.ax, w.ay
    x, y, alpha = w.x, w.y, 1.0
    dt = t - t0
    e = ln.entry

    # idle float while on screen
    y += math.sin((t + ln.seed * 10) * 1.3) * 3.0
    life = t - card.t_in
    s = 1.0 + 0.012 * min(life, 4.0) / 4.0

    if e == "write":
        dur = min(0.5, max(0.18, 0.8 * (w.end - w.start)))
        p = ease_out_cubic(dt / dur)
        arr = wipe_rtl(arr, p)
        y += 16 * (1 - p)
    elif e == "pop":
        p = dt / 0.34
        s *= 0.55 + 0.45 * ease_out_back(p)
        alpha = clamp01(dt / 0.12)
        if p < 1:
            arr = blur(arr, 6 * (1 - ease_out_cubic(p)))
    elif e in ("stamp", "crash"):
        p = dt / 0.16
        s *= 1 + 1.5 * (1 - ease_in_cubic(p))
        alpha = clamp01(dt / 0.07)
        if p < 1:
            arr = blur(arr, 5 * (1 - p))
    elif e == "slam":
        p = dt / 0.2
        y -= 280 * (1 - ease_in_cubic(p))
        s *= 1 + 0.35 * (1 - ease_in_cubic(p))
        alpha = clamp01(dt / 0.08)
        if p < 1:
            arr = blur(arr, 8 * (1 - p))
    elif e == "fall":
        p = dt / 0.65
        y -= 330 * (1 - ease_out_bounce(p))
        alpha = clamp01(dt / 0.1)
    elif e == "glitch":
        alpha = 1.0 if dt > 0.45 or hash01(int(t * 30), 7) > 0.3 else 0.25
        if dt < 0.5:
            arr = glitch(arr, 1 - dt / 0.5, t)
        elif hash01(int(t * 12), 3) > 0.9:
            arr = glitch(arr, 0.35, t)
    elif e == "angel":
        p = dt / 0.7
        alpha = ease_out_cubic(p)
        y += 40 * (1 - ease_out_cubic(p))
        if p < 1:
            arr = blur(arr, 10 * (1 - p))
    elif e == "devil":
        alpha = 1.0 if dt > 0.45 or hash01(int(t * 24), 9) > 0.45 else 0.0
        x += (hash01(int(t * 60), 1) - 0.5) * 10 * clamp01(1 - dt / 0.5)
    elif e == "prayer":
        p = dt / 1.1
        alpha = ease_in_out(p)
        y += 26 * (1 - ease_out_cubic(p))
        if p < 1:
            arr = blur(arr, 12 * (1 - ease_out_cubic(p)))
    elif e == "pulse":
        p = dt / 0.3
        s *= 0.6 + 0.4 * ease_out_back(p)
        alpha = clamp01(dt / 0.1)
        s *= 1 + 0.07 * fx_state["beat"]

    # exits
    if t > te:
        q = (t - te) / exit_dur
        x_ = ln.exit
        if x_ == "crash":
            return crash_ops(w, arr, ax, ay, x, y, s, q)
        if x_ == "wind":
            arr, shift = wind(arr, q, w)
            ax = ax + shift
            alpha *= 1 - ease_in_cubic(q)
        elif x_ == "sink":
            y += 150 * ease_in_cubic(q)
            arr = blur(arr, 16 * q)
            alpha *= 1 - ease_in_out(q)
        elif x_ == "fall":
            y += 900 * q * q
            alpha *= 1 - ease_in_cubic(q)
        elif x_ == "glitch":
            arr = glitch(arr, 0.6 + q, t)
            alpha *= 0 if q > 0.8 else 1 if hash01(int(t * 30), 5) > 0.3 else 0.3
        elif x_ == "prayer":
            y -= 30 * ease_in_out(q)
            arr = blur(arr, 10 * q)
            alpha *= 1 - ease_in_out(q)
        elif x_ == "angel":
            y -= 60 * ease_in_cubic(q)
            arr = blur(arr, 8 * q)
            alpha *= 1 - q
        elif x_ == "devil":
            alpha *= 0 if hash01(int(t * 30), 2) > 1 - q else 1 - q
        else:
            y -= 22 * ease_in_cubic(q)
            if ln.entry in ("stamp", "slam"):
                s *= 1 + 0.15 * q
            arr = blur(arr, 6 * q)
            alpha *= 1 - ease_in_out(q)

    arr, ax2, ay2 = scale_sprite(arr, ax, ay, s)
    return [(arr, x - ax2, y - ay2, alpha)]


def glitch(arr, amount, t):
    """RGB-split + horizontal slice tearing."""
    amount = max(0.0, min(1.5, amount))
    h, w = arr.shape[:2]
    d = int(16 * amount) + 2
    a = arr[..., 3].astype(np.float32) / 255
    al = np.roll(a, -d, axis=1)
    ar = np.roll(a, d, axis=1)
    rgb = arr[..., :3].astype(np.float32)
    red = np.array([255, 30, 60], np.float32)
    cyan = np.array([30, 230, 255], np.float32)
    wl = al * (1 - a)
    wr = ar * (1 - a) * (1 - al)
    out_a = a + wl + wr
    col = (rgb * a[..., None] + red * wl[..., None] + cyan * wr[..., None]) / np.maximum(out_a, 1e-4)[..., None]
    out = np.dstack([np.clip(col, 0, 255), np.clip(out_a * 255, 0, 255)]).astype(np.uint8)
    seed = int(t * 20)
    for i in range(int(3 + 5 * amount)):
        y0 = int(hash01(seed, i, 1) * h)
        hh = int(6 + hash01(seed, i, 2) * h * 0.1)
        sh = int((hash01(seed, i, 3) - 0.5) * 90 * amount)
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], sh, axis=1)
    return out


def wind(arr, q, w):
    """Thin strips of the word blow away to the left and dissolve.
    Returns (array, shift): the original content now starts `shift` px to the right."""
    if q <= 0:
        return arr, 0
    h, wd = arr.shape[:2]
    extra = int(750 * q * q) + 2
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


def crash_ops(w, arr, ax, ay, x, y, s, q):
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
                                   vx=(cxp - w.ax) * 1.6 + (r1 - 0.5) * 500,
                                   vy=-250 - r2 * 450, rot=(r3 - 0.5) * 540))
        w.shards = shards
    T = q * 1.1
    ops = []
    for sh in w.shards:
        px = x + (sh["cx"] - w.ax) * s + sh["vx"] * T
        py = y + (sh["cy"] - w.ay) * s + sh["vy"] * T + 0.5 * 2600 * T * T
        im = sh["img"].rotate(sh["rot"] * T, resample=Image.BILINEAR, expand=True)
        a = np.asarray(im)
        ops.append((a, px - a.shape[1] / 2, py - a.shape[0] / 2, 1 - ease_in_cubic(q)))
    return ops


# ----------------------------------------------------------------------------- background
class Background:
    def __init__(self, cover_path, audio_path, scene, duration):
        src = Image.open(cover_path).convert("RGB")
        self.S = 1.5
        big = src.resize((int(src.width * self.S), int(src.height * self.S)), Image.LANCZOS)
        a = np.asarray(big).astype(np.float32)
        # punchier contrast for the main look
        a = np.clip((a - 128) * 1.08 + 128, 0, 255)
        self.main = Image.fromarray(a.astype(np.uint8))
        # prayer look: desaturated, warm, darker, soft
        lum = a.mean(axis=2, keepdims=True)
        warm = lum * np.array([1.05, 0.86, 0.62])[None, None]
        pr = np.clip(warm * 0.75 + a * 0.12, 0, 255).astype(np.uint8)
        self.prayer = Image.fromarray(pr).filter(ImageFilter.GaussianBlur(4))
        self.bw, self.bh = big.size
        self.face = (760 * self.S, 420 * self.S)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vig = (1 - 0.62 * np.clip(r / 1.25, 0, 1) ** 2.2)[..., None].astype(np.float32)
        rng = np.random.default_rng(7)
        self.grain = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(12)]
        ly, lx = np.mgrid[0:H * 2:2, 0:W * 3:2].astype(np.float32)
        blob = np.exp(-(((lx - W * 0.9) / (W * 0.35)) ** 2 + ((ly - H * 0.6) / (H * 0.5)) ** 2))
        blob += 0.7 * np.exp(-(((lx - W * 2.1) / (W * 0.3)) ** 2 + ((ly - H * 1.3) / (H * 0.45)) ** 2))
        small = (blob[..., None] * np.array([255, 150, 70], np.float32)).astype(np.float32)
        self.leak = np.asarray(Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize(
            (W * 3 // 2 + W // 2, H * 3 // 2), Image.BILINEAR)).astype(np.float32)
        self.duration = duration
        self.analyse_audio(audio_path)
        self.build_looks(scene)
        self.particles = [dict(x=hash01(i, 1) * W, y=hash01(i, 2) * H, r=1.5 + hash01(i, 3) * 4.5,
                               sp=10 + hash01(i, 4) * 30, ph=hash01(i, 5) * 6.28) for i in range(70)]
        self.dot = {}

    def analyse_audio(self, path):
        import librosa
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "22050",
                              "-f", "f32le", "-"], capture_output=True, check=True).stdout
        y = np.frombuffer(raw, np.float32)
        tempo, beats = librosa.beat.beat_track(y=y, sr=22050, hop_length=256, units="time")
        self.beats = np.asarray(beats)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        self.rms_t = np.arange(len(rms)) * 512 / 22050
        self.rms = rms / (np.percentile(rms, 95) + 1e-6)

    def beat_env(self, t):
        i = np.searchsorted(self.beats, t) - 1
        if i < 0:
            return 0.0
        return math.exp(-(t - self.beats[i]) / 0.11)

    def energy(self, t):
        return float(np.interp(t, self.rms_t, self.rms))

    def build_looks(self, scene):
        """Keyframes of (time, brightness, punch, zoom_bias, prayer_mix) by song section."""
        sec = {}
        for c in scene:
            ws = [w for ln in c.lines for w in ln.words]
            a, b = min(w.start for w in ws), max(w.end for w in ws)
            s = sec.setdefault(c.section, [a, b])
            s[0], s[1] = min(s[0], a), max(s[1], b)
        order = sorted(sec.items(), key=lambda kv: kv[1][0])
        # (brightness, beat punch, zoom bias, prayer mix, light leak)
        look = {"פתיחה": (0.80, 0.35, 0.00, 0, 0.10), "בית": (0.74, 0.30, 0.02, 0, 0.05),
                "פזמון": (0.95, 1.00, 0.05, 0, 0.22), "סיום": (0.97, 1.10, 0.06, 0, 0.25),
                "תפילה": (0.80, 0.00, 0.12, 1, 0.0)}
        keys = [(0.0, 0.0, 0.0, -0.03, 0.0, 0.0), (3.0, 0.62, 0.0, -0.02, 0.0, 0.12)]
        prev_end = 0.0
        for name, (a, b) in order:
            kind = next((k for k in look if name.startswith(k)), "בית")
            br, pu, zb, pm, lk = look[kind]
            if prev_end > 0 and a - prev_end > 6:   # instrumental break before this section
                keys.append((prev_end + 1.0, 0.86, 0.85, 0.08, 0.0, 0.45))
                keys.append((a - 1.2, 0.80, 0.6, 0.12, 0.0, 0.35))
            if kind == "תפילה":
                keys.append((max(a - 0.6, prev_end + 0.5), br, pu, zb, pm, lk))
            else:
                keys.append((a - 0.4, br, pu, zb, pm, lk))
            keys.append((b, br, pu, zb, pm, lk))
            prev_end = b
        end = self.duration
        keys.append((min(prev_end + 4.0, end - 3.0), 0.7, 0.0, 0.16, 1.0, 0.0))
        keys.append((end - 0.6, 0.0, 0.0, 0.18, 1.0, 0.0))
        keys.append((end + 1, 0.0, 0.0, 0.18, 1.0, 0.0))
        keys.sort()
        self.keys = np.array(keys, np.float32)
        self.sections = order

    def look(self, t):
        k = self.keys
        return [float(np.interp(t, k[:, 0], k[:, i])) for i in range(1, 6)]

    def render(self, t, shake, flash):
        bright, punch, zbias, pmix, leak = self.look(t)
        be = self.beat_env(t) * punch
        z = 1.06 + zbias + 0.025 * math.sin(t * 2 * math.pi / 47) + 0.018 * be
        cx = self.face[0] * 0.25 + self.bw / 2 * 0.75 + 70 * math.sin(t * 2 * math.pi / 61)
        cy = 700 + 35 * math.sin(t * 2 * math.pi / 43 + 1.0)
        vw = self.bw / z
        vh = vw * H / W
        cx = min(max(cx, vw / 2), self.bw - vw / 2)
        cy = min(max(cy, vh / 2), self.bh - vh / 2)
        sc = vw / W
        data = (sc, 0, cx - vw / 2 - shake[0] * sc, 0, sc, cy - vh / 2 - shake[1] * sc)
        img = self.main.transform((W, H), Image.AFFINE, data, resample=Image.BILINEAR) if pmix < 0.999 else None
        arr = np.asarray(img).astype(np.float32) if img is not None else None
        if pmix > 0.001:
            p = np.asarray(self.prayer.transform((W, H), Image.AFFINE, data, resample=Image.BILINEAR)).astype(np.float32)
            arr = p if arr is None else arr * (1 - pmix) + p * pmix
        gain = bright * (1 + 0.10 * be)
        arr *= self.vig * gain
        if flash > 0.002:
            arr += np.array([255, 225, 190], np.float32) * flash
        if leak > 0.01:
            # warm light leak drifting across the frame
            lx = int((0.5 + 0.5 * math.sin(t * 0.21)) * (self.leak.shape[1] - W))
            ly = int((0.5 + 0.5 * math.sin(t * 0.13 + 2)) * (self.leak.shape[0] - H))
            arr += self.leak[ly:ly + H, lx:lx + W] * (leak * (0.75 + 0.25 * math.sin(t * 1.7)))
        if pmix > 0.05:
            self.draw_particles(arr, t, pmix * min(1.0, bright * 1.6))
        return arr

    def draw_particles(self, arr, t, amount):
        for i, p in enumerate(self.particles):
            r = p["r"]
            key = round(r * 2) / 2
            if key not in self.dot:
                s = int(key * 6) + 3
                yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - (s - 1) / 2
                self.dot[key] = np.exp(-(xx ** 2 + yy ** 2) / (2 * (key * 0.9) ** 2))[..., None]
            d = self.dot[key]
            x = (p["x"] + 25 * math.sin(t * 0.3 + p["ph"])) % W
            y = (p["y"] - p["sp"] * t) % H
            fl = 0.55 + 0.45 * math.sin(t * 2.1 + p["ph"] * 3)
            s = d.shape[0]
            x0, y0 = int(x - s / 2), int(y - s / 2)
            if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
                continue
            arr[y0:y0 + s, x0:x0 + s] += d * np.array([255, 200, 120], np.float32) * (0.7 * fl * amount)

    def finish(self, arr, t):
        g = self.grain[int(t * 12) % len(self.grain)]
        g = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)[..., None]
        arr += g * 5.0
        return np.clip(arr, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- frame
def blend(dst, src, x, y, alpha):
    if alpha <= 0.003:
        return
    x, y = int(round(x)), int(round(y))
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
        self.bg = Background(args.cover, args.audio, self.scene, self.duration)
        self.offset = args.offset
        self.title = args.title
        self.artist = args.artist
        self.impacts = []
        for card in self.scene:
            for ln in card.lines:
                for w in ln.words:
                    if ln.entry in ("stamp", "crash"):
                        self.impacts.append((w.start - 0.05 + 0.16, 22, 0.32))
                    elif ln.entry == "slam":
                        self.impacts.append((w.start - 0.05 + 0.20, 15, 0.22))
        prev = None
        for card in self.scene:
            if card.section != prev and card.section.startswith(("פזמון", "סיום")):
                self.impacts.append((card.t_in, 7, 0.22))   # chorus hits
            prev = card.section
        self.build_titles()

    def build_titles(self):
        f = ImageFont.truetype(FONT_HAND_LIGHT, 52, layout_engine=ImageFont.Layout.RAQM)
        self.corner = []
        spr, ax, ay, adv = make_sprite(self.title, f, 52, CREAM)
        self.corner.append((spr, 80 + adv / 2 - ax, 88 - ay))
        if self.artist:
            spr, ax, ay, adv = make_sprite(self.artist, f, 52, CREAM)
            self.corner.append((spr, W - 80 - adv / 2 - ax, 88 - ay))
        fb = ImageFont.truetype(FONT_HAND, 165, layout_engine=ImageFont.Layout.RAQM)
        self.big_title = make_sprite(self.title, fb, 165, CREAM)
        if self.artist:
            fa = ImageFont.truetype(FONT_HAND_LIGHT, 92, layout_engine=ImageFont.Layout.RAQM)
            self.big_artist = make_sprite(self.artist, fa, 92, CREAM)
        else:
            self.big_artist = None

    def shake_flash(self, t):
        sx = sy = fl = 0.0
        for ti, amp, f in self.impacts:
            d = t - ti
            if 0 <= d < 0.4:
                k = math.exp(-d / 0.09)
                sx += amp * k * math.sin(d * 95 + ti)
                sy += amp * k * math.cos(d * 80 + ti * 2)
                fl += f * math.exp(-d / 0.07)
        return (sx, sy), min(fl, 0.5)

    def title_card(self, frame, t, t_in, t_out):
        if t < t_in or t > t_out + 0.6:
            return
        spr, ax, ay, adv = self.big_title
        p = (t - t_in) / 1.6
        arr = wipe_rtl(spr, ease_in_out(p))
        alpha = 1.0 if t < t_out else 1 - ease_in_out((t - t_out) / 0.6)
        if t > t_out:
            arr = blur(arr, 8 * (t - t_out) / 0.6)
        blend(frame, arr, W / 2 - ax, 500 - ay, alpha)
        if self.big_artist is not None:
            spr, ax, ay, adv = self.big_artist
            a2 = clamp01((t - t_in - 1.2) / 0.8) * alpha
            blend(frame, spr, W / 2 - ax, 680 - ay, a2)

    def frame(self, t):
        shake, flash = self.shake_flash(t)
        arr = self.bg.render(t, shake, flash)
        tt = t + self.offset
        fx_state = {"beat": self.bg.beat_env(t)}
        first = self.scene[0].t_in
        # intro title card + closing title
        self.title_card(arr, t, 1.2, first - 1.3)
        last_card = self.scene[-1]
        end_in = last_card.t_exit + EXIT_DUR["prayer"] + 0.8
        if end_in < self.duration - 3:
            self.title_card(arr, t, end_in, self.duration - 1.6)
        # corner labels (hidden during the title cards)
        ca = clamp01((t - (first - 1.0)) / 1.0) * (1 - clamp01((t - (last_card.t_exit - 1)) / 1.5))
        for spr, x, y in self.corner:
            blend(arr, spr, x + shake[0], y + shake[1], 0.85 * ca)
        for card in self.scene:
            if tt < card.t_in - 0.2 or tt > card.t_exit + 1.6:
                continue
            for ln in card.lines:
                for w in ln.words:
                    for spr, x, y, a in word_ops(ln, w, card, tt, fx_state):
                        blend(arr, spr, x + shake[0], y + shake[1], a)
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
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--timings", default=os.path.join(HERE, "timings.json"))
    ap.add_argument("--lyrics", default=os.path.join(HERE, "lyrics.txt"))
    ap.add_argument("--cover", default=os.path.join(ASSETS, "cover.webp"))
    ap.add_argument("--title", default="משהו חייב להשתנות")
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
