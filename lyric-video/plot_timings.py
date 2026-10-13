"""Diagnostic: vocal loudness curve with word start markers, one row per time chunk.

usage: python plot_timings.py vocals.wav timings.json out.png [t0 t1]
"""
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = __file__.rsplit("/", 1)[0] + "/assets/fonts/FrankRuhlLibre-Hebrew-500.ttf"


def main():
    vocals, timings, out = sys.argv[1:4]
    t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    sr = 16000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", vocals, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, np.float32)
    t1 = float(sys.argv[5]) if len(sys.argv) > 5 else len(y) / sr
    hop = 160
    frames = len(y) // hop
    rms = np.sqrt(np.mean(y[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-6)
    top = np.percentile(db, 99)
    words = json.load(open(timings))
    chunk = 12.0
    rows = int(np.ceil((t1 - t0) / chunk))
    rw, rh = 2400, 230
    img = Image.new("RGB", (rw, rh * rows), (20, 20, 24))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 22, layout_engine=ImageFont.Layout.RAQM)
    fs = ImageFont.truetype(FONT, 16, layout_engine=ImageFont.Layout.RAQM)
    for r in range(rows):
        a = t0 + r * chunk
        oy = r * rh
        px = lambda t: (t - a) / chunk * rw
        k0, k1 = int(a * 100), int(min(a + chunk, t1) * 100)
        seg = db[k0:k1]
        pts = [(px(a + i / 100), oy + rh - 20 - max(0, (v - (top - 45))) / 45 * (rh - 70)) for i, v in enumerate(seg)]
        if len(pts) > 1:
            d.line(pts, fill=(90, 200, 255), width=2)
        for s in range(int(a), int(a + chunk) + 1):
            d.line([(px(s), oy + rh - 18), (px(s), oy + rh - 8)], fill=(120, 120, 120))
            d.text((px(s) + 2, oy + rh - 20), f"{s}", font=fs, fill=(140, 140, 140))
        for i, w in enumerate(words):
            if a <= w["start"] < a + chunk:
                x = px(w["start"])
                d.line([(x, oy + 30), (x, oy + rh - 20)], fill=(255, 120, 60), width=2)
                xe = px(min(w["end"], a + chunk))
                d.line([(x, oy + 32 + (i % 2) * 22), (xe, oy + 32 + (i % 2) * 22)], fill=(255, 200, 80), width=3)
                d.text((x + 3, oy + 4 + (i % 2) * 22), w["text"], font=f, fill=(255, 255, 255), direction="rtl", anchor="la")
    img.save(out)


if __name__ == "__main__":
    main()
