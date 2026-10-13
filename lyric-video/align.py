"""Word-level forced alignment of lyrics.txt to the song.

1. A first-pass transcript (transcribe.py output) gives approximate word times.
2. The exact lyric words are force-aligned with Whisper cross-attention (ctranslate2
   `align`) in overlapping <=28 s windows; each word keeps the result from the window
   where it sits farthest from a window edge.
3. Optionally, word starts are snapped to vocal onsets found on an isolated vocal stem.

usage: python align.py MODEL_DIR AUDIO FIRST_PASS.json lyrics.txt timings.json [--vocals vocals.wav]
"""
import argparse
import difflib
import json
import subprocess

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer

from lyrics_parse import normalize, parse

SR = 16000


def load(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def approx_times(words, first_pass):
    """Map lyric words onto first-pass words; interpolate the unmatched ones."""
    fp = [w for seg in first_pass for w in seg["words"]]
    a = [w.key for w in words]
    b = [normalize(w["w"]) for w in fp]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    st = [None] * len(words)
    en = [None] * len(words)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            st[blk.a + k] = fp[blk.b + k]["s"]
            en[blk.a + k] = fp[blk.b + k]["e"]
    # also accept near matches (e.g. "ודי" vs "ודאי") inside replace opcodes of equal length
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                if difflib.SequenceMatcher(a=a[i1 + k], b=b[j1 + k]).ratio() >= 0.4:
                    st[i1 + k] = fp[j1 + k]["s"]
                    en[i1 + k] = fp[j1 + k]["e"]
    known = [i for i in range(len(words)) if st[i] is not None]
    for i in range(len(words)):
        if st[i] is None:
            prev = max([k for k in known if k < i], default=None)
            nxt = min([k for k in known if k > i], default=None)
            if prev is None:
                st[i] = en[i] = st[nxt]
            elif nxt is None:
                st[i] = en[i] = en[prev]
            else:
                f = (i - prev) / (nxt - prev)
                st[i] = en[prev] + f * (st[nxt] - en[prev])
                en[i] = st[i] + 0.3
            print(f"  unmatched lyric word #{i} '{words[i].text}' -> ~{st[i]:.2f}")
    return np.array(st), np.array(en)


def plan_windows(st, en, max_len=27.0, clean_gap=0.8, break_gap=4.0):
    n = len(st)
    wins = []
    i = 0
    while i < n:
        i0 = i
        gap_before = st[i0] - en[i0 - 1] if i0 > 0 else 99
        start_clean = gap_before > clean_gap
        t0 = st[i0] - (min(0.6, gap_before * 0.5) if start_clean else 0.06)
        i1 = i0
        while i1 + 1 < n and en[i1 + 1] - t0 <= max_len and st[i1 + 1] - en[i1] < break_gap:
            i1 += 1
        gap_after = st[i1 + 1] - en[i1] if i1 + 1 < n else 99
        end_clean = gap_after > clean_gap
        t1 = en[i1] + (min(0.6, gap_after * 0.5) if end_clean else 0.06)
        wins.append(dict(i0=i0, i1=i1, t0=max(0.0, t0), t1=t1, sc=start_clean, ec=end_clean))
        if i1 == n - 1:
            break
        i = i1 + 1 if end_clean else max(i0 + 1, (i0 + i1 + 1) // 2)
    return wins


def force_align(model, tok, audio, words, wins):
    best = {}
    for w in wins:
        chunk = audio[int(w["t0"] * SR):int(w["t1"] * SR)]
        feats = model.feature_extractor(chunk)[..., :-1]
        num_frames = min(feats.shape[-1], 3000)
        enc = model.encode(pad_or_trim(feats))
        ws = words[w["i0"]:w["i1"] + 1]
        tokens = tok.encode(" " + " ".join(x.key for x in ws))
        res = model.find_alignment(tok, [tokens], enc, num_frames)[0]
        if len(res) != len(ws):
            print("  !! word count mismatch in window", w, len(res), len(ws))
            continue
        for x, r in zip(ws, res):
            s, e = w["t0"] + r["start"], w["t0"] + r["end"]
            score = min(99 if w["sc"] else s - w["t0"], 99 if w["ec"] else w["t1"] - e)
            if x.index not in best or score > best[x.index][0]:
                best[x.index] = (score, s, e, float(r["probability"]))
        print(f"  window {w['t0']:7.2f}-{w['t1']:7.2f}  words {w['i0']}-{w['i1']}")
    return best


def voiced_regions(voiced, t, min_gap=0.25, min_len=0.08):
    """Boolean frame mask -> list of (start, end) seconds, merging short gaps."""
    regs = []
    on = None
    for j, v in enumerate(voiced):
        if v and on is None:
            on = j
        elif not v and on is not None:
            regs.append([t[on], t[j]])
            on = None
    if on is not None:
        regs.append([t[on], t[-1]])
    merged = []
    for r in regs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def refine_with_vocals(words, vocals_path):
    """Fix word starts using the isolated vocal stem.

    Cross-attention DTW often lets a word that follows a pause start in the pause
    (or on the decaying tail of the previous word). Such a word is moved to the next
    voiced onset; then every start is snapped to a nearby vocal onset.
    """
    import librosa
    y = load(vocals_path, 22050)
    hop = 220  # ~10 ms
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=hop,
                                        backtrack=True, units="time")
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    t_rms = np.arange(len(rms)) * hop / 22050
    peak = np.percentile(rms_db, 98)
    voiced = rms_db > peak - 30
    regs = voiced_regions(voiced, t_rms)
    starts = np.array([r[0] for r in regs])

    def region_at(t):
        k = np.searchsorted(starts, t, side="right") - 1
        if k >= 0 and regs[k][0] <= t < regs[k][1]:
            return k
        return None

    n = len(words)
    for i, w in enumerate(words):
        nxt = words[i + 1].start if i + 1 < n else w.end + 1.0
        span_end = max(min(w.end, nxt), w.start + 0.05)
        k = region_at(w.start)
        target = None
        first_in_line = i == 0 or words[i - 1].line != w.line
        if k is None:
            # starts in silence -> next vocal onset
            k2 = np.searchsorted(starts, w.start)
            if k2 < len(regs) and regs[k2][0] < span_end + 0.6:
                target = regs[k2][0]
        elif first_in_line:
            # a line rarely starts with a word split by a pause: if the word's span holds a
            # real pause (>=120 ms below the voice threshold) the word starts after it, when
            # either the voice resumes well before the next word or only a tail precedes it
            k0 = np.searchsorted(t_rms, w.start)
            k1 = np.searchsorted(t_rms, span_end + 0.1)
            run, best = 0, None
            for j in range(k0, min(k1, len(voiced))):
                run = run + 1 if not voiced[j] else 0
                if run >= 12 and (j + 1 >= len(voiced) or voiced[j + 1]):
                    best = (j - run + 1, j + 1)
            if best is not None:
                pre = t_rms[best[0]] - w.start
                post = span_end - t_rms[min(best[1], len(t_rms) - 1)]
                if post >= 0.3 or pre < 0.35:
                    target = float(t_rms[min(best[1], len(t_rms) - 1)])
        if target is not None and target > w.start:
            print(f"  moved '{w.text}' {w.start:.2f} -> {target:.2f}")
            w.start = float(target)
            if i + 1 < n and words[i + 1].start < w.start + 0.15:
                words[i + 1].start = w.start + 0.15
    for i, w in enumerate(words):
        near = onsets[(onsets > w.start - 0.10) & (onsets < w.start + 0.12)]
        if len(near):
            w.start = float(near[np.argmin(np.abs(near - w.start))])
        if i > 0 and w.start < words[i - 1].start + 0.08:
            w.start = words[i - 1].start + 0.08
    # word ends: next start, cut at the end of the voiced region (breaths, breaks)
    for i, w in enumerate(words):
        nxt = words[i + 1].start if i + 1 < n else w.end
        w.end = min(max(w.end, w.start + 0.1), nxt) if i + 1 < n else w.end
        k = region_at(w.start + 0.02)
        if k is not None and regs[k][1] < w.end:
            w.end = max(w.start + 0.12, float(regs[k][1]))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("first_pass")
    ap.add_argument("lyrics")
    ap.add_argument("out")
    ap.add_argument("--vocals")
    a = ap.parse_args()

    cards, lines, words = parse(a.lyrics)
    st, en = approx_times(words, json.load(open(a.first_pass)))
    wins = plan_windows(st, en)
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=4)
    tok = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="he")
    audio = load(a.audio)
    best = force_align(model, tok, audio, words, wins)
    for w in words:
        if w.index in best:
            _, w.start, w.end, _ = best[w.index]
        else:
            w.start, w.end = float(st[w.index]), float(en[w.index])
    prev_sec = None
    for w in words:
        sec = cards[w.card].section
        if sec != prev_sec and st[w.index] - w.start > 0.8:
            print(f"  section start '{w.text}' {w.start:.2f} -> {st[w.index]:.2f} (first pass)")
            w.start = float(st[w.index])
        prev_sec = sec
    if a.vocals:
        refine_with_vocals(words, a.vocals)
    out = [dict(i=w.index, text=w.text, line=w.line, card=w.card,
                start=round(w.start, 3), end=round(w.end, 3),
                prob=round(best.get(w.index, (0, 0, 0, 0))[3], 3)) for w in words]
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for ln in lines:
        ws = [out[w.index] for w in ln.words]
        print(f"{ws[0]['start']:7.2f}-{ws[-1]['end']:7.2f}  " +
              " ".join(f"{x['text']}({x['start']:.2f})" for x in ws))


if __name__ == "__main__":
    main()
