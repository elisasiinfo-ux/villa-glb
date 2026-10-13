"""First-pass transcription with word timestamps (to learn the sung structure)."""
import json
import subprocess
import sys
import time

import numpy as np
from faster_whisper import WhisperModel


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


model_dir, audio, out = sys.argv[1], sys.argv[2], sys.argv[3]
t = time.time()
m = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
segs, info = m.transcribe(load(audio), language="he", beam_size=5, word_timestamps=True,
                          vad_filter=False, condition_on_previous_text=False)
res = []
for s in segs:
    res.append(dict(start=s.start, end=s.end, text=s.text,
                    words=[dict(w=w.word, s=w.start, e=w.end, p=w.probability) for w in s.words]))
    print(f"[{s.start:7.2f}-{s.end:7.2f}] {s.text}", flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("time", time.time() - t)
