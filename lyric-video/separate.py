"""Isolate the vocal (and drum) stems with Demucs from a local model repo.

usage: python separate.py SONG.wav DEMUCS_REPO_DIR OUT_DIR [model_name]
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main():
    song, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
    name = sys.argv[4] if len(sys.argv) > 4 else "mdx_extra_q"
    torch.set_num_threads(os.cpu_count())
    model = get_model(name, repo=__import__("pathlib").Path(repo))
    model.eval()
    wav, sr = sf.read(song, dtype="float32", always_2d=True)
    assert sr == model.samplerate, (sr, model.samplerate)
    x = torch.from_numpy(wav.T.copy())
    ref = x.mean(0)
    x = (x - ref.mean()) / ref.std()
    with torch.no_grad():
        stems = apply_model(model, x[None], shifts=1, split=True, overlap=0.25, progress=True)[0]
    stems = stems * ref.std() + ref.mean()
    os.makedirs(out, exist_ok=True)
    for s, name in zip(stems, model.sources):
        sf.write(os.path.join(out, f"{name}.wav"), s.numpy().T, sr)
        print("wrote", name)


if __name__ == "__main__":
    main()
