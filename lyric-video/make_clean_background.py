"""Build the under-lyrics background from the title cover by erasing its painted title.

cover_title.webp and cover.webp share the exact same background texture (they match
pixel-for-pixel outside the singer), but the singer differs between them. Using the
cover's own singer for the whole clip avoids a double image when the cover dissolves
into the clip. The title letters are masked and filled with the texture of cover.webp.

usage: python make_clean_background.py assets/cover_title.webp assets/cover.webp assets/background.webp
"""
import sys

import cv2
import numpy as np
from PIL import Image

# rough boxes around the painted title (x0, y0, x1, y1 in the 1536x1024 cover); the
# singer sits between them and is never touched
TITLE_BOXES = [(905, 30, 1536, 680), (0, 90, 495, 660)]


def cream(img):
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return (hsv[..., 2] > 150) & (hsv[..., 1] < 110)


def main():
    title_path, texture_path, out_path = sys.argv[1:4]
    B = np.asarray(Image.open(title_path).convert("RGB")).astype(np.float32)
    A = np.asarray(Image.open(texture_path).convert("RGB")).astype(np.float32)
    assert A.shape == B.shape
    box = np.zeros(B.shape[:2], bool)
    for x0, y0, x1, y1 in TITLE_BOXES:
        box[y0:y1, x0:x1] = True
    # big letter bodies
    m = cv2.morphologyEx((cream(B) & box).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    body = np.isin(lab, [i for i in range(1, n) if st[i][4] > 1500]).astype(np.uint8)
    mask = cv2.dilate(body, np.ones((25, 25), np.uint8))
    # thin drips/edges: cream in the cover but not in the texture source, near the title
    near = cv2.dilate(body, np.ones((161, 161), np.uint8)).astype(bool) & box
    extra = (cream(B) & ~cream(A) & near).astype(np.uint8)
    mask = np.maximum(mask, cv2.dilate(extra, np.ones((9, 9), np.uint8)))
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 4)[..., None]
    clean = B * (1 - a) + A * a
    Image.fromarray(clean.clip(0, 255).astype(np.uint8)).save(out_path, quality=95)
    print("masked fraction", float(a.mean()))


if __name__ == "__main__":
    main()
