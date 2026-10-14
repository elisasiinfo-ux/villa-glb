"""Build the under-lyrics background from the title cover by erasing its painted title.

cover_title.webp and cover.webp share the same background composition (orange wall,
splatter, scratches), but the singer is placed differently in the two. Using the cover's
own singer for the whole clip avoids a double image when the cover dissolves into the
clip. The title letters (with their drop shadows) are masked and filled with the wall
texture of cover.webp; next to the singer's face, where cover.webp has its own beard and
ear, the texture is borrowed from a little further right so no second beard edge appears.

usage: python make_clean_background.py assets/cover_title.webp assets/cover.webp assets/background.webp
"""
import sys

import cv2
import numpy as np
from PIL import Image

# polygons around the painted title (1536x1024 cover coordinates). They stop at the
# singer's beard (right) and follow the edge of his forearm (left), so he is never touched.
TITLE_AREAS = [
    [(908, 30), (1536, 30), (1536, 680), (908, 680)],
    [(0, 90), (495, 90), (495, 466), (466, 470), (466, 520), (461, 560), (445, 620), (432, 660), (0, 660)],
]
# where cover.webp holds its own ear/beard edge next to the right title letter
SINGER_EDGE = (895, 180, 965, 480)


def cream(img):
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return (hsv[..., 2] > 150) & (hsv[..., 1] < 110)


def lum(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


def main():
    title_path, texture_path, out_path = sys.argv[1:4]
    B = np.asarray(Image.open(title_path).convert("RGB")).astype(np.float32)
    A = np.asarray(Image.open(texture_path).convert("RGB")).astype(np.float32)
    assert A.shape == B.shape
    box8 = np.zeros(B.shape[:2], np.uint8)
    cv2.fillPoly(box8, [np.array(p, np.int32) for p in TITLE_AREAS], 1)
    box = box8.astype(bool)
    # soft limit that fades to zero at the polygon edge, so blur never leaks onto the singer
    limit = cv2.GaussianBlur(cv2.erode(box8, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
    # letter bodies, with the gaps between strokes closed (they hold the letters' shadow)
    m = cv2.morphologyEx((cream(B) & box).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    body = np.isin(lab, [i for i in range(1, n) if st[i][4] > 1500]).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    mask = cv2.dilate(body, np.ones((25, 25), np.uint8))
    # thin drips/edges (cream only in the cover) and drop shadows (darker only in the cover)
    near = cv2.dilate(body, np.ones((161, 161), np.uint8)).astype(bool) & box
    extra = (cream(B) & ~cream(A)) | (lum(A) - lum(B) > 12)
    mask = np.maximum(mask, cv2.dilate((extra & near).astype(np.uint8), np.ones((9, 9), np.uint8)))
    mask = (mask.astype(bool) & box).astype(np.float32)
    a = (cv2.GaussianBlur(mask, (0, 0), 4) * limit)[..., None]
    x0, y0, x1, y1 = SINGER_EDGE
    # texture source: cover.webp, but beside the face take its wall from 60 px further right
    z = np.zeros(B.shape[:2], np.float32)
    z[y0:y1, x0:x1] = 1
    z = cv2.GaussianBlur(z, (0, 0), 6)[..., None]
    fill = A * (1 - z) + np.roll(A, -60, axis=1) * z
    clean = (B * (1 - a) + fill * a).clip(0, 255).astype(np.uint8)
    # slivers of letter left where a letter touched the singer: inpaint the pure letter
    # cream (paler and less saturated than skin) along the polygon edge only
    hsv = cv2.cvtColor(clean, cv2.COLOR_RGB2HSV)
    band = cv2.dilate(box8, np.ones((31, 31), np.uint8)) & (1 - cv2.erode(box8, np.ones((15, 15), np.uint8)))
    sliver = ((hsv[..., 2] > 175) & (hsv[..., 1] < 80) & band.astype(bool)).astype(np.uint8)
    clean = cv2.inpaint(clean, cv2.dilate(sliver, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)
    Image.fromarray(clean).save(out_path, quality=95)
    print("masked fraction", float(a.mean()))


if __name__ == "__main__":
    main()
