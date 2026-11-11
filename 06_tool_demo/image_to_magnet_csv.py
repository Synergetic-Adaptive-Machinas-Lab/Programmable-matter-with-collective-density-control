import argparse
import sys

import numpy as np
from PIL import Image

ROWS = 32
COLS = 32
D_MM = 20.0
DX_MM = D_MM
DY_MM = D_MM * (3 ** 0.5) / 2

X_MIN = -D_MM / 2
X_MAX = (COLS - 1 + 0.5) * DX_MM + D_MM / 2
Y_MIN = -D_MM / 2
Y_MAX = (ROWS - 1) * DY_MM + D_MM / 2

CROP_ASPECT = (X_MAX - X_MIN) / (Y_MAX - Y_MIN)

def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(np.round(gray).astype(int), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    thresh = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            thresh = t
    return thresh

def center_crop_to_aspect(img: Image.Image, aspect: float) -> Image.Image:
    w, h = img.size
    if w / h >= aspect:
        new_w = h * aspect
        new_h = h
    else:
        new_w = w
        new_h = w / aspect
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    return img.crop((left, top, left + new_w, top + new_h))

def extract_gray32(img: Image.Image) -> np.ndarray:
    cropped = center_crop_to_aspect(img.convert("RGB"), CROP_ASPECT)
    small = cropped.resize((COLS, ROWS), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return gray

def build_state(gray: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    if invert:
        on = gray >= threshold
    else:
        on = gray < threshold
    return on.astype(np.int8)

def build_csv(state: np.ndarray, polarity: int) -> str:

    lines = []
    for col in range(COLS):
        vals = []
        for row in range(ROWS - 1, -1, -1):
            vals.append(str(int(state[row, col]) * polarity * 7))
        lines.append(",".join(vals))
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Convert an image to a 1024-magnet SAM LAB CSV.")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: <image>_magnet.csv)")
    ap.add_argument("--threshold", type=int, default=None, help="Manual threshold 0-255 (default: Otsu auto)")
    ap.add_argument("--invert", action="store_true", help="Light pixels -> on, instead of dark pixels -> on")
    ap.add_argument("--polarity", choices=["north", "south"], default="north", help="Polarity for 'on' magnets")
    ap.add_argument("--preview", action="store_true", help="Print an ASCII preview of the 32x32 pattern")
    args = ap.parse_args()

    img = Image.open(args.image)
    gray = extract_gray32(img)
    threshold = args.threshold if args.threshold is not None else otsu_threshold(gray)
    state = build_state(gray, threshold, args.invert)
    polarity = 1 if args.polarity == "north" else -1

    csv_text = build_csv(state, polarity)
    out_path = args.output or (args.image.rsplit(".", 1)[0] + "_magnet.csv")
    with open(out_path, "w") as f:
        f.write(csv_text + "\n")

    n_on = int(state.sum())
    print(f"threshold={threshold} (auto={'yes' if args.threshold is None else 'no'})  "
          f"magnets_on={n_on}/{ROWS*COLS}  polarity={args.polarity}")
    print(f"wrote {out_path}")

    if args.preview:
        for row in range(ROWS):
            print("".join("#" if state[row, col] else "." for col in range(COLS)))

if __name__ == "__main__":
    sys.exit(main())
