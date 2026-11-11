import argparse
import csv
import glob
import os
import shutil
import subprocess

ROWS = COLS = 32


def read_frame(path):
    with open(path) as f:
        rows = [[int(float(v)) for v in line if v != ""] for line in csv.reader(f)]
    if len(rows) != ROWS or any(len(r) != COLS for r in rows):
        return None
    cells = {}
    for i in range(ROWS):
        for j in range(COLS):
            v = rows[i][j]
            if v != 0:
                cells[(ROWS - 1 - j, i)] = v
    return cells


def render(csv_dir, out, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    pitch = 20.0
    dy = pitch * np.sqrt(3) / 2
    rad = pitch / np.sqrt(3)
    files = []
    for fpath in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        try:
            if read_frame(fpath) is not None:
                files.append(fpath)
        except ValueError:
            pass
    png = os.path.join(out, "png")
    os.makedirs(png, exist_ok=True)

    def center(r, c):
        return ((c + 0.5 * (r & 1)) * pitch, (ROWS - 1 - r) * dy)

    base = [RegularPolygon(center(r, c), 6, radius=rad * 0.98) for r in range(ROWS) for c in range(COLS)]
    for i, fpath in enumerate(files):
        cells = read_frame(fpath)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.add_collection(PatchCollection(base, facecolor="#f2f2f2", edgecolor="#d0d0d0", linewidth=0.3))
        if cells:
            cols = ["#1c5cab" if v > 0 else "#c0392b" for v in cells.values()]
            ax.add_collection(PatchCollection([RegularPolygon(center(r, c), 6, radius=rad * 0.98) for r, c in cells],
                                              facecolor=cols, edgecolor="white", linewidth=0.3))
        ax.set_xlim(-pitch, COLS * pitch + pitch)
        ax.set_ylim(-dy, ROWS * dy)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{os.path.basename(fpath)}  cells {len(cells)}", fontsize=10)
        fig.savefig(os.path.join(png, f"{i:04d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"{csv_dir}: {len(files)} frames -> {png}")
    if shutil.which("ffmpeg"):
        mp4 = os.path.join(out, os.path.basename(os.path.normpath(csv_dir)) + ".mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", os.path.join(png, "%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
        print("wrote", mp4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4)
    a = ap.parse_args()
    render(a.csv_dir, a.out, a.fps)


if __name__ == "__main__":
    main()
