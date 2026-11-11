import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hexgrid import ROWS, COLS, cell_key
from experiments import run_drst, run_transform, run_pseudopodia

RUNNERS = {"drst": run_drst, "transform": run_transform, "pseudopodia": run_pseudopodia}
ON = 7
WALL = 9


def walls_of(data):
    return [tuple(w) for w in data.get("walls", [])]


def frame_to_csv(cells, walls):
    payload = [0] * (ROWS * COLS)
    for r, c in walls:
        payload[c * ROWS + (ROWS - 1 - r)] = WALL
    wall_keys = {cell_key(r, c) for r, c in walls}
    for r, c in cells:
        if cell_key(r, c) not in wall_keys:
            payload[c * ROWS + (ROWS - 1 - r)] = ON
    return [payload[i:i + COLS] for i in range(0, ROWS * COLS, COLS)]


def write_frames(frames, walls, outdir):
    d = os.path.join(outdir, "frames")
    os.makedirs(d, exist_ok=True)
    for i, cells in enumerate(frames):
        with open(os.path.join(d, f"{i:04d}.csv"), "w", newline="") as f:
            csv.writer(f).writerows(frame_to_csv(cells, walls))
    return d


def write_density(result, outdir):
    path = os.path.join(outdir, "density.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "active_cells", "clusters", "mass", "density_per_cell", "min_cluster_density", "max_cluster_density"])
        for i, (cells, cl) in enumerate(zip(result["frames"], result["density_frames"])):
            mass = sum(c["mass"] for c in cl)
            n = len(cells)
            dens = [c["density"] for c in cl]
            w.writerow([i, n, len(cl), round(mass, 3), round(mass / n, 3) if n else 0,
                        round(min(dens), 3) if dens else 0, round(max(dens), 3) if dens else 0])
    return path


def cell_density_map(cells, clusters, walls):
    from density import extract_clusters
    out = {}
    raw = extract_clusters(cells, frozenset(cell_key(r, c) for r, c in walls))
    for rc, cl in zip(raw, clusters):
        for k in rc["cell_keys"]:
            out[(k // COLS, k % COLS)] = cl["density"]
    return out


def render(result, walls, outdir, fps, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    pitch = 20.0
    dy = pitch * np.sqrt(3) / 2
    rad = pitch / np.sqrt(3)
    frames = result["frames"]
    vmax = max((c["density"] for cl in result["density_frames"] for c in cl), default=1.0) or 1.0
    wall_set = set(walls)
    d = os.path.join(outdir, "png")
    os.makedirs(d, exist_ok=True)

    def center(r, c):
        return ((c + 0.5 * (r & 1)) * pitch, (ROWS - 1 - r) * dy)

    base = [RegularPolygon(center(r, c), 6, radius=rad * 0.98) for r in range(ROWS) for c in range(COLS)]
    for i, cells in enumerate(frames):
        dens = cell_density_map(cells, result["density_frames"][i], walls)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.add_collection(PatchCollection(base, facecolor="#f2f2f2", edgecolor="#d0d0d0", linewidth=0.3))
        if wall_set:
            ax.add_collection(PatchCollection([RegularPolygon(center(r, c), 6, radius=rad * 0.98) for r, c in wall_set],
                                              facecolor="#444444", edgecolor="#444444", linewidth=0.3))
        if cells:
            cols = [plt.cm.Blues(0.3 + 0.7 * min(dens.get(rc, 0) / vmax, 1.0)) for rc in cells]
            ax.add_collection(PatchCollection([RegularPolygon(center(r, c), 6, radius=rad * 0.98) for r, c in cells],
                                              facecolor=cols, edgecolor="white", linewidth=0.3))
        ax.set_xlim(-pitch, COLS * pitch + pitch)
        ax.set_ylim(-dy, ROWS * dy)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}  frame {i}/{len(frames) - 1}  cells {len(cells)}", fontsize=10)
        fig.savefig(os.path.join(d, f"{i:04d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    mp4 = os.path.join(outdir, f"{title}.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", os.path.join(d, "%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
        return mp4
    return d


def run(json_path, rho0, outdir, fps=2, video=True, name=None):
    with open(json_path) as f:
        data = json.load(f)
    name = name or os.path.splitext(os.path.basename(json_path))[0]
    tab = data["tab"]
    result = RUNNERS[tab](data, rho0)
    walls = walls_of(data)
    os.makedirs(outdir, exist_ok=True)
    write_frames(result["frames"], walls, outdir)
    write_density(result, outdir)
    out = render(result, walls, outdir, fps, name) if video else None
    n = len(result["frames"])
    last = result["density_frames"][-1]
    print(f"{name}: tab={tab} rho0={rho0} frames={n} final_cells={len(result['frames'][-1])} "
          f"final_clusters={len(last)} final_mass={sum(c['mass'] for c in last):.0f} -> {outdir}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--rho0", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    run(a.json, a.rho0, a.out, a.fps, not a.no_video)


if __name__ == "__main__":
    main()
