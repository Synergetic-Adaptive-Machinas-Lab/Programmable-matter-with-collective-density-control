import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection, LineCollection

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
sys.path[:0] = [os.path.join(ROOT, "planner"), os.path.join(ROOT, "simulation"),
                os.path.join(ROOT, "simulation", "SWARM")]

from hexgrid import ROWS, COLS, cell_key, neighbors
from planner import plan_unified
from experiments import run_transform
import force_model as fm
import simulate_swarm as sw

PITCH = 20.0
DY = PITCH * np.sqrt(3) / 2
RAD = PITCH / np.sqrt(3)


def center(r, c):
    return ((c + 0.5 * (r & 1)) * PITCH, (ROWS - 1 - r) * DY)


def hexes(ax, cells, **kw):
    cells = list(cells)
    if cells:
        ax.add_collection(PatchCollection([RegularPolygon(center(r, c), 6, radius=RAD) for r, c in cells], **kw))


def grid(ax, walls=()):
    hexes(ax, [(r, c) for r in range(ROWS) for c in range(COLS)], facecolor="0.95", edgecolor="0.85", linewidth=0.2)
    hexes(ax, walls, facecolor="0.3", edgecolor="0.3", linewidth=0.2)
    ax.set_xlim(-PITCH, COLS * PITCH + PITCH / 2)
    ax.set_ylim(-DY, ROWS * DY)
    ax.set_aspect("equal")
    ax.axis("off")


def spec_title(fig, spec, text):
    box = spec.get_position(fig)
    fig.text((box.x0 + box.x1) / 2, box.y1, text, ha="center", va="bottom", fontsize=12)


def load(folder, name):
    with open(os.path.join(ROOT, folder, "inputs", name + ".json")) as f:
        return json.load(f)


def plot_planner(fig, spec):
    ax = fig.add_subplot(spec)
    data = load("maze_demo", "maze_merge")
    starts = [tuple(c) for c in data["start"]]
    goals = [tuple(c) for c in data["goal"]]
    walls = [tuple(c) for c in data["walls"]]
    paths, _ = plan_unified(starts, goals, data["costMode"], frozenset(cell_key(r, c) for r, c in walls))
    grid(ax, walls)
    hexes(ax, starts, facecolor="tab:blue", edgecolor="w", linewidth=0.2)
    hexes(ax, goals, facecolor="tab:orange", edgecolor="w", linewidth=0.2)
    ax.add_collection(LineCollection([[center(r, c) for r, c in p] for p in paths], colors="k", linewidths=0.6, alpha=0.5))
    ax.set_title("planner: maze_merge paths (blue = start, orange = goal)")


def plot_transport(fig, spec):
    ax = fig.add_subplot(spec)
    row, n_disks, step_s = 14, 100, 1.0
    rng = np.random.default_rng(sw.SEED)
    cluster = lambda c: [(row, c)] + neighbors(row, c)
    x, y = sw.spawn_disks(cluster(1), rng, n_disks)
    vx, vy = np.zeros_like(x), np.zeros_like(y)
    hist = [(x.copy(), y.copy())]
    for col in range(1, COLS - 1):
        mx, my = (np.array(v) for v in zip(*(fm.em_xy_mm(r, c) for r, c in cluster(col))))
        for k in range(int(step_s / fm.DT)):
            x, y, vx, vy = fm.phys_step(x, y, vx, vy, mx, my, rng)
            if k % 100 == 99:
                hist.append((x.copy(), y.copy()))
    pts = np.array(hist).transpose(0, 2, 1)
    segs = np.stack([pts[:-1], pts[1:]], axis=2).reshape(-1, 2, 2)
    t = np.repeat(np.arange(1, len(pts)) * 100 * fm.DT, n_disks)
    lc = LineCollection(segs, array=t, cmap="viridis", linewidths=0.5, alpha=0.6)
    ax.add_collection(lc)
    ax.scatter(*hist[0], s=4, c="0.6", linewidths=0)
    ax.scatter(x, y, s=4, c="k", linewidths=0)
    ax.set_xlim(-PITCH, COLS * PITCH)
    ax.set_ylim(fm.em_xy_mm(row, 0)[1] - 4 * PITCH, fm.em_xy_mm(row, 0)[1] + 4 * PITCH)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.06, pad=0.28, label="time (s)")
    ax.set_title(f"simulation/force_model: {n_disks} disks towed left to right by 7 magnets, {step_s:g} s per column", fontsize=10)


def plot_ui(fig, spec):
    ax = fig.add_subplot(spec)
    ax.imshow(plt.imread(os.path.join(ROOT, "ui_screenshot.png")))
    ax.axis("off")
    ax.set_title("control UI")


def plot_swarm_demo(fig, spec):
    data = load("SWARM_demo", "S-W-A-R-M")
    data["states"] = data["states"][:2]
    frames = run_transform(data, 16)["frames"]
    n = len(frames) - 1
    keys = [0, n // 4, n // 2, n]
    sub = spec.subgridspec(2, len(keys), hspace=0.15, wspace=0.05)
    for j, k in enumerate(keys):
        ax = fig.add_subplot(sub[0, j])
        grid(ax)
        hexes(ax, frames[k], facecolor="tab:blue", edgecolor="tab:blue", linewidth=0.2)
        ax.set_title(f"frame {k}", fontsize=9)
    ax = fig.add_subplot(sub[1, :])
    ax.plot([len(f) for f in frames])
    ax.set_xlabel("frame")
    ax.set_ylabel("magnets on")
    spec_title(fig, spec, "SWARM_demo: S -> W")


def video_frame(path, index):
    w, h = (int(v) for v in subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{index})",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def plot_sim(fig, spec):
    video = os.path.join(ROOT, "simulation", "SWARM", "swarm_sim_S-W-A.mp4")
    sub = spec.subgridspec(1, 2, wspace=0.05)
    shots = [("S (after plan frame 0)", video_frame(video, round(1 / 47 * 746) - 1)), ("W (after plan frame 22)", video_frame(video, round(23 / 47 * 746)))]
    ys, xs = np.nonzero(np.minimum(shots[0][1], shots[1][1]).min(axis=2) < 128)
    for j, (label, img) in enumerate(shots):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(img[ys.min() - 10:ys.max() + 10, xs.min() - 10:xs.max() + 10])
        ax.axis("off")
        ax.set_title(label, fontsize=10)
    spec_title(fig, spec, "simulation/SWARM: frames from swarm_sim_S-W-A.mp4")


def save_panel(name, plot, figsize):
    fig = plt.figure(figsize=figsize)
    plot(fig, fig.add_gridspec(1, 1, left=0.08, right=0.95, top=0.9, bottom=0.15)[0])
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    save_panel("planner", plot_planner, (7, 6.5))
    save_panel("swarm_demo", plot_swarm_demo, (7, 6.5))
    save_panel("transport", plot_transport, (8, 4.4))
    save_panel("sim_swarm", plot_sim, (8, 4.2))

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.94, top=0.96, bottom=0.05, wspace=0.1, hspace=0.15)
    plot_planner(fig, gs[0, 0])
    plot_swarm_demo(fig, gs[1, 0])
    right = gs[:, 1].subgridspec(3, 1, height_ratios=[1.1, 0.8, 1], hspace=0.25)
    plot_ui(fig, right[0])
    plot_transport(fig, right[1])
    plot_sim(fig, right[2])
    path = os.path.join(ROOT, "overview.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
