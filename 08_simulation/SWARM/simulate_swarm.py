import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import force_model as fm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "plan_cache.json")
OUT_VIDEO = os.path.join(BASE_DIR, "swarm_sim.mp4")

ROWS, COLS = 32, 32
N_MAG = ROWS * COLS
TOTAL_ROBOTS = 3200
PER_FRAME_DWELL_S = 8.0
VIDEO_FPS = 25
VIDEO_TARGET_SECONDS = 30.0
SEED = 42

def load_frames():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    return [[tuple(cell) for cell in fr] for fr in cache["frames"]], cache["W"]

def magnet_state_1024(frame_cells):
    state = np.zeros(N_MAG, dtype=bool)
    for r, c in frame_cells:
        state[r * COLS + c] = True
    return state

def spawn_disks(start_cells, rng, total):
    n = len(start_cells)
    base = total // n
    remainder = total - base * n
    counts = np.full(n, base, dtype=int)
    bump_idx = np.linspace(0, n - 1, remainder, dtype=int) if remainder else []
    counts[bump_idx] += 1
    assert counts.sum() == total

    spawn_r = 0.45 * fm.EM_PITCH
    xs, ys = [], []
    for (r, c), k in zip(start_cells, counts):
        if k == 0:
            continue
        cx, cy = fm.em_xy_mm(r, c)
        ang = rng.uniform(0, 2 * np.pi, k)
        rad = spawn_r * np.sqrt(rng.uniform(0, 1, k))
        xs.append(cx + rad * np.cos(ang))
        ys.append(cy + rad * np.sin(ang))
    return np.concatenate(xs), np.concatenate(ys)

def frame_step_counts(n_frames, dwell=PER_FRAME_DWELL_S, dt=fm.DT):
    return np.full(n_frames, int(round(dwell / dt)), dtype=int)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    frames, W = load_frames()
    if args.frames is not None:
        frames = frames[:args.frames]
    n_frames = len(frames)
    print(f"loaded {n_frames} frames, W={W}"
          + (f"  (PREVIEW: first {args.frames})" if args.frames else ""))

    global OUT_VIDEO
    if args.out:
        OUT_VIDEO = args.out
    elif args.frames is not None:
        OUT_VIDEO = os.path.join(BASE_DIR, "swarm_sim_preview.mp4")

    start_cells = frames[0]
    print(f"spawning {TOTAL_ROBOTS} disks over {len(start_cells)} start cells "
          f"(~{TOTAL_ROBOTS/len(start_cells):.2f}/cell)")

    rng = np.random.default_rng(SEED)
    disk_x, disk_y = spawn_disks(start_cells, rng, TOTAL_ROBOTS)
    disk_vx = np.zeros_like(disk_x)
    disk_vy = np.zeros_like(disk_y)

    steps = frame_step_counts(n_frames)
    sim_seconds = n_frames * PER_FRAME_DWELL_S
    print(f"total physics steps: {steps.sum()} over {n_frames} frames "
          f"({PER_FRAME_DWELL_S}s/frame dwell -> {sim_seconds:.0f}s simulated, "
          f"@ DT={fm.DT}s)  steps/frame = {steps[0]}")

    all_cells = {cell for fr in frames for cell in fr} | set(start_cells)
    xs_all, ys_all = zip(*(fm.em_xy_mm(r, c) for r, c in all_cells))
    margin = 4 * fm.EM_PITCH
    xlim = (min(xs_all) - margin, max(xs_all) + margin)
    ylim = (min(ys_all) - margin, max(ys_all) + margin)

    fig_w_in = 8.0
    fig_h_in = fig_w_in * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=150)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    scat = ax.scatter(disk_x, disk_y, s=1, c="black", linewidths=0)
    fig.canvas.draw()
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((fm.DISK_RADIUS, 0))
    point_radius = np.hypot(*(p1 - p0)) / (fig.dpi / 72.0)
    scat.set_sizes([np.pi * point_radius ** 2])

    frames_dir = tempfile.mkdtemp(prefix="swarm_frames_")
    target_render_frames = VIDEO_TARGET_SECONDS * VIDEO_FPS
    render_every = max(1, round(steps.sum() / target_render_frames))
    print(f"rendering every {render_every} physics step(s) of {steps.sum()} total "
          f"-> ~{steps.sum()/render_every/VIDEO_FPS:.1f}s video @ {VIDEO_FPS}fps")

    global_step = 0
    render_idx = 0
    total_steps = int(steps.sum())
    t_start = time.time()

    def save_render(idx):
        scat.set_offsets(np.column_stack([disk_x, disk_y]))
        fig.canvas.draw()
        fig.savefig(os.path.join(frames_dir, f"f{idx:06d}.png"))

    save_render(render_idx); render_idx += 1

    for fi, frame_cells in enumerate(frames):
        state = magnet_state_1024(frame_cells)
        elapsed = time.time() - t_start
        eta = (elapsed / global_step * (total_steps - global_step)) if global_step else float("nan")
        print(f"  frame {fi+1}/{n_frames}: {state.sum()} / {N_MAG} magnets on, "
              f"{steps[fi]} physics steps  [elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min]",
              flush=True)
        if frame_cells:
            ax_, ay_ = zip(*(fm.em_xy_mm(r, c) for r, c in frame_cells))
            active_x, active_y = np.array(ax_), np.array(ay_)
        else:
            active_x, active_y = np.array([]), np.array([])

        for _ in range(steps[fi]):
            disk_x, disk_y, disk_vx, disk_vy = fm.phys_step(
                disk_x, disk_y, disk_vx, disk_vy, active_x, active_y, rng)
            global_step += 1
            if global_step % render_every == 0:
                save_render(render_idx); render_idx += 1

    print(f"rendered {render_idx} video frames in {frames_dir}")

    cmd = [
        "ffmpeg", "-y", "-framerate", str(VIDEO_FPS),
        "-i", os.path.join(frames_dir, "f%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        OUT_VIDEO,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT_VIDEO}")

if __name__ == "__main__":
    main()
