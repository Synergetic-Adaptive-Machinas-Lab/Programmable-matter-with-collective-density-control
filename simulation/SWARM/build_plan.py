import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexgrid import ROWS, COLS
from planner import plan_unified, build_frames

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "S-W-A-R-M.json")
CACHE_PATH = os.path.join(BASE_DIR, "plan_cache.json")

W = 7

def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    assert data["tab"] == "transform"
    states = data["states"]
    print(f"grid: {ROWS}x{COLS}  W={W}  {len(states)} states: "
          + ", ".join(f"{s['name']}({len(s['cells'])})" for s in states))

    all_frames = []
    transitions = []
    for i in range(len(states) - 1):
        starts = [tuple(c) for c in states[i]["cells"]]
        goals = [tuple(c) for c in states[i + 1]["cells"]]

        t0 = time.time()
        paths, waypoints = plan_unified(starts, goals, cost_mode="astar")
        t1 = time.time()
        frames = build_frames(paths, W, waypoints)
        append_frames = frames if i == 0 else frames[1:]
        frame_start = len(all_frames)
        all_frames.extend(append_frames)
        print(f"  transition {i}: {states[i]['name']} -> {states[i+1]['name']}  "
              f"{len(paths)} paths in {t1-t0:.1f}s, max path len={max(len(p) for p in paths)}, "
              f"+{len(append_frames)} frames (total {len(all_frames)})")
        transitions.append({
            "from": i, "to": i + 1,
            "fromName": states[i]["name"], "toName": states[i + 1]["name"],
            "paths": [[list(cell) for cell in p] for p in paths],
            "frameStart": frame_start, "frameEnd": len(all_frames) - 1,
        })

    print(f"total frames: {len(all_frames)}  max simultaneous active magnets = "
          f"{max(len(fr) for fr in all_frames)}")

    cache = {
        "W": W,
        "state_names": [s["name"] for s in states],
        "state_cell_counts": [len(s["cells"]) for s in states],
        "frames": [[list(cell) for cell in fr] for fr in all_frames],
        "transitions": transitions,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"wrote {CACHE_PATH}  ({len(all_frames)} frames, {os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
