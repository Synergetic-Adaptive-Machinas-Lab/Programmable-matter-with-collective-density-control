import math

from hexgrid import cell_key, hex_dist
from planner import astar, plan_unified, build_frames
from density import compute_cluster_density, merge_home_cells

def _walls(data):
    return frozenset(cell_key(r, c) for r, c in data.get("walls", []))

def run_drst(data, initial_robots_per_cell):

    starts = [tuple(c) for c in data["start"]]
    goals = [tuple(c) for c in data["goal"]]
    walls = _walls(data)
    W = data["traceWindow"]
    cost_mode = data.get("costMode", "astar")

    paths, waypoints = plan_unified(starts, goals, cost_mode, walls)
    frames = build_frames(paths, W, waypoints)
    density_frames = compute_cluster_density(frames, initial_robots_per_cell, walls)
    return {"paths": paths, "frames": frames, "density_frames": density_frames}

def run_transform(data, initial_robots_per_cell):

    states = data["states"]
    W = data["traceWindow"]
    walls = _walls(data)

    all_frames = []
    transitions = []
    for i in range(len(states) - 1):
        starts = [tuple(c) for c in states[i]["cells"]]
        goals = [tuple(c) for c in states[i + 1]["cells"]]
        if not starts or not goals:
            paths, waypoints = [], []
        else:
            paths, waypoints = plan_unified(starts, goals, "astar", walls)
        frames = build_frames(paths, W, waypoints) if paths else [list(goals)]
        append_frames = frames if i == 0 else frames[1:]
        frame_start = len(all_frames)
        all_frames.extend(append_frames)
        transitions.append({
            "from": i, "to": i + 1,
            "fromName": states[i].get("name"), "toName": states[i + 1].get("name"),
            "paths": paths, "frameStart": frame_start, "frameEnd": len(all_frames) - 1,
        })

    density_frames = compute_cluster_density(all_frames, initial_robots_per_cell, walls)
    return {"frames": all_frames, "density_frames": density_frames, "transitions": transitions}

def run_pseudopodia(data, initial_robots_per_cell):

    home = [tuple(c) for c in data["home"]]
    objs = [tuple(c) for c in data["object"]]
    walls = _walls(data)
    single_agent = bool(data.get("singleAgentMode", False))

    if single_agent:
        entries = []
        for obj in objs:
            best_dist, best_home = math.inf, None
            for h in home:
                d = hex_dist(h[0], h[1], obj[0], obj[1])
                if d < best_dist:
                    best_dist, best_home = d, h
            path = astar(best_home[0], best_home[1], obj[0], obj[1], walls)
            entries.append({"path": path, "home": best_home, "obj": obj, "bestDist": best_dist})

        paths = [e["path"] for e in entries]
        max_len = max(len(p) for p in paths)
        waypoints = [
            [
                {"r": e["obj"][0], "c": e["obj"][1], "step": len(e["path"]) - 1},
                {"r": e["home"][0], "c": e["home"][1], "step": 0},
            ]
            for e in entries
        ]
        full_frames = build_frames(paths, max_len, waypoints)
        extend_frames = full_frames[:max_len]
    else:
        W = data["traceWindow"]
        paths, waypoints_out = plan_unified(home, objs, "astar", walls)
        waypoints = [
            [{"r": paths[i][0][0], "c": paths[i][0][1], "step": 0}] + wps
            for i, wps in enumerate(waypoints_out)
        ]
        full_frames = build_frames(paths, W, waypoints)
        max_len = max(len(p) for p in paths)
        extend_frames = full_frames[:max_len]

    retract_frames = list(reversed(extend_frames))
    raw_frames = extend_frames + retract_frames

    frames = merge_home_cells(raw_frames, home)
    density_frames = compute_cluster_density(frames, initial_robots_per_cell, walls)
    return {"paths": paths, "frames": frames, "density_frames": density_frames, "single_agent": single_agent}
