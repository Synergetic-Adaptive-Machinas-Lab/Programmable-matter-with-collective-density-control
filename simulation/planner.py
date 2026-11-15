import heapq
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from hexgrid import ROWS, COLS, cell_key, hex_dist, neighbors

def astar(sr, sc, gr, gc, walls=frozenset()):

    if (sr, sc) == (gr, gc):
        return Path([(sr, sc)])

    N = ROWS * COLS
    g = [math.inf] * N
    f = [math.inf] * N
    par = [-1] * N
    sid, gid = cell_key(sr, sc), cell_key(gr, gc)
    g[sid] = 0.0
    f[sid] = hex_dist(sr, sc, gr, gc)
    heap = [(f[sid], sid)]
    in_open = {sid}

    while heap:
        _, cur = heapq.heappop(heap)
        if cur not in in_open:
            continue
        in_open.discard(cur)
        if cur == gid:
            path = []
            idx = cur
            while idx != -1:
                path.append((idx // COLS, idx % COLS))
                idx = par[idx]
            path.reverse()
            return Path(path)
        cr, cc = cur // COLS, cur % COLS
        for nr, nc in neighbors(cr, cc, walls):
            nid = cell_key(nr, nc)
            tg = g[cur] + hex_dist(cr, cc, nr, nc)
            if tg < g[nid]:
                par[nid] = cur
                g[nid] = tg
                f[nid] = tg + hex_dist(nr, nc, gr, gc)
                heapq.heappush(heap, (f[nid], nid))
                in_open.add(nid)

    return Path([(sr, sc), (gr, gc)], unreachable=True)

class Path(list):

    def __new__(cls, cells, unreachable=False):
        obj = super().__new__(cls, cells)
        return obj

    def __init__(self, cells, unreachable=False):
        super().__init__(cells)
        self.unreachable = unreachable

def path_len(path):

    if getattr(path, "unreachable", False):
        return 1e9
    d = 0.0
    for i in range(1, len(path)):
        d += hex_dist(path[i - 1][0], path[i - 1][1], path[i][0], path[i][1])
    return d

def plan_unified(starts, goals, cost_mode="astar", walls=frozenset()):

    nS, nG = len(starts), len(goals)
    k = max(nS, nG)
    path_cache = {}

    def get_path(si, gi):
        key = (si, gi)
        if key not in path_cache:
            path_cache[key] = astar(starts[si][0], starts[si][1], goals[gi][0], goals[gi][1], walls)
        return path_cache[key]

    source_slots = [(i % nS) if nS < nG else i for i in range(k)]
    goal_slots = [(j % nG) if nS > nG else j for j in range(k)]

    if cost_mode == "astar":
        for si in range(nS):
            for gi in range(nG):
                get_path(si, gi)

    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            si, gi = source_slots[i], goal_slots[j]
            if cost_mode == "astar":
                C[i, j] = path_len(get_path(si, gi))
            else:
                C[i, j] = hex_dist(starts[si][0], starts[si][1], goals[gi][0], goals[gi][1])

    row_ind, col_ind = linear_sum_assignment(C)

    assign = [None] * k
    for r, c in zip(row_ind, col_ind):
        assign[r] = c

    paths, waypoints = [], []
    for i in range(k):
        si = source_slots[i]
        gi = goal_slots[assign[i]]
        path = get_path(si, gi)
        paths.append(path)
        waypoints.append([{"r": goals[gi][0], "c": goals[gi][1], "step": len(path) - 1}])
    return paths, waypoints

def build_frames(paths, W, waypoints=None):

    if waypoints is None:
        waypoints = [[] for _ in paths]
    max_len = max(len(p) for p in paths)
    total = max_len + W - 1
    frames = []
    for f in range(total):
        cell_map = {}
        for i, path in enumerate(paths):
            ws = max(0, f - W + 1)
            we = min(f, len(path) - 1)
            for k in range(ws, we + 1):
                r, c = path[k]
                cell_map[cell_key(r, c)] = (r, c)
            for wp in waypoints[i]:
                if f >= wp["step"]:
                    cell_map[cell_key(wp["r"], wp["c"])] = (wp["r"], wp["c"])
        if f == total - 1:
            cell_map = {}
            for path in paths:
                r, c = path[-1]
                cell_map[cell_key(r, c)] = (r, c)
            for wps in waypoints:
                for wp in wps:
                    cell_map[cell_key(wp["r"], wp["c"])] = (wp["r"], wp["c"])
        frames.append(list(cell_map.values()))
    return frames
