from collections import deque

from hexgrid import cell_key, neighbors

def extract_clusters(cells, walls=frozenset()):

    cell_set = {cell_key(r, c) for r, c in cells}
    visited = set()
    clusters = []
    for r, c in cells:
        start_key = cell_key(r, c)
        if start_key in visited:
            continue
        keys = set()
        queue = deque([(r, c)])
        visited.add(start_key)
        while queue:
            cr, cc = queue.popleft()
            keys.add(cell_key(cr, cc))
            for nr, nc in neighbors(cr, cc, walls):
                nk = cell_key(nr, nc)
                if nk in cell_set and nk not in visited:
                    visited.add(nk)
                    queue.append((nr, nc))
        clusters.append({"cell_keys": keys, "magnetCount": len(keys)})
    clusters.sort(key=lambda cl: -cl["magnetCount"])
    return clusters

def merge_home_cells(frames, home_cells):

    out = []
    for fr in frames:
        m = {cell_key(r, c): (r, c) for r, c in fr}
        for r, c in home_cells:
            m[cell_key(r, c)] = (r, c)
        out.append(list(m.values()))
    return out

def compute_cluster_density(frames, initial_robots_per_cell, walls=frozenset(), log=None):

    out = []
    prev_clusters = []
    for k, frame_cells in enumerate(frames):
        raw = extract_clusters(frame_cells, walls)
        n, m = len(raw), len(prev_clusters)

        overlaps = []
        for rc in raw:
            ov = {}
            for pi in range(m):
                count = sum(1 for key in rc["cell_keys"] if key in prev_clusters[pi]["cell_keys"])
                if count > 0:
                    ov[pi] = count
            overlaps.append(ov)

        cur_visited = [False] * n
        prev_visited = [False] * m
        masses = [0.0] * n
        parent_ids_of = [[] for _ in range(n)]

        for ci in range(n):
            if cur_visited[ci]:
                continue
            comp_cur, comp_prev = [ci], []
            cur_visited[ci] = True
            stack = [("c", ci)]
            while stack:
                side, idx = stack.pop()
                if side == "c":
                    for pi in overlaps[idx].keys():
                        if not prev_visited[pi]:
                            prev_visited[pi] = True
                            comp_prev.append(pi)
                            stack.append(("p", pi))
                else:
                    for cj in range(n):
                        if cur_visited[cj] or idx not in overlaps[cj]:
                            continue
                        cur_visited[cj] = True
                        comp_cur.append(cj)
                        stack.append(("c", cj))

            if not comp_prev:
                if k == 0:
                    for cj in comp_cur:
                        masses[cj] = initial_robots_per_cell * raw[cj]["magnetCount"]
                        parent_ids_of[cj] = []
                else:
                    for cj in comp_cur:
                        masses[cj] = 0.0
                        parent_ids_of[cj] = []
                        if log:
                            log(f"[DENSITY] Frame {k}: cluster of {raw[cj]['magnetCount']} magnet(s) "
                                f"has no overlapping parent — mass set to 0 (no route-continuity data available)")
                continue

            event_mass = sum(prev_clusters[pi]["mass"] for pi in comp_prev)
            event_magnet_count = sum(raw[cj]["magnetCount"] for cj in comp_cur)
            event_density = event_mass / event_magnet_count if event_magnet_count > 0 else 0.0
            for cj in comp_cur:
                masses[cj] = event_density * raw[cj]["magnetCount"]
                parent_ids_of[cj] = [pi + 1 for pi in comp_prev]

        out.append([
            {
                "id": idx + 1,
                "magnetCount": rc["magnetCount"],
                "mass": masses[idx],
                "density": masses[idx] / rc["magnetCount"] if rc["magnetCount"] > 0 else 0.0,
                "parentIds": parent_ids_of[idx],
            }
            for idx, rc in enumerate(raw)
        ])
        prev_clusters = [{"cell_keys": rc["cell_keys"], "mass": masses[idx]} for idx, rc in enumerate(raw)]

    return out
