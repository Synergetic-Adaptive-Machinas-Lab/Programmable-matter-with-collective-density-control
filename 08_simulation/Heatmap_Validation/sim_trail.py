import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SCALE_EXTERNAL = 28.8831
SCALE_INTERNAL = 3.23407e-07
C_EFF = 0.000260499
RADIUS_CAPTURE_TIME = 1.0

N_STEPS = 30
EXTERNAL_RANGE = 35.0
INTERNAL_RANGE = 15.0
SPAWN_RADIUS = 30.0
DISK_RADIUS = 1.5
MIN_DIST = 2 * DISK_RADIUS
SWARM_MARGIN_FACTOR = 1.5

PAIR_A = 9318.0
PAIR_N = 4.22

DISK_MOMENT = 1.7e-5
BZ_TABLE = np.array([
    [357.3266, 30.3195, 7.1030, 2.6410],
    [-30.1112, 11.3810, 3.9168, 2.1647],
    [-6.6998, -0.1569, 1.3462, 0.9944],
    [-1.9077, -0.7306, 0.2489, 0.2774],
])
BZ_RADIAL_PROFILE = 0.5 * (BZ_TABLE[:, 2] + BZ_TABLE[:, 3])
GAUSS_TO_TESLA = 1e-4
MM_TO_M = 1e-3
RADIAL_STEP = 10.0

EM_PITCH = 20.0
N_ROWS = 32
N_COLS = 32
CENTER_ROW = 14

DT = 0.01

_rows, _cols = np.meshgrid(np.arange(N_ROWS), np.arange(N_COLS), indexing='ij')
EM_X = (_cols + 0.5 * (_rows % 2)) * EM_PITCH
EM_Y = (CENTER_ROW - _rows) * EM_PITCH

SPAWN_X = (1 + 0.5 * (CENTER_ROW % 2)) * EM_PITCH
SPAWN_Y = 0.0

def lookup_dbz_dr(r):
    idx_f = np.clip(r / RADIAL_STEP, 0.0, 2.9999)
    idx_lo = np.floor(idx_f).astype(int)
    bz_lo = BZ_RADIAL_PROFILE[idx_lo]
    bz_hi = BZ_RADIAL_PROFILE[idx_lo + 1]
    return (bz_hi - bz_lo) / RADIAL_STEP * GAUSS_TO_TESLA / MM_TO_M

def ex3_active_cells(lead_col, n_active):

    cr, cc = CENTER_ROW, lead_col
    if n_active == 1:
        cells = [(cr, cc)]
    elif n_active == 3:
        if cr % 2 == 0:
            above = [(cr - 1, cc - 1), (cr - 1, cc)]
        else:
            above = [(cr - 1, cc), (cr - 1, cc + 1)]
        cells = [(cr, cc)] + above
    elif n_active == 7:
        if cr % 2 == 0:
            nbr = [(cr, cc - 1), (cr, cc + 1), (cr - 1, cc - 1), (cr - 1, cc),
                   (cr + 1, cc - 1), (cr + 1, cc)]
        else:
            nbr = [(cr, cc - 1), (cr, cc + 1), (cr - 1, cc), (cr - 1, cc + 1),
                   (cr + 1, cc), (cr + 1, cc + 1)]
        cells = [(cr, cc)] + nbr
    else:
        cells = [(cr, cc)]
    return [(r, c) for (r, c) in cells if 0 <= r < N_ROWS and 0 <= c < N_COLS]

def vel_step(disk_x, disk_y, active_cells, rng, dt=DT):
    Fx_mag = np.zeros_like(disk_x)
    Fy_mag = np.zeros_like(disk_y)
    for (r, c) in active_cells:
        em_x, em_y = EM_X[r, c], EM_Y[r, c]
        dx = disk_x - em_x
        dy = disk_y - em_y
        r_ = np.hypot(dx, dy)
        in_range = r_ < EXTERNAL_RANGE
        dbz_dr = lookup_dbz_dr(r_)
        r_safe = np.maximum(r_, 1e-9)
        Fx_mag += DISK_MOMENT * dbz_dr * (dx / r_safe) * in_range
        Fy_mag += DISK_MOMENT * dbz_dr * (dy / r_safe) * in_range

    n = disk_x.shape[0]
    dx_mat = disk_x[:, None] - disk_x[None, :]
    dy_mat = disk_y[:, None] - disk_y[None, :]
    r_mat = np.hypot(dx_mat, dy_mat)
    not_self = ~np.eye(n, dtype=bool)

    act_mat = (r_mat < INTERNAL_RANGE) & not_self
    r_clamped = np.maximum(r_mat, MIN_DIST)
    F_mat = act_mat * PAIR_A * r_clamped ** (-PAIR_N)

    ux = dx_mat / r_clamped
    uy = dy_mat / r_clamped
    degenerate = (r_mat < 1e-6) & not_self
    upper = degenerate & (np.arange(n)[:, None] < np.arange(n)[None, :])
    if np.any(upper):
        iu, ju = np.nonzero(upper)
        theta = rng.uniform(0, 2 * np.pi, len(iu))
        ux[iu, ju], uy[iu, ju] = np.cos(theta), np.sin(theta)
        ux[ju, iu], uy[ju, iu] = -np.cos(theta), -np.sin(theta)

    Fx_p = np.sum(F_mat * ux, axis=1)
    Fy_p = np.sum(F_mat * uy, axis=1)

    vx = (SCALE_EXTERNAL * Fx_mag + SCALE_INTERNAL * Fx_p) / C_EFF
    vy = (SCALE_EXTERNAL * Fy_mag + SCALE_INTERNAL * Fy_p) / C_EFF
    nx = disk_x + vx * dt / MM_TO_M
    ny = disk_y + vy * dt / MM_TO_M
    return nx, ny

def spawn_disks(n_disks, rng):
    disk_x = np.zeros(n_disks)
    disk_y = np.zeros(n_disks)
    placed = 0
    r_sp = SPAWN_RADIUS
    attempts = 0
    while placed < n_disks:
        angle = rng.uniform(0, 2 * np.pi)
        d = MIN_DIST + rng.uniform() * (r_sp - MIN_DIST)
        cx = SPAWN_X + d * np.cos(angle)
        cy = SPAWN_Y + d * np.sin(angle)
        ok = True
        if placed > 0:
            dd = np.hypot(disk_x[:placed] - cx, disk_y[:placed] - cy)
            ok = np.all(dd >= MIN_DIST * 1.05)
        if ok:
            disk_x[placed] = cx
            disk_y[placed] = cy
            placed += 1
        attempts += 1
        if attempts > 50000:
            r_sp += 5.0
            attempts = 0
    return disk_x, disk_y

def run_one_sim(n_active, n_disks, duration, seed=42, log_every=1, dt=DT):
    rng = np.random.default_rng(seed)
    disk_x, disk_y = spawn_disks(n_disks, rng)

    active_per_col = [ex3_active_cells(c, n_active) for c in range(N_COLS)]

    t_sim = 0.0
    for _ in range(round(RADIUS_CAPTURE_TIME / dt)):
        t_sim += dt
        disk_x, disk_y = vel_step(disk_x, disk_y, active_per_col[1], rng, dt)

    ic_x, ic_y = disk_x.mean(), disk_y.mean()
    d_c = np.hypot(disk_x - ic_x, disk_y - ic_y)
    initial_radius_mm = np.max(d_c + DISK_RADIUS)
    swarm_radius_mm = SWARM_MARGIN_FACTOR * initial_radius_mm

    total_steps = round(N_STEPS * duration / dt)
    hist_x, hist_y, hist_t, hist_col = [], [], [], []
    for s in range(1, total_steps + 1):
        t_sim += dt
        t_transport = t_sim - RADIUS_CAPTURE_TIME
        step_idx = min(int(np.floor(t_transport / duration)), N_STEPS - 1)
        col = 1 + step_idx
        disk_x, disk_y = vel_step(disk_x, disk_y, active_per_col[col], rng, dt)
        if s % log_every == 0:
            hist_x.append(disk_x.copy())
            hist_y.append(disk_y.copy())
            hist_t.append(t_transport)
            hist_col.append(col)

    ri = np.argmax(disk_x)
    ax_ = disk_x[ri] - initial_radius_mm
    ay_ = disk_y[ri]
    frac = np.mean(np.hypot(disk_x - ax_, disk_y - ay_) > swarm_radius_mm)

    return {
        'hist_x': np.array(hist_x), 'hist_y': np.array(hist_y),
        'hist_t': np.array(hist_t), 'hist_col': np.array(hist_col),
        'final_frac': frac, 'swarm_radius_mm': swarm_radius_mm,
        'initial_radius_mm': initial_radius_mm,
    }

ACCENT = (0.05, 0.05, 0.05)
TRAIL_COLOR = (0.55, 0.68, 0.82)
FULL_TRANSPORT_TIME_S = 90.0
SUNSET_CMAP = LinearSegmentedColormap.from_list(
    "rainbow_recommended",
    ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    N=256,
)
SUNSET_NORM = Normalize(0, FULL_TRANSPORT_TIME_S)

TRUE_ROBOT_DIAM_MM = 2.53
ROBOT_DISPLAY_SCALE = 1.35

def snapshot_bounds(result, trail_seconds=8.0, x_margin=(20, 40), y_margin=20,
                    y_min_half=30.0, snapshot_time=None):

    hist_x, hist_y, hist_t = result['hist_x'], result['hist_y'], result['hist_t']
    snapshot_idx = (len(hist_t) // 2 if snapshot_time is None
                    else int(np.argmin(np.abs(hist_t - snapshot_time))))
    t_now = hist_t[snapshot_idx]
    win_idx = np.nonzero((hist_t > t_now - trail_seconds) & (hist_t <= t_now))[0]
    win_idx = win_idx[win_idx <= snapshot_idx]

    x_now, y_now = hist_x[snapshot_idx], hist_y[snapshot_idx]
    x_lo = x_now.min() - x_margin[0]
    x_hi = x_now.max() + x_margin[1]
    y_mid = (hist_y[win_idx[0]:snapshot_idx + 1].mean() if len(win_idx) else y_now.mean())
    y_half = max(y_min_half, np.ptp(y_now) / 2 + y_margin)
    return x_lo, x_hi, y_mid, y_half

def render_trail(result, out_path, bounds, trail_seconds=8.0, title=None,
                 snapshot_time=None, magnet_pos=None):
    hist_x, hist_y, hist_t = result['hist_x'], result['hist_y'], result['hist_t']
    n_frames, n_disks = hist_x.shape
    x_lo, x_hi, y_mid, y_half = bounds

    snapshot_idx = (n_frames // 2 if snapshot_time is None
                    else int(np.argmin(np.abs(hist_t - snapshot_time))))
    t_now = hist_t[snapshot_idx]
    window_mask = (hist_t > t_now - trail_seconds) & (hist_t <= t_now)
    win_idx = np.nonzero(window_mask)[0]
    win_idx = win_idx[win_idx <= snapshot_idx]

    x_now, y_now = hist_x[snapshot_idx], hist_y[snapshot_idx]

    fig_w = 9.0
    fig_h = fig_w * (2 * y_half) / (x_hi - x_lo)
    fig_h = float(np.clip(fig_h, 1.6, 3.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='w')

    for i in range(n_disks):
        xs = hist_x[win_idx, i]
        ys = hist_y[win_idx, i]
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        segment_times = hist_t[win_idx][1:]
        colors = SUNSET_CMAP(SUNSET_NORM(segment_times))
        colors[:, 3] = np.linspace(0.18, 0.80, len(segs))
        lc = LineCollection(segs, colors=colors, linewidths=1.3, zorder=2)
        ax.add_collection(lc)

    mm_per_inch = (x_hi - x_lo) / fig_w
    marker_diam_pt = TRUE_ROBOT_DIAM_MM / mm_per_inch * 72 * ROBOT_DISPLAY_SCALE
    marker_area_pt2 = np.pi * (marker_diam_pt / 2) ** 2
    ax.scatter(x_now, y_now, s=marker_area_pt2, color=ACCENT, edgecolors='none', zorder=3)

    if magnet_pos:
        mx = [p[0] for p in magnet_pos]
        my = [p[1] for p in magnet_pos]

        magnet_diam_pt = 20.0 / mm_per_inch * 72
        magnet_area_pt2 = np.pi * (magnet_diam_pt / 2) ** 2
        ax.scatter(mx, my, s=magnet_area_pt2, marker='o', facecolors='none',
                   edgecolors='#e0692a', linewidths=2.2, zorder=4)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_mid - y_half, y_mid + y_half)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, pad=6)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300, facecolor='w')
    plt.close(fig)
    print('saved', out_path)

if __name__ == '__main__':
    N_DISKS = 100
    DURATION = 3
    results = {}
    for n_active in [1, 3, 7]:
        results[n_active] = run_one_sim(n_active, N_DISKS, DURATION)
        print(f'  n_active={n_active}  final_left_fraction={results[n_active]["final_frac"]:.3f}')

    all_bounds = [snapshot_bounds(r) for r in results.values()]
    x_lo = min(b[0] for b in all_bounds)
    x_hi = max(b[1] for b in all_bounds)
    y_half = max(b[3] for b in all_bounds)
    y_mid = np.mean([b[2] for b in all_bounds])
    shared_bounds = (x_lo, x_hi, y_mid, y_half)

    for n_active, res in results.items():
        sfx = '' if n_active == 1 else 's'
        out = os.path.join(OUT_DIR, f'sim_{n_active}magnet.png')
        render_trail(res, out, shared_bounds, title=None)
