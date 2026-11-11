import os
import numpy as np

import sim_trail as vel

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FORCE_SCALE_EXTERNAL = 28.0
FORCE_SCALE_INTERNAL = 0.625e-6
FORCE_C_EFF = 0.00028
MASS_DISK = 2.4e-7
FORCE_DT = 0.001

N_STEPS = vel.N_STEPS
RADIUS_CAPTURE_TIME = vel.RADIUS_CAPTURE_TIME
DISK_RADIUS = vel.DISK_RADIUS
SWARM_MARGIN_FACTOR = vel.SWARM_MARGIN_FACTOR
PAIR_A = vel.PAIR_A
PAIR_N = vel.PAIR_N
MIN_DIST = vel.MIN_DIST
EXTERNAL_RANGE = vel.EXTERNAL_RANGE
INTERNAL_RANGE = vel.INTERNAL_RANGE
DISK_MOMENT = vel.DISK_MOMENT
MM_TO_M = vel.MM_TO_M
EM_X, EM_Y = vel.EM_X, vel.EM_Y

def phys_step(disk_x, disk_y, disk_vx, disk_vy, active_cells, rng,
              scale_ext=FORCE_SCALE_EXTERNAL, scale_int=FORCE_SCALE_INTERNAL,
              c_eff=FORCE_C_EFF, mass=MASS_DISK, dt=FORCE_DT):
    Fx_mag = np.zeros_like(disk_x)
    Fy_mag = np.zeros_like(disk_y)
    for (r, c) in active_cells:
        em_x, em_y = EM_X[r, c], EM_Y[r, c]
        dx = disk_x - em_x
        dy = disk_y - em_y
        r_ = np.hypot(dx, dy)
        in_range = r_ < EXTERNAL_RANGE
        dbz_dr = vel.lookup_dbz_dr(r_)
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

    Fx = scale_ext * Fx_mag + scale_int * Fx_p - c_eff * disk_vx
    Fy = scale_ext * Fy_mag + scale_int * Fy_p - c_eff * disk_vy

    nvx = disk_vx + (Fx / mass) * dt
    nvy = disk_vy + (Fy / mass) * dt
    nx = disk_x + nvx * dt / MM_TO_M
    ny = disk_y + nvy * dt / MM_TO_M
    return nx, ny, nvx, nvy

def run_final_fraction(n_active, n_disks, duration, seed=42,
                        scale_ext=FORCE_SCALE_EXTERNAL, scale_int=FORCE_SCALE_INTERNAL,
                        c_eff=FORCE_C_EFF, mass=MASS_DISK, dt=FORCE_DT):

    rng = np.random.default_rng(seed)
    disk_x, disk_y = vel.spawn_disks(n_disks, rng)
    disk_vx = np.zeros(n_disks)
    disk_vy = np.zeros(n_disks)

    active_per_col = [vel.ex3_active_cells(c, n_active) for c in range(vel.N_COLS)]

    t_sim = 0.0
    for _ in range(round(RADIUS_CAPTURE_TIME / dt)):
        t_sim += dt
        disk_x, disk_y, disk_vx, disk_vy = phys_step(
            disk_x, disk_y, disk_vx, disk_vy, active_per_col[1], rng,
            scale_ext, scale_int, c_eff, mass, dt)

    ic_x, ic_y = disk_x.mean(), disk_y.mean()
    d_c = np.hypot(disk_x - ic_x, disk_y - ic_y)
    initial_radius_mm = np.max(d_c + DISK_RADIUS)
    swarm_radius_mm = SWARM_MARGIN_FACTOR * initial_radius_mm

    total_steps = round(N_STEPS * duration / dt)
    for s in range(1, total_steps + 1):
        t_sim += dt
        t_transport = t_sim - RADIUS_CAPTURE_TIME
        step_idx = min(int(np.floor(t_transport / duration)), N_STEPS - 1)
        col = 1 + step_idx
        disk_x, disk_y, disk_vx, disk_vy = phys_step(
            disk_x, disk_y, disk_vx, disk_vy, active_per_col[col], rng,
            scale_ext, scale_int, c_eff, mass, dt)

    ri = np.argmax(disk_x)
    ax_ = disk_x[ri] - initial_radius_mm
    ay_ = disk_y[ri]
    return np.mean(np.hypot(disk_x - ax_, disk_y - ay_) > swarm_radius_mm)

def run_one_sim(n_active, n_disks, duration, seed=42, log_every=10,
                 scale_ext=FORCE_SCALE_EXTERNAL, scale_int=FORCE_SCALE_INTERNAL,
                 c_eff=FORCE_C_EFF, mass=MASS_DISK, dt=FORCE_DT):

    rng = np.random.default_rng(seed)
    disk_x, disk_y = vel.spawn_disks(n_disks, rng)
    disk_vx = np.zeros(n_disks)
    disk_vy = np.zeros(n_disks)

    active_per_col = [vel.ex3_active_cells(c, n_active) for c in range(vel.N_COLS)]

    t_sim = 0.0
    for _ in range(round(RADIUS_CAPTURE_TIME / dt)):
        t_sim += dt
        disk_x, disk_y, disk_vx, disk_vy = phys_step(
            disk_x, disk_y, disk_vx, disk_vy, active_per_col[1], rng,
            scale_ext, scale_int, c_eff, mass, dt)

    ic_x, ic_y = disk_x.mean(), disk_y.mean()
    d_c = np.hypot(disk_x - ic_x, disk_y - ic_y)
    initial_radius_mm = np.max(d_c + DISK_RADIUS)
    swarm_radius_mm = SWARM_MARGIN_FACTOR * initial_radius_mm

    total_steps = round(N_STEPS * duration / dt)
    hist_x, hist_y, hist_t = [], [], []
    for s in range(1, total_steps + 1):
        t_sim += dt
        t_transport = t_sim - RADIUS_CAPTURE_TIME
        step_idx = min(int(np.floor(t_transport / duration)), N_STEPS - 1)
        col = 1 + step_idx
        disk_x, disk_y, disk_vx, disk_vy = phys_step(
            disk_x, disk_y, disk_vx, disk_vy, active_per_col[col], rng,
            scale_ext, scale_int, c_eff, mass, dt)
        if s % log_every == 0:
            hist_x.append(disk_x.copy())
            hist_y.append(disk_y.copy())
            hist_t.append(t_transport)

    ri = np.argmax(disk_x)
    ax_ = disk_x[ri] - initial_radius_mm
    ay_ = disk_y[ri]
    frac = np.mean(np.hypot(disk_x - ax_, disk_y - ay_) > swarm_radius_mm)

    return {
        'hist_x': np.array(hist_x), 'hist_y': np.array(hist_y),
        'hist_t': np.array(hist_t), 'final_frac': frac,
        'swarm_radius_mm': swarm_radius_mm, 'initial_radius_mm': initial_radius_mm,
    }

def load_calibrated_params():

    fit_path = os.path.join(OUT_DIR, 'force_model_fit_result.json')
    if os.path.isfile(fit_path):
        import json
        with open(fit_path) as f:
            fit = json.load(f)
        p = fit['best_params']
        print(f'Using calibrated params ({fit["date"]}): '
              f'SCALE_EXTERNAL={p["SCALE_EXTERNAL"]:.4f}  SCALE_INTERNAL={p["SCALE_INTERNAL"]:.6e}')
        return p['SCALE_EXTERNAL'], p['SCALE_INTERNAL']
    print('force_model_fit_result.json not found -- using uncalibrated FINAL_2.m defaults')
    return FORCE_SCALE_EXTERNAL, FORCE_SCALE_INTERNAL

if __name__ == '__main__':
    N_DISKS = 100
    DURATION = 3
    TARGET_STEP = 15
    TARGET_TRANSPORT_TIME = TARGET_STEP * DURATION
    scale_ext, scale_int = load_calibrated_params()
    results = {}
    for n_active in [1, 3, 7]:
        results[n_active] = run_one_sim(n_active, N_DISKS, DURATION,
                                         scale_ext=scale_ext, scale_int=scale_int)
        print(f'  n_active={n_active}  final_left_fraction={results[n_active]["final_frac"]:.3f}'
              f'  n_logged_frames={len(results[n_active]["hist_t"])}')

    all_bounds = [vel.snapshot_bounds(
        r, trail_seconds=30.0, snapshot_time=TARGET_TRANSPORT_TIME
    ) for r in results.values()]
    x_lo = min(b[0] for b in all_bounds)
    x_hi = x_lo + 29 * vel.EM_PITCH + 80.0
    y_half = max(b[3] for b in all_bounds)
    y_mid = np.mean([b[2] for b in all_bounds])
    shared_bounds = (x_lo, x_hi, y_mid, y_half)

    for n_active, res in results.items():
        out = os.path.join(OUT_DIR, f'simforce_{n_active}magnet.png')
        vel.render_trail(
            res, out, shared_bounds, trail_seconds=30.0, title=None,
            snapshot_time=TARGET_TRANSPORT_TIME
        )
