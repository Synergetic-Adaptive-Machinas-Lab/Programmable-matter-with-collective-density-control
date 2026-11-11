import numpy as np
from scipy.spatial import cKDTree

SCALE_EXTERNAL = 37.61814936649954
SCALE_INTERNAL = 4.237444959265931e-07
C_EFF = 0.00028
MASS_DISK = 2.4e-7
DT = 0.001

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

EXTERNAL_RANGE = 35.0
INTERNAL_RANGE = 15.0
DISK_RADIUS = 1.5
MIN_DIST = 2 * DISK_RADIUS

EM_PITCH = 20.0

def em_xy_mm(r, c):

    x = (c + 0.5 * (r & 1)) * EM_PITCH
    y = r * np.sqrt(3) / 2 * EM_PITCH
    return x, y

def _bz_gradient(r):

    idx_f = np.clip(r / RADIAL_STEP, 0, 2.9999)
    idx_lo = np.floor(idx_f).astype(int)
    bz_lo = BZ_RADIAL_PROFILE[idx_lo]
    bz_hi = BZ_RADIAL_PROFILE[idx_lo + 1]
    return (bz_hi - bz_lo) / RADIAL_STEP * GAUSS_TO_TESLA / MM_TO_M

def magnetic_force(disk_x, disk_y, active_x, active_y):

    if len(active_x) == 0:
        return np.zeros_like(disk_x), np.zeros_like(disk_y)
    dx = disk_x[:, None] - active_x[None, :]
    dy = disk_y[:, None] - active_y[None, :]
    r = np.hypot(dx, dy)
    in_range = r < EXTERNAL_RANGE
    dbz_dr = _bz_gradient(r)
    r_safe = np.maximum(r, 1e-9)
    coef = DISK_MOMENT * dbz_dr * in_range
    Fx = np.sum(coef * dx / r_safe, axis=1)
    Fy = np.sum(coef * dy / r_safe, axis=1)
    return Fx, Fy

def pairwise_force(disk_x, disk_y, rng):

    n = len(disk_x)
    pts = np.column_stack([disk_x, disk_y])
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=INTERNAL_RANGE, output_type="ndarray")
    Fx = np.zeros(n)
    Fy = np.zeros(n)
    if len(pairs) == 0:
        return Fx, Fy
    i, j = pairs[:, 0], pairs[:, 1]
    dx = disk_x[i] - disk_x[j]
    dy = disk_y[i] - disk_y[j]
    r = np.hypot(dx, dy)
    r_clamped = np.maximum(r, MIN_DIST)
    f = PAIR_A * r_clamped ** (-PAIR_N)
    ux = dx / r_clamped
    uy = dy / r_clamped

    degenerate = r < 1e-6
    if np.any(degenerate):
        k = np.count_nonzero(degenerate)
        theta = rng.uniform(0, 2 * np.pi, k)
        ux[degenerate], uy[degenerate] = np.cos(theta), np.sin(theta)

    fx, fy = f * ux, f * uy
    np.add.at(Fx, i, fx); np.add.at(Fx, j, -fx)
    np.add.at(Fy, i, fy); np.add.at(Fy, j, -fy)
    return Fx, Fy

def phys_step(disk_x, disk_y, disk_vx, disk_vy, active_x, active_y, rng, dt=DT):

    Fx_mag, Fy_mag = magnetic_force(disk_x, disk_y, active_x, active_y)
    Fx_p, Fy_p = pairwise_force(disk_x, disk_y, rng)

    Fx = SCALE_EXTERNAL * Fx_mag + SCALE_INTERNAL * Fx_p - C_EFF * disk_vx
    Fy = SCALE_EXTERNAL * Fy_mag + SCALE_INTERNAL * Fy_p - C_EFF * disk_vy

    nvx = disk_vx + (Fx / MASS_DISK) * dt
    nvy = disk_vy + (Fy / MASS_DISK) * dt
    nx = disk_x + nvx * dt / MM_TO_M
    ny = disk_y + nvy * dt / MM_TO_M
    return nx, ny, nvx, nvy
