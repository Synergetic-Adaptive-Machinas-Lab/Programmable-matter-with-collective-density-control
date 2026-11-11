import os
import csv
import json
import time
import datetime
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize

import sim_trail_force as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_CSV = os.path.join(BASE_DIR, 'ground_truth_left_behind_final.csv')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

C_EFF_FIXED = sf.FORCE_C_EFF
MASS_FIXED = sf.MASS_DISK
DT_FIXED = sf.FORCE_DT

INIT_SCALE_EXTERNAL = 35.108171892561295
INIT_SCALE_INTERNAL = 3.7069009459652895e-07 * 5

def load_gt():
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            key = (int(row['activated_magnets']), int(row['duration_s']), int(row['agent_count']))
            gt[key] = float(row['final_left_fraction'])
    return gt

def build_fitting_set(gt):
    agents = [30, 50, 100, 150, 200]
    conds = [(1, d, a) for d in range(1, 6) for a in agents]
    conds += [(3, 1, a) for a in agents]
    return [(m, d, a, gt[(m, d, a)]) for (m, d, a) in conds]

def _eval_one(args):
    m, d, a, gt_val, scale_ext, scale_int = args
    sim_val = sf.run_final_fraction(m, a, d, scale_ext=scale_ext, scale_int=scale_int,
                                     c_eff=C_EFF_FIXED, mass=MASS_FIXED, dt=DT_FIXED)
    err = ((sim_val - gt_val) / max(gt_val, 0.05)) ** 2
    return m, d, a, gt_val, sim_val, err

def loss_fn(log_params, fitting_set, pool, history):
    scale_ext, scale_int = 10 ** log_params[0], 10 ** log_params[1]
    args = [(m, d, a, gt_val, scale_ext, scale_int) for (m, d, a, gt_val) in fitting_set]
    t0 = time.time()
    results = pool.map(_eval_one, args)
    loss = float(np.mean([r[5] for r in results]))
    dt_eval = time.time() - t0
    history.append({'scale_ext': scale_ext, 'scale_int': scale_int, 'loss': loss, 'seconds': dt_eval})
    print(f'  eval #{len(history):3d}  SCALE_EXTERNAL={scale_ext:8.4f}  '
          f'SCALE_INTERNAL={scale_int:.5e}  loss={loss:.5f}  ({dt_eval:.0f}s)', flush=True)
    return loss

def main():
    gt = load_gt()
    fitting_set = build_fitting_set(gt)
    print(f'Fitting set: {len(fitting_set)} conditions')
    print(f'Workers: {mp.cpu_count()}')
    print(f'Initial guess (repulsion x5 vs. FINAL SIMUL): '
          f'SCALE_EXTERNAL={INIT_SCALE_EXTERNAL:.4f}  SCALE_INTERNAL={INIT_SCALE_INTERNAL:.6e}')

    history = []
    x0 = np.log10([INIT_SCALE_EXTERNAL, INIT_SCALE_INTERNAL])

    with mp.Pool(mp.cpu_count()) as pool:
        t_start = time.time()
        res = minimize(
            loss_fn, x0, args=(fitting_set, pool, history),
            method='Nelder-Mead',
            options={'maxiter': 40, 'maxfev': 40, 'xatol': 1e-3, 'fatol': 1e-4, 'adaptive': True},
        )
        total_time = time.time() - t_start

    best_scale_ext, best_scale_int = 10 ** res.x[0], 10 ** res.x[1]
    print(f'\nConverged: {res.success}  message: {res.message}')
    print(f'Best SCALE_EXTERNAL = {best_scale_ext:.4f}')
    print(f'Best SCALE_INTERNAL = {best_scale_int:.6e}')
    print(f'Best loss = {res.fun:.5f}')
    print(f'Total optimization time: {total_time:.0f}s over {len(history)} evaluations')

    out = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'method': 'scipy.optimize.minimize, Nelder-Mead, log10-parameterized',
        'note': 'FINAL SIMUL_2: same fitting set/methodology as FINAL SIMUL, but '
                'started from SCALE_INTERNAL (repulsion) x5 the previous best fit, '
                'to explore whether a stronger-repulsion region converges differently.',
        'fitting_set_size': len(fitting_set),
        'fitting_set_description': (
            'N_ACTIVE=1 all 5 durations x agent_count in {30,50,100,150,200}; '
            'N_ACTIVE=3 duration=1 x agent_count in {30,50,100,150,200} '
            '(same fitting set as FINAL_3.m / FINAL SIMUL)'),
        'initial_guess': {'SCALE_EXTERNAL': INIT_SCALE_EXTERNAL, 'SCALE_INTERNAL': INIT_SCALE_INTERNAL},
        'previous_best_for_reference': {'SCALE_EXTERNAL': 35.108171892561295,
                                         'SCALE_INTERNAL': 3.7069009459652895e-07,
                                         'best_loss': 0.09043617418895389},
        'fixed_params': {'C_EFF': C_EFF_FIXED, 'MASS_DISK': MASS_FIXED, 'DT': DT_FIXED},
        'best_params': {'SCALE_EXTERNAL': best_scale_ext, 'SCALE_INTERNAL': best_scale_int},
        'best_loss': res.fun,
        'converged': bool(res.success),
        'message': str(res.message),
        'n_evaluations': len(history),
        'total_seconds': total_time,
        'history': history,
    }
    out_path = os.path.join(OUT_DIR, 'force_model_fit_result.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', out_path)

if __name__ == '__main__':
    main()
