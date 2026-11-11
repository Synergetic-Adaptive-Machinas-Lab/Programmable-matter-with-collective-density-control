import os
import csv
import json
import time
import multiprocessing as mp
import numpy as np

import sim_trail_force as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_CSV = os.path.join(BASE_DIR, 'ground_truth_left_behind_final.csv')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIT_JSON = os.path.join(OUT_DIR, 'force_model_fit_result.json')

MAG_LIST = [1, 3, 7]
AGENT_LIST = list(range(10, 201, 10))
DURATION_LIST = [1, 2, 3, 4, 5]

def load_params():
    if os.path.isfile(FIT_JSON):
        with open(FIT_JSON) as f:
            fit = json.load(f)
        p = fit['best_params']
        print(f'Using calibrated params from {FIT_JSON} (dated {fit["date"]}): '
              f'SCALE_EXTERNAL={p["SCALE_EXTERNAL"]:.4f}  SCALE_INTERNAL={p["SCALE_INTERNAL"]:.6e}')
        return p['SCALE_EXTERNAL'], p['SCALE_INTERNAL']
    print('WARNING: force_model_fit_result.json not found -- using initial (uncalibrated) guess')
    return sf.FORCE_SCALE_EXTERNAL, sf.FORCE_SCALE_INTERNAL

def load_gt():
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            key = (int(row['activated_magnets']), int(row['duration_s']), int(row['agent_count']))
            gt[key] = float(row['final_left_fraction'])
    return gt

def _eval_one(args):
    m, d, a, gt_val, scale_ext, scale_int = args
    sim_val = sf.run_final_fraction(m, a, d, scale_ext=scale_ext, scale_int=scale_int)
    return m, d, a, sim_val, gt_val

def main():
    scale_ext, scale_int = load_params()
    gt = load_gt()

    conds = [(m, d, a, gt.get((m, d, a), float('nan')), scale_ext, scale_int)
              for m in MAG_LIST for a in AGENT_LIST for d in DURATION_LIST]
    print(f'Running full sweep: {len(conds)} conditions on {mp.cpu_count()} cores...')

    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_eval_one, conds, chunksize=1)):
            results.append(r)
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(conds)}  ({time.time()-t0:.0f}s elapsed)', flush=True)
    print(f'Done in {time.time()-t0:.0f}s')

    out_csv = os.path.join(OUT_DIR, 'sim_vs_gt_300conditions_force.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['activated_magnets', 'duration_s', 'agent_count',
                    'sim_final_left_fraction', 'gt_final_left_fraction'])
        for m, d, a, sim_val, gt_val in sorted(results, key=lambda r: (r[0], r[2], r[1])):
            w.writerow([m, d, a, sim_val, gt_val])
    print('saved', out_csv)

    sim_arr = np.array([r[3] for r in results])
    gt_arr = np.array([r[4] for r in results])
    valid = ~np.isnan(gt_arr)
    rmse = np.sqrt(np.mean((sim_arr[valid] - gt_arr[valid]) ** 2))
    mae = np.mean(np.abs(sim_arr[valid] - gt_arr[valid]))
    print(f'RMSE = {rmse:.4f}   MAE = {mae:.4f}   (over {valid.sum()} conditions with GT)')

    summary_path = os.path.join(OUT_DIR, 'force_model_full_sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'scale_external': scale_ext, 'scale_internal': scale_int,
                    'rmse': float(rmse), 'mae': float(mae), 'n_conditions': int(valid.sum()),
                    'total_seconds': time.time() - t0}, f, indent=2)
    print('saved', summary_path)

if __name__ == '__main__':
    main()
