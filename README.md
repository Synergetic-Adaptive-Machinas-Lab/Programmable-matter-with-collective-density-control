# 1024-magnet microrobot platform: code

| Folder | Contents |
|---|---|
| `firmware/` | Pico firmware and PWM gamma curve |
| `planner/` | Assignment + A* path planner, cluster density |
| `maze_demo/` | Maze transport demo (`maze_merge`) |
| `SWARM_demo/` | Shape-to-shape transform demo (S-W-A-R-M) |
| `simulation/` | Disk physics model and swarm simulation |

## Run

```
pip install -r requirements.txt
python overview.py
```

Needs `ffmpeg`. Writes `overview.png` and `figures/`.

## Figures

![overview](overview.png)

**Planner.** Start groups (blue) assigned to goal cells (orange); lines are the A* paths around the walls.

![planner](figures/planner.png)

**Shape transform.** Magnet pattern going from S to W, and how many magnets are on in each frame.

![swarm demo](figures/swarm_demo.png)

**Control UI.** Browser interface that drives the 1024 magnets.

![control UI](ui_screenshot.png)

**Transport simulation.** 100 disks towed left to right by a 7-magnet cluster; colour is time.

![transport](figures/transport.png)

**Swarm simulation.** 3200 simulated disks settled into the S and W patterns (frames from `simulation/SWARM/swarm_sim_S-W-A.mp4`).

![swarm simulation](figures/sim_swarm.png)
