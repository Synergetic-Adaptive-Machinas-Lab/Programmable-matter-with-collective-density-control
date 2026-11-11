import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planner"))
from run_demo import run

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = [("maze_sourceToGoal", 13), ("maze_merge", 12)]

for name, rho0 in DEMOS:
    run(os.path.join(HERE, "inputs", name + ".json"), rho0, os.path.join(HERE, "outputs", name), fps=2, video="--no_video" not in sys.argv)
