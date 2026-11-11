import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planner"))
from run_demo import run

HERE = os.path.dirname(os.path.abspath(__file__))
run(os.path.join(HERE, "inputs", "S-W-A-R-M.json"), 16, os.path.join(HERE, "outputs", "S-W-A-R-M"), fps=2, video="--no_video" not in sys.argv)
