import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planner"))
from play_frames import render

HERE = os.path.dirname(os.path.abspath(__file__))
for name in ["hammer", "wrench"]:
    render(os.path.join(HERE, "frames", name), os.path.join(HERE, "outputs", name), fps=4)
