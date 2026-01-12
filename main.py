import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from flight_environment import FlightEnvironment
from path_planner import RRTStarPlanner
from trajectory_generator import QuinticTrajectoryGenerator

# ================== 全局字体设置 ==================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

# ================== 起点/终点 ==================
start = (1, 2, 0)
goal = (18, 18, 3)

# ================== 创建环境 ==================
env = FlightEnvironment(50, reserved_points=[start, goal])

# ================== 保存路径的文件夹 ==================
save_dir = "img/result"
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

planner = RRTStarPlanner(env, step_size=0.5, goal_sample_rate=0.1, use_shortcut=True)

path, raw_path, runtime, nodes_before, nodes_after, len_before, len_after = planner.plan(start, goal)

# --------------------------------------------------------------------------------------------------- #

fig = env.plot_cylinders(path)

# 添加图题在下方居中
fig_text = f"RRT* Path (Shortcut) with 50 Obstacles\nLength: {len_after:.2f} m, Nodes: {nodes_after}"
fig.text(0.5, 0.01, fig_text, ha='center', fontname='Times New Roman', fontsize=10)

# 保存图像
plt.show()
fig_path = os.path.join(save_dir, "RRTStar_path_shortcut.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved path figure: {fig_path}")

# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

traj_gen = QuinticTrajectoryGenerator(path)
t, x, y, z = traj_gen.generate_trajectory()

# 绘制轨迹
fig = traj_gen.plot_trajectory()

# 添加图题在下方居中
fig_text = f"RRT* Trajectory (Shortcut) with 50 Obstacles"
fig.text(0.5, 0.01, fig_text, ha='center', fontname='Times New Roman', fontsize=12)

# 保存图像
plt.show()
fig_path = os.path.join(save_dir, "RRTStar_trajectory_shortcut.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved path figure: {fig_path}")

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
