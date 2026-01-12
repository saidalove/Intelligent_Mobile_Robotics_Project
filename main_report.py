import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from flight_environment import FlightEnvironment
from path_planner import (
    DijkstraPlanner,
    AStarPlanner,
    RRTPlanner,
    RRTStarPlanner
)

# ================== 全局字体设置 ==================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10

# ================== 保存路径 ==================
save_dir = "img/report"
os.makedirs(save_dir, exist_ok=True)

start = (1, 2, 0.5)
goal  = (18, 18, 3)

obs_list = [50, 100, 150]
algorithms = [
    ("Dijkstra", DijkstraPlanner),
    ("AStar", AStarPlanner),
    ("RRT", RRTPlanner),
    ("RRTStar", RRTStarPlanner)
]

# 表格表头
table_data = [["Algorithm", "Obstacles", "Shortcut", "Time(s)", "Nodes", "Path Len(m)"]]

# ================== 实验循环 ==================
for obs_num in obs_list:
    env = FlightEnvironment(obs_num, reserved_points=[start, goal])

    for name, PlannerClass in algorithms:
        print(f"Running {name} | Obstacles={obs_num}")

        planner = PlannerClass(env, use_shortcut=True)

        # ------------------ 统一 plan() 返回 ------------------
        # path, raw_path, runtime, nodes_before, nodes_after, len_before, len_after
        path, raw_path, runtime, nodes_before, nodes_after, len_before, len_after = planner.plan(start, goal)

        # ================== 保存 shortcut 前后的路径图 ==================
        save_paths = [
            ("NoShortcut", raw_path, nodes_before, len_before),
            ("Shortcut", path, nodes_after, len_after)
        ]

        for label, plot_path, nodes, path_len in save_paths:
            fig_name = f"{name}_obs{obs_num}_{label}.png"
            fig_path = os.path.join(save_dir, fig_name)

            # 绘制路径
            fig=env.plot_cylinders(plot_path)

            # 图题在下方居中
            fig_text = f"{name} Path ({label}) with {obs_num} Obstacles\nLength: {path_len:.2f} m, Nodes: {nodes}"
            fig.text(0.5, 0.01, fig_text, ha='center', fontname='Times New Roman', fontsize=10)

            # 保存
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved path figure: {fig_path}")

        # ========== 表格：两行（无 / 有 shortcut） ==========
        table_data.append([
            name, obs_num, "No",
            f"{runtime:.2f}", nodes_before, f"{len_before:.2f}"
        ])
        table_data.append([
            name, obs_num, "Yes",
            f"{runtime:.2f}", nodes_after, f"{len_after:.2f}"
        ])

# ================== 保存表格 ==================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)

table_path = os.path.join(save_dir, "path_planning_report.png")
plt.savefig(table_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved report table: {table_path}")
