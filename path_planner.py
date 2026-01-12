"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

"""
Path planners: A*, Dijkstra, RRT, RRT*
Each planner returns:
    path, raw_path, runtime, nodes_before, nodes_after, len_before, len_after
"""

import math
import heapq
import numpy as np
import random
import time

# ---------------------- Base Class ---------------------- #
class PathPlannerBase:
    """Base class for path planners"""
    def plan(self, start, goal):
        raise NotImplementedError("Must implement plan method")

# ---------------------- Utility function ---------------------- #
def path_length(path):
    if len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

# ---------------------- A* Planner ---------------------- #
class AStarPlanner(PathPlannerBase):
    """A* path planner on a 3D regular grid with optional shortcut pruning."""
    def __init__(self, env, resolution=0.5, max_iters=1_000_000, use_shortcut=True):
        self.env = env
        self.resolution = float(resolution)
        self.max_iters = max_iters
        self.use_shortcut = use_shortcut

        self.x_max = env.env_width
        self.y_max = env.env_length
        self.z_max = env.env_height
        self.nx = int(math.floor(self.x_max / self.resolution)) + 1
        self.ny = int(math.floor(self.y_max / self.resolution)) + 1
        self.nz = int(math.floor(self.z_max / self.resolution)) + 1

    def _coord_to_index(self, coord):
        x, y, z = coord
        return (int(round(x/self.resolution)),
                int(round(y/self.resolution)),
                int(round(z/self.resolution)))

    def _index_to_coord(self, idx):
        ix, iy, iz = idx
        return (ix*self.resolution, iy*self.resolution, iz*self.resolution)

    def _in_bounds_index(self, idx):
        ix, iy, iz = idx
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    def _is_valid_index(self, idx):
        if not self._in_bounds_index(idx):
            return False
        coord = self._index_to_coord(idx)
        return not (self.env.is_outside(coord) or self.env.is_collide(coord))

    def _heuristic(self, a_idx, b_idx):
        ax, ay, az = self._index_to_coord(a_idx)
        bx, by, bz = self._index_to_coord(b_idx)
        return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)

    def _neighbors(self, idx):
        ix, iy, iz = idx
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if dx==0 and dy==0 and dz==0:
                        continue
                    yield (ix+dx, iy+dy, iz+dz)

    def _line_collision_free(self, a, b, step=0.2):
        a = np.array(a)
        b = np.array(b)
        dist = np.linalg.norm(b-a)
        if dist == 0:
            return not self.env.is_collide(a)
        n = max(2, int(np.ceil(dist/step)))
        for i in range(n+1):
            pt = a + (b-a)*i/n
            if self.env.is_outside(pt) or self.env.is_collide(pt):
                return False
        return True

    def _reconstruct_path(self, came_from, current):
        path_idx = [current]
        while current in came_from:
            current = came_from[current]
            path_idx.append(current)
        path_idx.reverse()
        return [self._index_to_coord(idx) for idx in path_idx]

    def _shortcut_path(self, coords):
        if len(coords) == 0:
            return coords
        pruned = [coords[0]]
        i = 0
        N = len(coords)
        while i < N-1:
            j = N-1
            while j > i+1:
                if self._line_collision_free(coords[i], coords[j], step=self.resolution/2):
                    break
                j -= 1
            pruned.append(coords[j])
            i = j
        return pruned

    def plan(self, start, goal):
        start_time = time.time()
        start_idx = self._coord_to_index(start)
        goal_idx = self._coord_to_index(goal)

        if self.env.is_outside(start) or self.env.is_collide(start):
            raise RuntimeError("Start position invalid")
        if self.env.is_outside(goal) or self.env.is_collide(goal):
            raise RuntimeError("Goal position invalid")

        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        came_from = {}
        g_score = {start_idx: 0.0}
        expanded = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                raw_path = self._reconstruct_path(came_from, current)
                nodes_before = len(raw_path)
                len_before = path_length(raw_path)
                path = self._shortcut_path(raw_path) if self.use_shortcut else raw_path
                nodes_after = len(path)
                len_after = path_length(path)
                runtime = time.time() - start_time
                return np.array(path), np.array(raw_path), runtime, nodes_before, nodes_after, len_before, len_after

            expanded += 1
            if expanded > self.max_iters:
                break

            for nb in self._neighbors(current):
                if not self._is_valid_index(nb):
                    continue
                tentative_g = g_score[current] + self._heuristic(current, nb)
                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal_idx)
                    heapq.heappush(open_set, (f, nb))

        raise RuntimeError(f"A* failed after {expanded} expansions")


# ---------------------- Dijkstra Planner ---------------------- #
class DijkstraPlanner(AStarPlanner):
    def _heuristic(self, a_idx, b_idx):
        return 0.0


# ---------------------- RRT Planner ---------------------- #
class RRTPlanner(PathPlannerBase):
    class Node:
        def __init__(self, coord, parent=None):
            self.coord = coord
            self.parent = parent

    def __init__(self, env, max_iters=1_000_000, step_size=0.5,
                 goal_sample_rate=0.1, use_shortcut=True):
        self.env = env
        self.max_iters = max_iters
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.use_shortcut = use_shortcut

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a)-np.array(b))

    def _steer(self, from_node, to_coord):
        vec = np.array(to_coord)-np.array(from_node.coord)
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            return to_coord
        vec = vec/dist*self.step_size
        return tuple(np.array(from_node.coord)+vec)

    def _line_collision_free(self, p1, p2, step=0.1):
        dist = self._distance(p1, p2)
        n = max(2,int(np.ceil(dist/step)))
        for i in range(n+1):
            pt = np.array(p1)*(1-i/n)+np.array(p2)*(i/n)
            if self.env.is_outside(pt) or self.env.is_collide(pt):
                return False
        return True

    def _shortcut_path(self, path):
        if isinstance(path, np.ndarray):
            path = path.tolist()
        if len(path)<=2:
            return path
        pruned = [path[0]]
        i = 0
        while i < len(path)-1:
            j = len(path)-1
            while j>i+1:
                if self._line_collision_free(path[i], path[j]):
                    break
                j -= 1
            pruned.append(path[j])
            i = j
        return pruned

    def plan(self, start, goal):
        start_time = time.time()
        start_node = self.Node(start)
        nodes = [start_node]

        for _ in range(self.max_iters):
            rnd = goal if random.random()<self.goal_sample_rate else (
                random.uniform(0,self.env.env_width),
                random.uniform(0,self.env.env_length),
                random.uniform(0,self.env.env_height)
            )
            nearest = min(nodes, key=lambda n:self._distance(n.coord, rnd))
            new_coord = self._steer(nearest, rnd)
            if not self._line_collision_free(nearest.coord, new_coord):
                continue

            new_node = self.Node(new_coord, parent=nearest)
            nodes.append(new_node)

            if self._distance(new_coord, goal)<self.step_size:
                goal_node = self.Node(goal, parent=new_node)
                nodes.append(goal_node)

                raw_path = []
                node = goal_node
                while node:
                    raw_path.append(node.coord)
                    node = node.parent
                raw_path.reverse()

                nodes_before = len(raw_path)
                len_before = path_length(raw_path)
                path = self._shortcut_path(raw_path) if self.use_shortcut else raw_path
                nodes_after = len(path)
                len_after = path_length(path)
                runtime = time.time() - start_time
                return np.array(path), np.array(raw_path), runtime, nodes_before, nodes_after, len_before, len_after

        raise RuntimeError("RRT failed to find a path")


# ---------------------- RRT* Planner ---------------------- #
class RRTStarPlanner(PathPlannerBase):
    class Node:
        def __init__(self, coord, parent=None, cost=0.0):
            self.coord = coord
            self.parent = parent
            self.cost = cost

    def __init__(self, env, max_iters=1_000_000, step_size=0.5, search_radius=1.0,
                 goal_sample_rate=0.1, use_shortcut=True):
        self.env = env
        self.max_iters = max_iters
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.use_shortcut = use_shortcut

    def _distance(self,a,b):
        return np.linalg.norm(np.array(a)-np.array(b))

    def _steer(self, from_node, to_coord):
        vec = np.array(to_coord)-np.array(from_node.coord)
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            return to_coord
        vec = vec/dist*self.step_size
        return tuple(np.array(from_node.coord)+vec)

    def _near(self, nodes, coord):
        return [n for n in nodes if self._distance(n.coord, coord)<=self.search_radius]

    def _line_collision_free(self, p1, p2, step=0.1):
        dist = self._distance(p1,p2)
        n = int(np.ceil(dist/step))
        for i in range(n+1):
            pt = np.array(p1)*(1-i/n)+np.array(p2)*(i/n)
            if self.env.is_outside(pt) or self.env.is_collide(pt):
                return False
        return True

    def _shortcut_path(self, path):
        if isinstance(path,np.ndarray):
            path = path.tolist()
        if len(path)<=2:
            return path
        pruned = [path[0]]
        i = 0
        N = len(path)
        while i<N-1:
            j = N-1
            while j>i+1:
                if self._line_collision_free(path[i], path[j]):
                    break
                j -= 1
            pruned.append(path[j])
            i = j
        return pruned

    def plan(self,start,goal):
        start_time = time.time()
        start_node = self.Node(start)
        goal_node = self.Node(goal)
        nodes = [start_node]

        for _ in range(self.max_iters):
            rnd = goal if random.random()<self.goal_sample_rate else (
                random.uniform(0,self.env.env_width),
                random.uniform(0,self.env.env_length),
                random.uniform(0,self.env.env_height)
            )
            nearest = min(nodes,key=lambda n:self._distance(n.coord,rnd))
            new_coord = self._steer(nearest,rnd)
            if not self._line_collision_free(nearest.coord,new_coord):
                continue

            neighbors = self._near(nodes,new_coord)
            min_cost = nearest.cost+self._distance(nearest.coord,new_coord)
            best_parent = nearest
            for n in neighbors:
                cost = n.cost+self._distance(n.coord,new_coord)
                if cost<min_cost and self._line_collision_free(n.coord,new_coord):
                    min_cost = cost
                    best_parent = n

            new_node = self.Node(new_coord,parent=best_parent,cost=min_cost)
            nodes.append(new_node)

            # rewire
            for n in neighbors:
                new_cost = new_node.cost+self._distance(new_node.coord,n.coord)
                if new_cost < n.cost and self._line_collision_free(new_node.coord,n.coord):
                    n.parent = new_node
                    n.cost = new_cost

            if self._distance(new_coord,goal)<self.step_size:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost+self._distance(new_node.coord,goal)
                nodes.append(goal_node)

                raw_path = []
                node = goal_node
                while node:
                    raw_path.append(node.coord)
                    node = node.parent
                raw_path.reverse()

                nodes_before = len(raw_path)
                len_before = path_length(raw_path)
                path = self._shortcut_path(raw_path) if self.use_shortcut else raw_path
                nodes_after = len(path)
                len_after = path_length(path)
                runtime = time.time() - start_time
                return np.array(path), np.array(raw_path), runtime, nodes_before, nodes_after, len_before, len_after

        raise RuntimeError("RRT* failed to find a path")
