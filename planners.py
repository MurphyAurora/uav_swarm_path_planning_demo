import math
import heapq
import random
from utils import is_collision, segment_collision, in_bounds


def astar(start, goal, obstacles, width=60, height=60, resolution=1.0):
    start_node = (round(start[0] / resolution), round(start[1] / resolution))
    goal_node = (round(goal[0] / resolution), round(goal[1] / resolution))

    def to_world(node):
        return (node[0] * resolution, node[1] * resolution)

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    motions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {start_node: 0}

    max_x = int(width / resolution)
    max_y = int(height / resolution)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            path = []
            while current in came_from:
                path.append(to_world(current))
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in motions:
            nxt = (current[0] + dx, current[1] + dy)

            if not (0 <= nxt[0] <= max_x and 0 <= nxt[1] <= max_y):
                continue

            world_nxt = to_world(nxt)
            if is_collision(world_nxt, obstacles):
                continue

            move_cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + move_cost

            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f = tentative_g + heuristic(nxt, goal_node)
                heapq.heappush(open_set, (f, nxt))

    return None


class RRTNode:
    def __init__(self, x, y, parent=None, cost=0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost)

    @property
    def point(self):
        return (self.x, self.y)


def nearest_node(nodes, point):
    return min(nodes, key=lambda n: math.hypot(n.x - point[0], n.y - point[1]))


def steer(from_node, to_point, step_size):
    dx = to_point[0] - from_node.x
    dy = to_point[1] - from_node.y
    dist = math.hypot(dx, dy)

    if dist <= step_size:
        return RRTNode(to_point[0], to_point[1])

    theta = math.atan2(dy, dx)
    return RRTNode(
        from_node.x + step_size * math.cos(theta),
        from_node.y + step_size * math.sin(theta)
    )


def extract_path(node):
    path = []
    while node is not None:
        path.append(node.point)
        node = node.parent
    return path[::-1]


def rrt(start, goal, obstacles, width=60, height=60, max_iter=6000, step_size=3.0, goal_sample_rate=0.25):
    start_node = RRTNode(start[0], start[1])
    nodes = [start_node]

    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (random.uniform(0, width), random.uniform(0, height))

        nearest = nearest_node(nodes, sample)
        new_node = steer(nearest, sample, step_size)

        if not in_bounds(new_node.point, width, height):
            continue
        if is_collision(new_node.point, obstacles):
            continue
        if segment_collision(nearest.point, new_node.point, obstacles):
            continue

        new_node.parent = nearest
        new_node.cost = nearest.cost + math.hypot(new_node.x - nearest.x, new_node.y - nearest.y)
        nodes.append(new_node)

        if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) < step_size:
            if not segment_collision(new_node.point, goal, obstacles):
                goal_node = RRTNode(goal[0], goal[1], parent=new_node)
                return extract_path(goal_node)

    return None


def rrt_star(start, goal, obstacles, width=60, height=60, max_iter=8000, step_size=3.0, search_radius=9.0, goal_sample_rate=0.25):
    start_node = RRTNode(start[0], start[1])
    nodes = [start_node]

    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (random.uniform(0, width), random.uniform(0, height))

        nearest = nearest_node(nodes, sample)
        new_node = steer(nearest, sample, step_size)

        if not in_bounds(new_node.point, width, height):
            continue
        if is_collision(new_node.point, obstacles):
            continue
        if segment_collision(nearest.point, new_node.point, obstacles):
            continue

        near_nodes = [
            n for n in nodes
            if math.hypot(n.x - new_node.x, n.y - new_node.y) <= search_radius
        ]

        best_parent = nearest
        best_cost = nearest.cost + math.hypot(new_node.x - nearest.x, new_node.y - nearest.y)

        for n in near_nodes:
            if not segment_collision(n.point, new_node.point, obstacles):
                cost = n.cost + math.hypot(new_node.x - n.x, new_node.y - n.y)
                if cost < best_cost:
                    best_parent = n
                    best_cost = cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        nodes.append(new_node)

        for n in near_nodes:
            if n is best_parent:
                continue
            if not segment_collision(new_node.point, n.point, obstacles):
                new_cost = new_node.cost + math.hypot(new_node.x - n.x, new_node.y - n.y)
                if new_cost < n.cost:
                    n.parent = new_node
                    n.cost = new_cost

        if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) < step_size:
            if not segment_collision(new_node.point, goal, obstacles):
                goal_node = RRTNode(goal[0], goal[1], parent=new_node)
                return extract_path(goal_node)

    return None
