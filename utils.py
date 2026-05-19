import math
import random
from pathlib import Path
import matplotlib.pyplot as plt


PLOT_SCALE = 50.0


class Obstacle:
    def __init__(self, x, y, r):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)

    def contains(self, point, margin=0.0):
        px, py = point
        return math.hypot(px - self.x, py - self.y) <= self.r + margin


def create_obstacles(width=80, height=80, seed=19):
    rng = random.Random(seed)
    sx = width / 80
    sy = height / 80
    sr = min(sx, sy)
    anchors = [
        (22, 31, 4.2),
        (31, 21, 3.8),
        (34, 46, 4.5),
        (46, 35, 5.0),
        (40, 40, 3.4),
        (51, 57, 4.1),
        (61, 48, 3.7),
        (35, 61, 3.3),
        (64, 35, 3.8),
    ]

    obstacles = []
    for x, y, r in anchors:
        obstacles.append(Obstacle(
            x * sx + rng.uniform(-3.5, 3.5) * sx,
            y * sy + rng.uniform(-3.5, 3.5) * sy,
            r * sr + rng.uniform(-0.6, 0.8) * sr,
        ))

    return obstacles


def in_bounds(point, width, height):
    x, y = point
    return 0 <= x <= width and 0 <= y <= height


def is_collision(point, obstacles, margin=0.8):
    return any(obs.contains(point, margin) for obs in obstacles)


def segment_collision(p1, p2, obstacles, margin=0.8, step=0.5):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    n = max(2, int(dist / step))

    for i in range(n + 1):
        t = i / n
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        if is_collision((x, y), obstacles, margin):
            return True

    return False


def path_length(path):
    if path is None or len(path) < 2:
        return float("inf")
    return sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )


def scale_point(point):
    return (point[0] * PLOT_SCALE, point[1] * PLOT_SCALE)


def scale_path(path):
    return [scale_point(point) for point in path]


def smooth_display_path(path, passes=1):
    if path is None or len(path) < 3:
        return path

    smoothed = list(path)
    for _ in range(passes):
        next_path = [smoothed[0]]
        for p1, p2 in zip(smoothed[:-1], smoothed[1:]):
            q = (0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1])
            r = (0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1])
            next_path.extend([q, r])
        next_path.append(smoothed[-1])
        smoothed = next_path

    return smoothed


def plot_environment(ax, width, height, starts, goals, obstacles):
    plot_width = width * PLOT_SCALE
    plot_height = height * PLOT_SCALE

    ax.set_xlim(0, plot_width)
    ax.set_ylim(0, plot_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.7)

    for obs in obstacles:
        circle = plt.Circle(
            (obs.x * PLOT_SCALE, obs.y * PLOT_SCALE),
            obs.r * PLOT_SCALE,
            fill=False,
            color="#b00000",
            linewidth=1.4,
            alpha=0.75,
        )
        ax.add_patch(circle)

    unique_starts = list(dict.fromkeys(starts))
    unique_goals = list(dict.fromkeys(goals))

    for i, start in enumerate(unique_starts):
        x, y = scale_point(start)
        ax.scatter(x, y, marker="o", s=22, facecolors="none", edgecolors="black", linewidths=0.8)
        ax.text(x + 35, y + 35, f"{i+1}", fontsize=7, color="black")

    for i, goal in enumerate(unique_goals):
        x, y = scale_point(goal)
        ax.scatter(x, y, marker="o", s=26, c="lime", edgecolors="green", linewidths=0.5, zorder=5)
        ax.text(x + 35, y + 35, f"{i+1}", fontsize=7, color="black")


def save_swarm_figure(
    paths,
    title,
    filename,
    width,
    height,
    starts,
    goals,
    obstacles,
    leader_path=None,
    direct_path=None,
    crash_points=None,
    uav_collision_points=None,
):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plot_environment(ax, width, height, starts, goals, obstacles)

    if direct_path and len(direct_path) > 1:
        scaled = scale_path(direct_path)
        xs = [p[0] for p in scaled]
        ys = [p[1] for p in scaled]
        ax.plot(xs, ys, linestyle=":", color="0.55", linewidth=1.0)

    if leader_path and len(leader_path) > 1:
        display_path = smooth_display_path(leader_path, passes=1)
        scaled = scale_path(display_path)
        xs = [p[0] for p in scaled]
        ys = [p[1] for p in scaled]
        ax.plot(xs, ys, linestyle="--", color="black", linewidth=1.1)

    for i, path in enumerate(paths):
        if path and len(path) > 1:
            display_path = smooth_display_path(path, passes=1)
            scaled = scale_path(display_path)
            xs = [p[0] for p in scaled]
            ys = [p[1] for p in scaled]
            line = ax.plot(xs, ys, linewidth=0.9, alpha=0.92)[0]
            color = line.get_color()
            start_x, start_y = scale_point(path[0])
            end_x, end_y = scale_point(path[-1])
            ax.scatter(start_x, start_y, s=20, facecolors="none", edgecolors=color, linewidths=1.3)
            ax.scatter(end_x, end_y, marker=">", s=28, c=color)
            ax.text(end_x + 35, end_y + 35, f"{i+1}", fontsize=7, color="black")
            if crash_points and crash_points[i] is not None:
                crash_x, crash_y = scale_point(crash_points[i])
                ax.scatter(crash_x, crash_y, marker="x", s=55, c="red", linewidths=1.5)
                ax.text(crash_x + 35, crash_y + 35, f"Crash{i+1}", color="red", fontsize=7)
            if uav_collision_points and uav_collision_points[i] is not None:
                hit_x, hit_y = scale_point(uav_collision_points[i])
                ax.scatter(hit_x, hit_y, marker="x", s=55, c="black", linewidths=1.5)
                ax.text(hit_x + 35, hit_y - 80, f"Hit{i+1}", color="black", fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
