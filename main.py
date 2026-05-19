import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from planners import astar, rrt, rrt_star
from swarm import interpolate_path_by_distance, summarize_swarm_result
from utils import Obstacle, create_obstacles, path_length, save_swarm_figure, segment_collision


def inflate_obstacles(obstacles, clearance):
    return [Obstacle(obs.x, obs.y, obs.r + clearance) for obs in obstacles]


def formation_scale(distance, total_distance):
    return 1.0


def path_distances(path):
    distances = [0.0]

    for p1, p2 in zip(path[:-1], path[1:]):
        distances.append(distances[-1] + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    return distances


def smooth_path(path, obstacles):
    if path is None or len(path) < 3:
        return path

    smoothed = [path[0]]
    current = 0

    while current < len(path) - 1:
        next_index = len(path) - 1
        while next_index > current + 1:
            if not segment_collision(path[current], path[next_index], obstacles):
                break
            next_index -= 1

        smoothed.append(path[next_index])
        current = next_index

    return smoothed


def local_frame(path, index):
    if index < len(path) - 1:
        p1 = path[index]
        p2 = path[index + 1]
    else:
        p1 = path[index - 1]
        p2 = path[index]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return (1.0, 0.0), (0.0, 1.0)

    tangent = (dx / length, dy / length)
    normal = (-tangent[1], tangent[0])
    return tangent, normal


def apply_local_offset(point, tangent, normal, offset, scale):
    along_track, cross_track = offset
    return (
        point[0] + scale * (along_track * tangent[0] + cross_track * normal[0]),
        point[1] + scale * (along_track * tangent[1] + cross_track * normal[1]),
    )


def build_virtual_leader_formation_paths(leader_path, formation_offsets):
    if leader_path is None or len(leader_path) < 2:
        return [None for _ in formation_offsets]

    total_distance = path_length(leader_path)
    distances = path_distances(leader_path)
    paths = []

    for offset in formation_offsets:
        path = []
        for i, (point, distance) in enumerate(zip(leader_path, distances)):
            scale = formation_scale(distance, total_distance)
            tangent, normal = local_frame(leader_path, i)
            path.append(apply_local_offset(point, tangent, normal, offset, scale))
        paths.append(path)

    return paths


def first_collision_point(p1, p2, obstacles, margin=0.0, step=0.25):
    distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    samples = max(2, int(distance / step))

    for sample in range(1, samples + 1):
        t = sample / samples
        point = (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
        )
        if any(obs.contains(point, margin) for obs in obstacles):
            return point

    return None


def stop_paths_at_obstacle_collision(paths, obstacles):
    checked_paths = []
    crash_points = []

    for path in paths:
        if path is None or len(path) < 2:
            checked_paths.append(path)
            crash_points.append(None)
            continue

        stopped_path = [path[0]]
        crash_point = None

        for p1, p2 in zip(path[:-1], path[1:]):
            collision_point = first_collision_point(p1, p2, obstacles)
            if collision_point is not None:
                stopped_path.append(collision_point)
                crash_point = collision_point
                break
            stopped_path.append(p2)

        checked_paths.append(stopped_path)
        crash_points.append(crash_point)

    return checked_paths, crash_points


def truncate_path_by_distance(path, stop_distance):
    if path is None or len(path) < 2:
        return path

    truncated = [path[0]]
    traveled = 0.0

    for p1, p2 in zip(path[:-1], path[1:]):
        segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if traveled + segment_length >= stop_distance:
            ratio = (stop_distance - traveled) / segment_length if segment_length > 1e-9 else 0.0
            truncated.append((
                p1[0] + ratio * (p2[0] - p1[0]),
                p1[1] + ratio * (p2[1] - p1[1]),
            ))
            return truncated

        truncated.append(p2)
        traveled += segment_length

    return truncated


def stop_paths_at_uav_collisions(
    paths,
    collision_distance=1.8,
    sample_step=1.0,
    ignore_start_distance=0.0,
    ignore_goal_distance=0.0,
):
    sampled_paths = [interpolate_path_by_distance(path, step=sample_step) for path in paths]
    valid_lengths = [len(path) for path in sampled_paths if len(path) > 0]

    if not valid_lengths:
        return paths, [], []

    max_steps = max(valid_lengths)
    first_collision = [None for _ in paths]
    collision_events = []

    for step_index in range(max_steps):
        active_positions = []

        for uav_index, sampled in enumerate(sampled_paths):
            if sampled:
                position = sampled[min(step_index, len(sampled) - 1)]
                active_positions.append((uav_index, position))

        distance_along_path = step_index * sample_step
        if distance_along_path < ignore_start_distance:
            continue

        for i in range(len(active_positions)):
            uav_i, p1 = active_positions[i]
            path_i_length = path_length(paths[uav_i])
            if path_i_length - distance_along_path < ignore_goal_distance:
                continue

            for j in range(i + 1, len(active_positions)):
                uav_j, p2 = active_positions[j]
                path_j_length = path_length(paths[uav_j])
                if path_j_length - distance_along_path < ignore_goal_distance:
                    continue

                distance = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if distance >= collision_distance:
                    continue

                collision_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                collision_events.append((uav_i, uav_j, collision_point))

                if first_collision[uav_i] is None:
                    first_collision[uav_i] = distance_along_path
                if first_collision[uav_j] is None:
                    first_collision[uav_j] = distance_along_path

        if any(distance is not None for distance in first_collision):
            break

    stopped_paths = []
    uav_collision_points = [None for _ in paths]

    for uav_index, path in enumerate(paths):
        stop_distance = first_collision[uav_index]
        if stop_distance is None:
            stopped_paths.append(path)
            continue

        stopped_paths.append(truncate_path_by_distance(path, stop_distance))
        if stopped_paths[-1]:
            uav_collision_points[uav_index] = stopped_paths[-1][-1]

    return stopped_paths, uav_collision_points, collision_events


def merge_segment(base_path, segment):
    if segment is None or len(segment) < 2:
        return None

    if base_path:
        base_path.extend(segment[1:])
    else:
        base_path.extend(segment)

    return base_path


def point_in_obstacle(point, obstacles):
    return any(obs.contains(point) for obs in obstacles)


def repair_path_with_local_avoidance(path, planner, obstacles, width, height, repair_clearance=0.8):
    if path is None or len(path) < 2:
        return path

    repaired = [path[0]]
    planning_obstacles = inflate_obstacles(obstacles, repair_clearance)
    index = 1

    while index < len(path):
        target = path[index]
        current = repaired[-1]

        if not segment_collision(current, target, planning_obstacles):
            repaired.append(target)
            index += 1
            continue

        detour = None
        rejoin_index = None

        for candidate_index in range(index, min(len(path), index + 10)):
            candidate = path[candidate_index]
            if point_in_obstacle(candidate, planning_obstacles):
                continue

            detour = astar(current, candidate, planning_obstacles, width=width, height=height)
            detour = smooth_path(detour, planning_obstacles)
            if detour is not None:
                rejoin_index = candidate_index
                break

        if merge_segment(repaired, detour) is None:
            final_target = path[-1]
            detour = astar(current, final_target, planning_obstacles, width=width, height=height)
            detour = smooth_path(detour, planning_obstacles)

            if merge_segment(repaired, detour) is None:
                repaired.append(target)
                index += 1
            else:
                break
        else:
            index = rejoin_index + 1

    return repaired


def repair_paths_with_local_avoidance(paths, planner, obstacles, width, height):
    return [
        repair_path_with_local_avoidance(path, planner, obstacles, width, height)
        for path in paths
    ]


def plan_virtual_leader_path(planner, common_start, common_goal, obstacles, width, height, formation_clearance):
    planning_obstacles = inflate_obstacles(obstacles, formation_clearance)
    raw_path = planner(common_start, common_goal, planning_obstacles, width=width, height=height)
    return smooth_path(raw_path, planning_obstacles)


def plan_for_swarm(
    algorithm_name,
    planner,
    common_start,
    common_goal,
    formation_offsets,
    obstacles,
    width,
    height,
):
    t0 = time.perf_counter()
    leader_path = plan_virtual_leader_path(
        planner,
        common_start,
        common_goal,
        obstacles,
        width,
        height,
        formation_clearance=1.2,
    )
    paths = build_virtual_leader_formation_paths(leader_path, formation_offsets)
    paths = repair_paths_with_local_avoidance(paths, planner, obstacles, width, height)
    formation_starts = [path[0] if path else None for path in paths]
    formation_goals = [path[-1] if path else None for path in paths]
    paths, crash_points = stop_paths_at_obstacle_collision(paths, obstacles)
    paths, uav_collision_points, uav_collision_events = stop_paths_at_uav_collisions(paths)
    t1 = time.perf_counter()

    summary = summarize_swarm_result(paths, safe_distance=3.0, goals=formation_goals)
    summary["algorithm"] = algorithm_name
    summary["planning_time_sec"] = t1 - t0
    summary["num_uav"] = len(formation_offsets)
    summary["paths"] = paths
    summary["starts"] = formation_starts
    summary["goals"] = formation_goals
    summary["crash_points"] = crash_points
    summary["crash_count"] = sum(point is not None for point in crash_points)
    summary["uav_collision_points"] = uav_collision_points
    summary["uav_collision_count"] = len(uav_collision_events)
    summary["leader_path"] = leader_path
    summary["leader_path_length"] = path_length(leader_path)

    return summary


def result_filename(algorithm_name):
    name_map = {
        "A*": "astar",
        "RRT": "rrt",
        "RRT*": "rrt_star",
    }
    slug = name_map.get(
        algorithm_name,
        algorithm_name.lower().replace("*", "_star").replace("+", "_plus"),
    )
    return f"results/swarm_{slug}.png"


def cleanup_results():
    result_dir = Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    for file in result_dir.glob("swarm_*.png"):
        file.unlink()

    for filename in ["comparison.png", "metrics.csv"]:
        file = result_dir / filename
        if file.exists():
            file.unlink()


def create_paper_formation():
    return [
        (-1, 0),
        (-5, 5),
        (-5, -5),
        (-9, 0),
        (-10, 8),
        (-10, -8),
        (-14, 12),
        (-14, -12),
        (-17, 5),
        (-17, -5),
    ]


def main():
    random.seed(12)

    width, height = 120, 120
    cleanup_results()
    obstacles = create_obstacles(width=width, height=height, seed=19)

    common_start = (22, 24)
    common_goal = (108, 108)
    formation_offsets = create_paper_formation()

    planners = [
        ("A*", astar),
        ("RRT", rrt),
        ("RRT*", rrt_star),
    ]

    results = []

    for name, planner in planners:
        result = plan_for_swarm(
            name,
            planner,
            common_start,
            common_goal,
            formation_offsets,
            obstacles,
            width,
            height,
        )
        results.append(result)

        save_swarm_figure(
            result["paths"],
            f"Virtual Leader Formation Obstacle Avoidance - {name}",
            result_filename(name),
            width,
            height,
            result["starts"],
            result["goals"],
            obstacles,
            leader_path=result["leader_path"],
            direct_path=[common_start, common_goal],
            crash_points=result["crash_points"],
            uav_collision_points=result["uav_collision_points"],
        )

    df = pd.DataFrame([
        {
            "algorithm": r["algorithm"],
            "num_uav": r["num_uav"],
            "all_success": r["all_success"],
            "success_count": r["success_count"],
            "success_probability": r["success_probability"],
            "total_path_length": r["total_path_length"],
            "avg_path_length": r["avg_path_length"],
            "planning_time_sec": r["planning_time_sec"],
            "leader_path_length": r["leader_path_length"],
            "crash_count": r["crash_count"],
            "uav_collision_count": r["uav_collision_count"],
            "min_uav_distance": r["min_uav_distance"],
            "safe_distance": 3.0,
            "conflict_count": r["conflict_count"],
            "has_conflict": r["has_conflict"],
        }
        for r in results
    ])

    df.to_csv("results/metrics.csv", index=False, encoding="utf-8-sig")
    print(df)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    plot_df = df.copy()
    plot_df["total_path_length"] = plot_df["total_path_length"].fillna(0)

    ax1.bar(plot_df["algorithm"], plot_df["total_path_length"], alpha=0.7)
    ax1.set_ylabel("Total Path Length")
    ax1.set_title("Virtual Leader Formation Obstacle Avoidance Comparison")

    ax2 = ax1.twinx()
    ax2.plot(plot_df["algorithm"], plot_df["planning_time_sec"], marker="o")
    ax2.set_ylabel("Planning Time / sec")

    plt.savefig("results/comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nResults saved to the results folder.")
    print("Key files:")
    print("1. results/swarm_astar.png")
    print("2. results/swarm_rrt.png")
    print("3. results/swarm_rrt_star.png")
    print("4. results/metrics.csv")
    print("5. results/comparison.png")


if __name__ == "__main__":
    main()
