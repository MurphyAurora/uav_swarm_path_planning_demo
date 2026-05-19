import math
from utils import path_length


def interpolate_path_by_distance(path, step=1.0):
    # 把不同长度、不同点数的路径，按照近似等距离重新采样。
    # 这样可以模拟多无人机“同步飞行”时每个时刻的位置。
    if path is None or len(path) < 2:
        return []

    sampled = [path[0]]
    carry = 0.0

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)

        if seg_len < 1e-9:
            continue

        dist = step - carry
        while dist <= seg_len:
            t = dist / seg_len
            sampled.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            dist += step

        carry = seg_len - (dist - step)

    if sampled[-1] != path[-1]:
        sampled.append(path[-1])

    return sampled


def pad_paths(sampled_paths):
    # 不同无人机路径长度不同，较早到达的无人机保持在终点。
    max_len = max(len(p) for p in sampled_paths)
    padded = []

    for p in sampled_paths:
        if len(p) == 0:
            padded.append([])
            continue
        q = list(p)
        while len(q) < max_len:
            q.append(q[-1])
        padded.append(q)

    return padded


def check_swarm_conflicts(paths, safe_distance=3.0, sample_step=1.0):
    # 检查多无人机同步飞行时是否小于安全距离。
    if any(p is None or len(p) < 2 for p in paths):
        return {
            "min_distance": None,
            "conflict_count": None,
            "has_conflict": None,
        }

    sampled_paths = [interpolate_path_by_distance(p, step=sample_step) for p in paths]
    sampled_paths = pad_paths(sampled_paths)

    min_distance = float("inf")
    conflict_count = 0

    time_steps = len(sampled_paths[0])
    num_uav = len(sampled_paths)

    for t in range(time_steps):
        for i in range(num_uav):
            for j in range(i + 1, num_uav):
                p1 = sampled_paths[i][t]
                p2 = sampled_paths[j][t]
                d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                min_distance = min(min_distance, d)

                if d < safe_distance:
                    conflict_count += 1

    return {
        "min_distance": min_distance,
        "conflict_count": conflict_count,
        "has_conflict": conflict_count > 0,
    }


def reached_goal(path, goal, tolerance=1.0):
    if path is None or len(path) < 2:
        return False

    end = path[-1]
    return math.hypot(end[0] - goal[0], end[1] - goal[1]) <= tolerance


def summarize_swarm_result(paths, safe_distance=3.0, goals=None):
    if goals is None:
        successes = [p is not None and len(p) > 1 for p in paths]
    else:
        successes = [
            reached_goal(path, goal)
            for path, goal in zip(paths, goals)
        ]

    success_count = sum(successes)
    success_probability = success_count / len(paths) if paths else 0.0
    all_success = all(successes)

    lengths = [path_length(p) if p is not None else None for p in paths]
    total_length = sum(v for v in lengths if v is not None)

    conflict = check_swarm_conflicts(paths, safe_distance=safe_distance)

    return {
        "all_success": all_success,
        "success_count": success_count,
        "success_probability": success_probability,
        "total_path_length": total_length if all_success else None,
        "avg_path_length": total_length / len(paths) if all_success else None,
        "min_uav_distance": conflict["min_distance"],
        "conflict_count": conflict["conflict_count"],
        "has_conflict": conflict["has_conflict"],
    }
