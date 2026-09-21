"""Deterministic 2D bicycle-model scenes; these are not road-driving benchmarks."""

from dataclasses import asdict, dataclass
import heapq
import math
import random


SCENARIOS = {
    "clear_route": {"title": "Clear route", "description": "A control scene with no obstruction."},
    "blocked_route": {"title": "Blocked route", "description": "A stationary rock blocks the original route."},
    "changed_route": {"title": "Changing route", "description": "A second obstruction appears while the episode runs."},
    "low_visibility": {"title": "Observation gap", "description": "A blocked route plus a three-second sensor interruption."},
}
DT = 0.1
VEHICLE_RADIUS = 1.1
PLANNING_MARGIN = 1.2
SENSOR_RANGE = 24.0
GOAL = (94.0, 0.0)
START = (5.0, 0.0)


@dataclass(frozen=True)
class Obstacle:
    id: str
    x: float
    y: float
    radius: float
    appears_at: float = 0.0

    def to_dict(self):
        return asdict(self)


def make_obstacles(scenario, seed):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    rng = random.Random(seed)
    if scenario == "clear_route":
        return []
    rocks = [Obstacle("rock-1", 40 + rng.uniform(-2, 2), rng.uniform(-0.4, 0.4), 2.2 + rng.uniform(0, 0.4))]
    if scenario == "changed_route":
        rocks.append(Obstacle("rock-2", 67 + rng.uniform(-1, 1), -2.7, 2.7, appears_at=5.0))
    return rocks


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0 if length2 == 0 else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
    return distance(point, (a[0] + t * dx, a[1] + t * dy))


def segment_clear(a, b, obstacles, margin=VEHICLE_RADIUS):
    return all(segment_distance((o.x, o.y), a, b) > o.radius + margin for o in obstacles)


def plan_path(start, goal, obstacles):
    """A* with swept edge checks and obstacle inflation; no hidden scene access."""
    origin = (round(start[0]), round(start[1]))
    target = (round(goal[0]), round(goal[1]))
    margin = VEHICLE_RADIUS + PLANNING_MARGIN
    if not segment_clear(start, origin, obstacles, margin):
        return None
    if not segment_clear(target, goal, obstacles, margin):
        return None
    queue = [(distance(origin, target), 0.0, origin)]
    costs = {origin: 0.0}
    parents = {}
    offsets = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue
        if current == target:
            path = [current]
            while current in parents:
                current = parents[current]
                path.append(current)
            path.reverse()
            path = [start, *path, goal]
            # Greedy visibility smoothing preserves inflated-obstacle clearance.
            smooth, index = [start], 0
            while index < len(path) - 1:
                end = len(path) - 1
                while end > index + 1 and not segment_clear(path[index], path[end], obstacles, margin):
                    end -= 1
                smooth.append(path[end])
                index = end
            return smooth
        for dx, dy in offsets:
            nxt = (current[0] + dx, current[1] + dy)
            if not (2 <= nxt[0] <= 98 and -10 <= nxt[1] <= 10):
                continue
            if not segment_clear(current, nxt, obstacles, margin):
                continue
            candidate = cost + math.hypot(dx, dy)
            if candidate >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = candidate
            parents[nxt] = current
            heapq.heappush(queue, (candidate + distance(nxt, target), candidate, nxt))
    return None


@dataclass
class Vehicle:
    x: float = START[0]
    y: float = START[1]
    yaw: float = 0.0
    speed: float = 0.0
    steering: float = 0.0

    @property
    def position(self):
        return (self.x, self.y)

    def step(self, target, stop):
        aim_distance = max(1.0, distance(self.position, target))
        alpha = math.atan2(target[1] - self.y, target[0] - self.x) - self.yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        self.steering = max(-0.6, min(0.6, math.atan2(4.8 * math.sin(alpha), aim_distance)))
        desired = 0.0 if stop else max(1.7, 4.0 * (1 - abs(self.steering)))
        self.speed += max(-4.0 * DT, min(2.0 * DT, desired - self.speed))
        self.speed = max(0.0, self.speed)
        self.yaw += self.speed / 2.4 * math.tan(self.steering) * DT
        self.x += self.speed * math.cos(self.yaw) * DT
        self.y += self.speed * math.sin(self.yaw) * DT


def pursuit_target(position, path, index, lookahead=3.0):
    while index < len(path) - 1 and distance(position, path[index]) < lookahead:
        index += 1
    end = path[index]
    if index == 0:
        return end, index
    start = path[index - 1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return end, index
    projection = ((position[0] - start[0]) * dx + (position[1] - start[1]) * dy) / length
    fraction = min(1.0, max(0.0, projection + lookahead) / length)
    return (start[0] + dx * fraction, start[1] + dy * fraction), index
