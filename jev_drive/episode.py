"""Closed-loop episodes with explicit observation provenance and real timing."""

from collections import deque
from dataclasses import asdict
import math
import time

from .providers import make_provider
from .world import DT, GOAL, START, SENSOR_RANGE, VEHICLE_RADIUS, Vehicle, distance, make_obstacles, plan_path, pursuit_target, segment_clear


class Episode:
    def __init__(self, scenario, seed, provider="rules", max_seconds=60.0):
        self.scenario, self.seed = scenario, seed
        self.provider = make_provider(provider) if isinstance(provider, str) else provider
        self.obstacles = make_obstacles(scenario, seed)
        self.vehicle = Vehicle()
        self.path = [START, GOAL]
        self.path_index = 1
        self.path_revision = 0
        self.known = {}
        self.sim_time = 0.0
        self.max_seconds = max_seconds
        self.status = "running"
        self.action = "continue"
        self.history = deque(maxlen=51)
        self.trail = [list(START)]
        self.frames = []
        self.decisions = []
        self.latest_observation = None
        self.last_signature = None
        self.last_decision_at = -math.inf
        self.last_replan_at = -math.inf
        self.replans = 0
        self.failed_replans = 0
        self.unnecessary_interventions = 0
        self.collisions = 0
        self.guard_stops = 0
        self.guard_active = False
        self.compute_seconds = 0.0
        self.started_at = time.perf_counter()
        self.finished_at = None

    def active_obstacles(self):
        return [o for o in self.obstacles if self.sim_time + 1e-8 >= o.appears_at]

    def observe(self):
        fresh = not (self.scenario == "low_visibility" and 7.0 <= self.sim_time < 10.0)
        if fresh:
            for obstacle in self.active_obstacles():
                if distance(self.vehicle.position, (obstacle.x, obstacle.y)) <= SENSOR_RANGE:
                    self.known[obstacle.id] = obstacle
        remaining = [self.vehicle.position, *self.path[self.path_index:]]
        blocked = any(not segment_clear(a, b, self.known.values(), VEHICLE_RADIUS + 0.4) for a, b in zip(remaining, remaining[1:]))
        moved = distance(self.history[0][1], self.vehicle.position) if self.history else 0
        stalled = len(self.history) == self.history.maxlen and moved < 0.5
        observation = {
            "contract_version": "0.1.0",
            "provenance": "synthetic 2D bicycle simulation; range-limited exact geometry, no camera model",
            "observation_id": f"{self.scenario}-{self.seed}-{round(self.sim_time / DT)}",
            "task": "Reach the destination using the existing local controller and available planner.",
            "facts": {"observation_fresh": fresh, "route_blocked": blocked, "progress_stalled": stalled},
            "computed": {
                "sim_seconds": round(self.sim_time, 2),
                "speed_mps": round(self.vehicle.speed, 3),
                "goal_distance_m": round(distance(self.vehicle.position, GOAL), 3),
                "recent_displacement_m": round(moved, 3),
                "history_window_seconds": round(self.sim_time - self.history[0][0], 2) if self.history else 0,
                "path_revision": self.path_revision,
            },
            "observed_obstacles": [o.to_dict() for o in self.known.values()],
            "available_decisions": ["continue", "replan", "observe", "defer"],
        }
        self.latest_observation = observation
        return observation

    def step(self):
        if self.status != "running":
            return
        started = time.perf_counter()
        observation = self.observe()
        facts = observation["facts"]
        signature = tuple(facts.values())
        changed = signature != self.last_signature
        retry_hold = self.action in ("observe", "defer") and self.sim_time - self.last_decision_at >= 5.0
        if changed or retry_hold:
            decision = self.provider.decide(observation)
            self.last_signature = signature
            self.last_decision_at = self.sim_time
            self.action = decision.action
            entry = {"sim_seconds": round(self.sim_time, 2), "observation": observation, **decision.to_dict()}
            if decision.action == "replan":
                if not facts["route_blocked"] and not facts["progress_stalled"]:
                    self.unnecessary_interventions += 1
                path = plan_path(self.vehicle.position, GOAL, self.known.values()) if facts["observation_fresh"] else None
                if path:
                    self.path, self.path_index = path, 1
                    self.path_revision += 1
                    self.replans += 1
                    self.last_replan_at = self.sim_time
                    self.action = "continue"
                    entry["execution"] = "local_path_updated"
                else:
                    self.failed_replans += 1
                    self.action = "defer"
                    entry["execution"] = "planner_unavailable_or_no_path"
            self.decisions.append(entry)

        target, self.path_index = pursuit_target(self.vehicle.position, self.path, self.path_index)
        braking_distance = self.vehicle.speed ** 2 / 8.0 + 2.5
        forward = (
            self.vehicle.x + math.cos(self.vehicle.yaw) * braking_distance,
            self.vehicle.y + math.sin(self.vehicle.yaw) * braking_distance,
        )
        imminent = not segment_clear(self.vehicle.position, forward, self.known.values(), VEHICLE_RADIUS + 0.3)
        guard = imminent or not facts["observation_fresh"]
        if guard and not self.guard_active:
            self.guard_stops += 1
        self.guard_active = guard
        previous = self.vehicle.position
        self.vehicle.step(target, stop=guard or self.action in ("observe", "defer"))
        self.sim_time += DT
        self.history.append((self.sim_time, self.vehicle.position))
        if not segment_clear(previous, self.vehicle.position, self.active_obstacles()):
            self.collisions += 1
            self.status = "collision"
        elif abs(self.vehicle.y) > 12 or not (0 <= self.vehicle.x <= 101):
            self.status = "out_of_bounds"
        elif distance(self.vehicle.position, GOAL) < 1.8:
            self.status = "success"
        elif self.sim_time + 1e-8 >= self.max_seconds:
            self.status = "timeout"
        self.compute_seconds += time.perf_counter() - started
        if round(self.sim_time / DT) % 2 == 0 or self.status != "running":
            self.trail.append([round(self.vehicle.x, 3), round(self.vehicle.y, 3)])
            self.frames.append({"sim_seconds": round(self.sim_time, 2), "vehicle": asdict(self.vehicle), "status": self.status, "path_revision": self.path_revision})
        if self.status != "running":
            self.finished_at = time.perf_counter()

    def cancel(self):
        if self.status == "running":
            self.status = "cancelled"
            self.finished_at = time.perf_counter()

    def metrics(self):
        latencies = sorted(d["latency_ms"] for d in self.decisions if d["api_attempts"])
        costs = [d["estimated_api_usd"] for d in self.decisions if d["estimated_api_usd"] is not None]
        attempts = sum(d["api_attempts"] for d in self.decisions)
        successes = sum(d["api_successes"] for d in self.decisions)
        known_cost = sum(costs)
        return {
            "status": self.status,
            "success": self.status == "success",
            "sim_seconds": round(self.sim_time, 2),
            "wall_seconds": round((self.finished_at or time.perf_counter()) - self.started_at, 3),
            "compute_seconds": round(self.compute_seconds, 6),
            "goal_distance_m": round(distance(self.vehicle.position, GOAL), 3),
            "replans": self.replans,
            "failed_replans": self.failed_replans,
            "unnecessary_interventions": self.unnecessary_interventions,
            "collisions": self.collisions,
            "guard_stops": self.guard_stops,
            "decision_count": len(self.decisions),
            "api_attempts": attempts,
            "api_successes": successes,
            "api_errors": sum(bool(d["error"]) for d in self.decisions),
            "api_latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
            "api_latency_p95_ms": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else None,
            "estimated_api_usd": round(known_cost, 10) if len(costs) == attempts else None,
            "known_usage_estimated_usd": round(known_cost, 10),
            "cost_note": "Estimate at $0.042/M input tokens; missing usage and failed calls have unknown cost.",
        }

    def snapshot(self):
        return {
            "provider": self.provider.name,
            "vehicle": asdict(self.vehicle),
            "path": self.path,
            "trail": self.trail,
            "obstacles": [o.to_dict() for o in self.active_obstacles()],
            "observed_ids": list(self.known),
            "goal": GOAL,
            "action": self.action,
            "guard_active": self.guard_active,
            "metrics": self.metrics(),
            "decisions": self.decisions[-8:],
            "observation": self.latest_observation,
        }

    def report(self):
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "provider": self.provider.name,
            "metrics": self.metrics(),
            "decisions": self.decisions,
            "frames": self.frames,
            "final_state": self.snapshot(),
        }


def run_episode(scenario, seed, provider="rules"):
    episode = Episode(scenario, seed, provider)
    while episode.status == "running":
        episode.step()
    return episode
