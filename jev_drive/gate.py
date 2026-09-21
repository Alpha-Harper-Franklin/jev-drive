"""Host-side freshness and availability check for asynchronous decision responses.

This does not implement a vehicle controller or certify trajectory safety.
The simulator integration must provide its own local feasibility veto.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Ticket:
    observation_id: str
    observed_at: float  # host monotonic clock, seconds
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    action: str | None
    reason: str


def assess(ticket, choice, *, now, current_observation_id, available, feasible, max_age=0.5):
    if not ticket.observation_id or len(set(ticket.candidates)) != len(ticket.candidates):
        return GateResult(False, None, 'invalid_ticket')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [now, ticket.observed_at, max_age]) or max_age <= 0:
        return GateResult(False, None, 'invalid_clock')
    if current_observation_id != ticket.observation_id:
        return GateResult(False, None, 'superseded_observation')
    age = now - ticket.observed_at
    if age < 0 or age >= max_age:
        return GateResult(False, None, 'expired_observation')
    if choice not in ticket.candidates or choice not in available:
        return GateResult(False, None, 'unavailable_candidate')
    if feasible.get(choice) is not True:
        return GateResult(False, None, 'local_veto_or_unknown')
    return GateResult(True, choice, 'accepted_request')
