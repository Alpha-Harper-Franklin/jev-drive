"""Minimal live Jev Choice client with explicit errors and response validation."""
from dataclasses import dataclass, asdict
import json
import math
import os
import time
import urllib.error
import urllib.request


class JevError(RuntimeError):
    """Safe to log: stores an error code, never response text or credentials."""


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


@dataclass
class ChoiceResult:
    choice: str
    probabilities: dict
    confidence: float
    model: str
    input_tokens: int | None
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def validate_choice(response, candidates):
    try:
        answer = response['answers']['decision']
        selected, probs, confidence = answer['choice'], answer['probabilities'], answer['confidence']
        if answer['type'] != 'choice' or selected not in candidates:
            raise ValueError()
        if not isinstance(probs, dict) or set(probs) != set(candidates):
            raise ValueError()
        if not all(probability(v) for v in probs.values()) or not probability(confidence):
            raise ValueError()
        if not math.isclose(sum(probs.values()), 1, abs_tol=0.005):
            raise ValueError()
        if probs[selected] + 1e-8 < max(probs.values()):
            raise ValueError()
        model = response['model']
        if not isinstance(model, str) or not model:
            raise ValueError()
        tokens = response.get('usage', {}).get('input_tokens')
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError()
        return ChoiceResult(selected, probs, confidence, model, tokens)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise JevError('invalid_choice_response') from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class JevClient:
    def __init__(self, api_key=None, model='jev-1.13.0', timeout=15, transport=None):
        self.key = api_key or os.environ.get('TYPESAFE_API_KEY')
        if not self.key:
            raise JevError('missing_TYPESAFE_API_KEY')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.model, self.timeout = model, timeout
        self.transport = transport or self._http

    def _http(self, payload):
        request = urllib.request.Request(
            'https://api.typesafe.ai/v1/systemone',
            data=json.dumps(payload, allow_nan=False).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
            return json.load(response)

    def choice(self, state, question, candidates):
        if not isinstance(candidates, dict) or not 1 <= len(candidates) <= 255:
            raise ValueError('Offer between 1 and 255 uniquely named candidates')
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in candidates.items()):
            raise ValueError('Candidate IDs and criteria must be nonempty strings')
        payload = {'model': self.model, 'state': state, 'questions': {
            'decision': {'type': 'choice', 'instructions': question, 'criteria': candidates}}}
        started = time.perf_counter()
        try:
            result = validate_choice(self.transport(payload), candidates)
        except urllib.error.HTTPError as error:
            raise JevError(f'http_{error.code}') from None
        except (OSError, ValueError, TypeError):
            raise JevError('transport_or_json_error') from None
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
