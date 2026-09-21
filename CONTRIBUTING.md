# Contributing

Useful contributions include independent simulator adapters, public recovery scenarios, stronger local baselines, and reproducible failure cases.

Run `python -m unittest discover -s tests -v` before a pull request. Include the command, model/version, data provenance, and test scope. Keep synthetic contracts, recorded-image diagnostics, and closed-loop outcomes distinct. Log fallback interventions and the distance/time they control.

Do not include credentials, private data, incompatible third-party code, or unsupported performance claims. Simulator adapters should reject stale observations and unavailable behaviors and leave vehicle-control authority with the host.
