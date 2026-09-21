# Research target and admission criteria

## Question

Can a semantic supervisor improve a fixed driving policy's recovery success under a fixed compute budget, once perception cost, network delays, stale decisions, and fallback interventions are included?

The proposed contribution is an empirical study of **when structured semantic decisions help recovery**. Jev is one decision backend. Replacing a language-model call with a Jev call is not, by itself, a research contribution. No novelty or driving-performance claim has been established.

## Mechanism under investigation

At a decision boundary, an unchanged driving policy and local planner expose a finite set of available behaviors. A perception module produces current observations, while code supplies numerical progress and feasibility facts. A supervisor requests continuation, additional observation, or replanning. The host rejects decisions that refer to an old observation, exceed the response deadline, name an unavailable candidate, or fail its local feasibility checks.

The testable hypothesis is that semantic context adds value on ambiguous route and recovery situations that numerical rules alone cannot resolve, while a small local model handles common cases. A second hypothesis is that the benefit disappears when perception and stale-response costs dominate. Both outcomes are publishable evidence only if evaluated rigorously; neither is assumed here.

## Existing work that constrains the claim

| Work | Established scope from its official project | Consequence for this project |
| --- | --- | --- |
| [AutoVLA](https://github.com/ucla-mobility/AutoVLA), [paper](https://arxiv.org/abs/2506.13757) | Adaptive fast/slow driving reasoning; released model | A two-system switch or fewer reasoning tokens is not sufficient novelty. Include an adaptive-reasoning baseline when license and integration allow. |
| [DriveVLM](https://github.com/Tsinghua-MARS-Lab/DriveVLM) | VLM-based driving planning | Semantic driving planning is established prior art. |
| [SimLingo](https://github.com/RenzKa/simlingo), [paper](https://arxiv.org/abs/2503.09594) | Language-action driving model with released closed-loop evaluation | A practical fixed-policy baseline candidate. Keep its model/software license separate. |
| [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | Standardized closed-loop CARLA evaluation, development and full route suites | Use development routes for integration, then freeze settings before full evaluation. |
| [Bench2Drive-Robust](https://github.com/Thinklab-SJTU/Bench2Drive-Robust) | Camera, state, and compute/control-delay disturbances | Delay robustness alone is already studied. Compare against this protocol before claiming a new benchmark. |
| [Bench2Drive-VL](https://github.com/Thinklab-SJTU/Bench2Drive-VL) | Closed-loop vision-language evaluation infrastructure | Reuse public interfaces where compatible instead of redefining benchmark rules. |
| [jevpilot](https://github.com/standardagents/jevpilot) | Existing Jev driving demonstration | A browser driving demo does not distinguish this repository. |

This is a scoped prior-art check, not an exhaustive novelty certification. Exact release commits and model licenses must be recorded at integration time.

## Paired baseline matrix

Keep the underlying driving policy, sensor access, route, seed, local guard, and planner constant:

1. Original policy, with all inherited fallback behavior counted.
2. Fixed-interval planner calls.
3. Numerical event rules (progress, availability, geometry, observation freshness).
4. A small trained local classifier using the same observable inputs.
5. A direct VLM supervisor.
6. VLM perception followed by a text-model decision, sharing exactly the Jev text input.
7. VLM perception followed by Jev.

Then ablate event triggering, semantic fields, confidence thresholds, and response-age rejection separately. All methods must run under matched intervention and compute budgets. Model outputs may request only behaviors actually implemented by the simulator integration.

## Outcome data and calibration

- Obtain labels from executed candidate rollouts: collision, route completion, timeout, recovery duration, and intervention count. Teacher trajectory proximity is not a safety or recovery label.
- Keep route/source groups intact across train, calibration, and test. Near-duplicate frames from one run are not independent test scenes.
- Fit any confidence-to-outcome mapping only on calibration routes. Report Brier score, reliability bins, selective coverage/risk, and failure rates on untouched routes.
- Jev's Choice confidence summarizes its distribution. It is not automatically a collision probability. The repository's metric helpers require explicit measured outcome labels.
- Do not feed future frames, executed outcomes, teacher actions, privileged route IDs, or target coordinates into the semantic state unless all compared methods receive the same explicitly declared privilege.

## Closed-loop evaluation

Start with a licensed, independently installed CARLA/Bench2Drive environment and a released policy. Run the public development route subset as an integration check. Final evaluation should use the official held-out route suite and repeated seeds, with route-level paired bootstrap intervals. A second independently instrumented off-road suite tests transfer. Existing private research code is not part of this repository and is not modified by the camera probe.

Measure route completion and collision per distance jointly; a stopped vehicle can have few collisions. Report recovery success, timeout rate, comfort, unnecessary replans, fallback-controlled distance, policy-controlled distance, perception latency, decision latency, deadline misses, total wall time, GPU work, and API usage. Inject real delays while the simulator advances; do not pause simulated time around a slow API call and call that real-time driving.

## Milestone gates

| Gate | Required evidence | Current status |
| --- | --- | --- |
| API | Live versioned response, latency, usage, malformed-response handling | Implemented and smoke-tested |
| Recorded vision | Real RGB inference, image/model hashes, direct image and shared-text baselines | Probe code implemented; see diagnostic report for measured run |
| Outcome data | Independent route splits and measured recovery outcomes | Not collected |
| Closed loop | Fixed released policy, real simulator runs, matched guards and budgets | Not integrated |
| Training | Real-data admission canary, checkpoint plus independent reload before long jobs | No training started |
| Paper claim | Replicated held-out improvement against strong baselines, or a well-supported negative result | No result yet |

Star count and conference acceptance are external outcomes. The engineering targets are reproducible commands, useful adapters, transparent failure cases, and evidence strong enough for independent review.
