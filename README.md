# JevDrive

**Semantic recovery and replanning decisions for autonomous-driving simulation.**

Turn structured driving observations into bounded decisions: continue the current policy, request a fresh observation, switch to an available recovery behavior, or ask a planner to replan. Jev is the planned semantic decision backend; local code retains trajectory feasibility and vehicle control.

**Status: design-stage project.** This repository currently contains the project specification and an illustrative observation contract. No Jev integration, simulator adapter, trained model, or measured driving improvement is released yet. Examples are authored fixtures, not recorded driving results.

## The first problem

A driving policy can keep repeating an unproductive action after an obstacle, terrain change, or loss of progress. The first experiment will ask whether a semantic supervisor can recognize these situations and select an appropriate next step without calling a large planner continuously.

The initial focus is off-road simulation: loss of progress, blocked routes, and requests to replan around an obstacle. The target demonstration is a paired replay of the same policy and disturbance, with and without the supervisor.

## Proposed architecture

```text
Sensors / simulator observations
        -> perception and state history
        -> code-computed features + available behavior candidates
        -> Jev semantic decisions
        -> local validity checks
        -> existing driving policy / recovery behavior / replanner
```

- Perception produces structured observations. Jev currently accepts text, not raw camera frames.
- Code computes elapsed time, distances, progress, collision checks, and other numerical features.
- The semantic model selects among candidates supplied by the environment integration.
- The vehicle controller owns steering, throttle, braking, and execution timing.
- Missing observations, unavailable behaviors, stale responses, and service failures have explicit outcomes.

`examples/observation.json` describes the proposed input contract. It is not the TypeSafe HTTP request schema and does not invoke an API.

## First milestone

- [ ] A versioned observation/action contract and recorded scenario format.
- [ ] One simulator adapter; BeamNG.tech is the first intended integration, subject to its external installation and license.
- [ ] A rule supervisor and an unmodified-policy baseline.
- [ ] A Jev adapter with model/version, latency, usage, and decision logging.
- [ ] Three repeatable disturbances: blocked route, persistent loss of progress, and changed local route conditions.
- [ ] An aligned comparison showing full elapsed time, including perception and API waits.

CARLA support is a later target, not an available feature. There is no real-vehicle deployment in this project.

## Evaluation contract

Compare the same policy and observation access under no supervisor, a rule supervisor, a VLM supervisor, and a Jev supervisor. Report route completion, collision counts, recovery success, unnecessary interventions, total elapsed time, p50/p95 decision latency, and all compute/API costs. Separate simulator-state experiments from camera-perception experiments. Split threshold tuning from final evaluation.

A confidence value is a model output statistic, not a certified collision probability or an end-to-end success rate. The project will report failures and unsuccessful recovery attempts alongside successes. No performance claims are available yet.

## 中文说明

JevDrive 研究 Jev 能否帮助已有自动驾驶策略判断“继续、补充观察、调用已有恢复行为、重新规划”。首个方向是仿真中的越野脱困与路线受阻处理。当前只有项目设计与示例数据契约，尚未实现 Jev 接入、BeamNG/CARLA 适配或闭环实验。

目标是用同一策略、同一场景和同一扰动做可复现对照。感知、几何计算、车辆控制与语义判断的职责分别记录；不会把离线判断准确率表述为真实驾驶能力。

## Related work and sources

- [TypeSafe model capabilities](https://docs.typesafe.ai/models)
- [TypeSafe confidence](https://docs.typesafe.ai/confidence)
- [Jev model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [jevpilot](https://github.com/standardagents/jevpilot): an existing Jev-powered browser driving simulator.
- [BeamNGpy](https://github.com/BeamNG/BeamNGpy): an intended simulator interface.

Independent community project; not affiliated with TypeSafe. The project code and documentation use the MIT license. External simulators and models keep their own licenses.
