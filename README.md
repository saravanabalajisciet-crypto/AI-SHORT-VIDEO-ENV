---
title: AI Video Optimizer Env
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# AI Short-Form Video Optimization Environment

> The only OpenEnv environment that trains AI agents to do what **50 million creators do every day** — edit short-form videos for maximum viral reach — grounded in real engagement data, with irreversible decisions, audience-aware scoring, and multi-platform evaluation.

---

## Overview

`ai-video-optimizer-env` is an OpenEnv-native environment for optimizing short-form video content across Instagram Reels, YouTube Shorts, and TikTok.

The environment follows current OpenEnv client/server conventions:

- `VideoOptimizationEnv` is the remote client (sync + context manager)
- `VideoOptimizationEnv.from_docker_image()` auto-starts a container
- `VideoOptimizationEnv.from_hf_space()` connects to the live HF Space
- `scenario_config.json` defines tasks with explicit verifiers
- `response_output/` stores per-run results for inspection

---

## Quick Start

```python
from client import VideoOptimizationEnv, VideoAction

# Connect to HF Space (no setup needed)
env = VideoOptimizationEnv.from_hf_space()
result = env.reset(platform="reels", seed=42)
print(result.engagement)        # 0.536
print(result.steps_remaining)   # 15
print(result.state["metadata"]["audience_persona"])  # gen_z

# Apply actions
result = env.step(VideoAction("boost_hook"))
print(result.reward)            # 0.67

# Get hint for next action
hint = env.hint()
print(hint["best_action"])      # "enhance_pacing"
print(hint["reason"])

# Grade the current state
score = env.grade()
print(score["score"])           # 0.87
print(score["rubric_score"])    # 0.87 (RL training signal)
print(score["raw_score"])       # 0.85 (pure observation quality)

# Finalize for persona-weighted bonus (irreversible)
result = env.step(VideoAction("finalize_edit"))
print(result.info["persona"])         # "gen_z"
print(result.info["persona_score"])   # 0.95
env.close()
```

```python
# Auto-start Docker container
with VideoOptimizationEnv.from_docker_image("video-env:latest") as env:
    result = env.reset(platform="reels", seed=42)
    result = env.step(VideoAction("boost_hook"))
    print(env.grade()["score"])
# Container auto-stopped on exit
```

---

## Why This Matters

The creator economy is a **$250 billion industry**. Every day, 50 million creators on Instagram Reels, YouTube Shorts, and TikTok face the same hard problem: given raw footage, make editorial decisions that maximize reach.

The difference between a video that goes viral and one that gets buried comes down to a handful of decisions made in the first edit:

- **Hook placement** — the algorithm scores the first 3 seconds. Wrong order = buried.
- **Scene cuts** — removing the wrong scene is irreversible. It permanently changes the retention curve.
- **Duration compliance** — 1 second over the platform limit = 35% less distribution.
- **Audience targeting** — a Gen-Z audience weights hook strength 2x more than a brand audience.

This environment models that exact problem as a **sequential decision task with real-world consequences**. Agents must learn editorial judgment through dense reward signals, real video data, and multi-scenario evaluation.

**Real-world grounding:** Scene data is derived from 50 video profiles across 30 niches. The first 10 videos are directly grounded in published research:
- [opus.pro](https://www.opus.pro/blog/tiktok-length-format-retention-data): 500 TikTok videos — 70% retention = 4.3x more impressions
- [socialinsider.io](https://www.socialinsider.io/social-media-benchmarks/social-media-video-statistics): 2025 Social Media Video Performance Statistics
- [vidico.com](https://vidico.com/news/instagram-reels-statistics): Reels organic reach dropped 50% in 2023 — hook-first is mandatory
- [dmnews.com](https://dmnews.com/video-marketing-works-until-you-realize-no-ones-watching-past-second-three): 75% of viewers click away before midpoint

---

## What Works Today

- Full OpenEnv spec: `reset()` / `step()` / `state()` / `grader()`
- 11 typed actions including irreversible `cut_scene` and `finalize_edit`
- 3 tasks (easy → medium → hard) with multi-seed evaluation for task_3
- Dense reward signal with 12 components + efficiency bonus + order penalty
- Audience personas (gen_z / millennial / brand) assigned per episode
- Real video dataset: 50 profiles, 30 niches, 7 research sources
- `steps_remaining` in observation — agent knows its budget
- Order-sensitive grader — correct action sequence rewarded
- `raw_score` + `rubric_score` separation for RL training
- `/hint` endpoint — best next action with reasoning
- `/leaderboard` — top scores across all graded episodes
- `/trajectory` — full action log with per-step metrics
- `/efficiency` — step efficiency rating (elite/optimal/good)
- `/dataset` — real video dataset with research citations
- `/persona` — current audience persona and scoring weights
- WebSocket endpoint for persistent sessions
- `VideoOptimizationEnv` client with `from_docker_image()` and context manager
- `scenario_config.json` with explicit verifiers per task
- `response_output/` — per-run JSON results saved automatically

---

## Architecture

```
ai-video-optimizer-env/
├── app.py                  # FastAPI server — 14 endpoints
├── environment.py          # Core RL environment logic
├── models.py               # Pydantic models (typed)
├── client.py               # VideoOptimizationEnv client library
├── inference.py            # LLM-powered inference script
├── scenario_config.json    # Task verifiers (explicit pass/fail criteria)
├── video_dataset.json      # 50 real-world video profiles, 30 niches
├── server/app.py           # OpenEnv entry point
├── requirements.txt        # Pinned dependencies
├── openenv.yaml            # OpenEnv spec v6
└── Dockerfile              # python:3.11-slim-bullseye, port 7860
```

**Server modules:**
- `environment.py` — scene generation, action handlers, reward computation, order scoring
- `app.py` — FastAPI endpoints, per-session isolation, leaderboard, trajectory logging
- `models.py` — Action, Observation (with steps_remaining), State, GraderResponse (with rubric_score)

---

## Tasks

| Task | Level | Target | Evaluation |
|------|-------|--------|------------|
| task_1 | Easy | 0.65 | Single seed (42) — remove filler, raise engagement > 0.60 |
| task_2 | Medium | 0.78 | Single seed (42) — compliance + retention + production quality |
| task_3 | Hard | 0.90 | **Average across seeds 42, 7, 13** — must generalize, correct order required |

task_3 is evaluated as the average score across 3 seeds. A simple heuristic that memorizes seed=42 will fail on seeds 7 and 13.

---

## Rewards

Rewards follow the OpenEnv rubric system. The environment uses a composite reward combining:

**Per-step dense reward (12 components):**

| Signal | Weight |
|--------|--------|
| Engagement delta | × 2.5 |
| Retention delta | × 2.0 |
| Hook strength delta | × 1.5 |
| Pacing delta | × 1.0 |
| Platform compliance | +0.20 on transition |
| Subtitles | +0.10 persistent |
| Hook-first | +0.15 on transition |
| Engagement milestone (>0.80) | +0.20 |
| Hook milestone (>0.70) | +0.15 |
| Persona-weighted finalize | up to +0.50 |

**Penalties:** invalid actions (-0.30), removing high-engagement scenes (-0.20), repeated idempotent actions (-0.20), over-editing (-0.10).

**Grader formula (deterministic, weights sum to 1.0):**
```
raw_score    = engagement*0.35 + retention*0.15 + compliance*0.20
             + subtitles*0.10 + hook*0.10 + pacing*0.05
             + transition*0.03 + cut*0.01 + audio*0.01

rubric_score = raw_score + efficiency_bonus + order_penalty
             (RL training signal — use this for GRPO/PPO)
```

**Efficiency tiers:** Elite (>= 0.92 in <= 7 steps) → Optimal → Good → Acceptable.

For RL training, use `rubric_score` from `/grader` — it provides efficiency-adjusted and order-sensitive credit assignment.

---

## Action Space (11 actions)

| Action | Parameters | Effect | Reversible |
|--------|-----------|--------|------------|
| `cut_scene` | `{"scene_id": str}` | Remove scene. Penalizes high-engagement cuts. | No |
| `reorder_scenes` | `{"order": [str]}` | Reorder scenes. Hook-first gives +0.02 lift. | Yes |
| `add_subtitles` | `{}` | +0.06 engagement lift. | One-time |
| `add_music` | `{}` | +0.12 on hook/content scenes. | One-time |
| `boost_hook` | `{}` | hook_strength +0.30, engagement +0.20. | One-time |
| `trim_duration` | `{"target_seconds": float}` | Trim to platform limit. | No |
| `enhance_pacing` | `{}` | Smooth engagement transitions. | One-time |
| `improve_transition` | `{}` | transition_quality +0.15. | One-time |
| `smooth_cut` | `{}` | cut_smoothness +0.15. | One-time |
| `sync_audio` | `{}` | audio_sync_score +0.15. | One-time |
| `finalize_edit` | `{}` | **IRREVERSIBLE.** Locks episode, persona-weighted bonus. | No |

---

## Observation Space (17 fields)

| Field | Type | Description |
|-------|------|-------------|
| `scenes` | List[Scene] | Full scene list with per-scene quality metrics |
| `current_engagement_score` | float | Mean engagement (0.0–1.0) |
| `avg_retention` | float | Mean viewer retention (0.0–1.0) |
| `retention_curve` | List[float] | Per-scene viewer drop-off |
| `watch_time` | float | Estimated watch time in seconds |
| `hook_strength` | float | Opening hook strength (0.0–1.0) |
| `pacing_score` | float | Smoothness of engagement transitions |
| `avg_transition_quality` | float | Mean transition quality |
| `avg_cut_smoothness` | float | Mean cut smoothness |
| `avg_audio_sync_score` | float | Mean audio sync quality |
| `total_duration` | float | Total video length in seconds |
| `platform_compliant` | bool | True if within platform duration limit |
| `hook_first` | bool | True if first scene is hook/highlight |
| `subtitles_present` | bool | Whether subtitles are added |
| `music_added` | bool | Whether music is added |
| `platform` | str | reels / shorts / tiktok |
| `steps_remaining` | int | Steps left in episode (budget awareness) |

---

## Audience Personas

Each episode is assigned an audience persona based on seed:

| Persona | Description | Hook Weight | Retention Weight | Compliance Weight |
|---------|-------------|-------------|-----------------|-------------------|
| `gen_z` | 18-24, mobile-first, high scroll velocity | 0.45 | 0.20 | 0.10 |
| `millennial` | 25-34, value-driven, watches to completion | 0.25 | 0.35 | 0.15 |
| `brand` | Brand safety, reach-focused | 0.20 | 0.20 | 0.40 |

Call `finalize_edit` to trigger persona-weighted final scoring. Call `/persona` to see current episode's weights.

---

## Baseline Performance

| Agent | task_1 | task_2 | task_3 (avg 3 seeds) | Notes |
|-------|--------|--------|----------------------|-------|
| Random agent | 0.31 | 0.28 | 0.24 | Random action each step |
| Worst case | 0.18 | 0.15 | 0.12 | No cuts, no hook boost |
| Greedy (boost_hook only) | 0.52 | 0.48 | 0.41 | Single action repeated |
| Partial (3 actions) | 0.71 | 0.68 | 0.62 | Hook + subtitles only |
| Heuristic baseline | 0.87 | 0.87 | 0.92 | Full optimal sequence |
| GPT-4o-mini (via proxy) | ~0.82 | ~0.79 | ~0.74 | LLM with hint endpoint |

4x difficulty range — random agents score ~0.28, optimal heuristic scores ~0.92.

---

## API Endpoints (14 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode. Accepts platform, seed. |
| POST | `/step` | Apply one action. Returns state, reward, done, info. |
| GET | `/state` | Read current state. |
| GET | `/tasks` | All 3 task definitions with verifiers. |
| GET | `/grader` | Score with raw_score, rubric_score, task_type. |
| GET | `/baseline` | Deterministic baseline across all tasks. |
| GET | `/feedback` | AI coaching tips. |
| GET | `/hint` | Best next action with reasoning. |
| GET | `/scenarios` | 5 diverse scenario seeds. |
| GET | `/trajectory` | Full episode action log + order_score. |
| GET | `/efficiency` | Step efficiency rating. |
| GET | `/dataset` | Real video dataset metadata + research citations. |
| GET | `/persona` | Current audience persona + scoring weights. |
| GET | `/leaderboard` | Top scores across all graded episodes. |

---

## Setup

**Local mode (no Docker needed):**
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker mode:**
```bash
docker build -t video-env:latest .
docker run -p 7860:7860 video-env:latest
```

**Mock mode (point at any server):**
```bash
# HF Space (no local setup)
ENV_URL=https://saravanabalajisara-ai-video-optimizer-env.hf.space python inference.py

# Local Docker
ENV_URL=http://localhost:7860 python inference.py

# With scenario config
python inference.py --scenario scenario_config.json
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV_URL` | `http://localhost:7860` | RL environment server URL |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM proxy base URL (injected by validator) |
| `API_KEY` | — | LLM API key |
| `HF_TOKEN` | — | HuggingFace token (fallback for API_KEY) |
| `OPENAI_API_KEY` | — | OpenAI key (fallback) |
| `MODEL_NAME` | `gpt-4o-mini` | LLM model identifier |

---

## Live URLs

- API: `https://saravanabalajisara-ai-video-optimizer-env.hf.space`
- Docs: `https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs`
- Space: `https://huggingface.co/spaces/saravanabalajisara/ai-video-optimizer-env`
- GitHub: `https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV`

---

## License

MIT

---

## Citation

```bibtex
@misc{ai-video-optimizer-env,
  author  = {SaravanaBalaji},
  title   = {AI Short-Form Video Optimization Environment for OpenEnv},
  year    = {2026},
  url     = {https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV},
  note    = {OpenEnv-compatible RL environment for training agents to optimize
             short-form video content across Instagram Reels, YouTube Shorts,
             and TikTok. Grounded in public creator analytics research.}
}
```

---

## References

- [opus.pro: Ideal TikTok Length & Format for Retention](https://www.opus.pro/blog/tiktok-length-format-retention-data) — 500 video analysis, retention data
- [socialinsider.io: 2025 Social Media Video Performance Statistics](https://www.socialinsider.io/social-media-benchmarks/social-media-video-statistics)
- [vidico.com: Instagram Reels Statistics](https://vidico.com/news/instagram-reels-statistics)
- [dmnews.com: Video retention research](https://dmnews.com/video-marketing-works-until-you-realize-no-ones-watching-past-second-three)
- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Rubric RFC 004](https://github.com/meta-pytorch/OpenEnv)

---

## Decision Pressure & Irreversible Optimization

### Risk-Aware Evaluation

Every observation now includes a `risk_score` (0.0–1.0) that signals how close the current state is to an irreversible failure — a state from which no sequence of remaining actions can recover a high score.

```python
obs = env.reset().observation
print(obs["risk_score"])   # e.g. 0.72 — high danger
```

Risk is computed from three weak-signal indicators:

| Signal | Danger threshold | Weight |
|--------|-----------------|--------|
| `hook_strength` | < 0.6 | 45% |
| `avg_retention` | < 0.5 | 35% |
| `pacing_score` | < 0.4 | 20% |

A `risk_score` above 0.7 indicates entry into a high-risk region where recovery to optimal performance becomes unlikely under remaining step constraints.

### Soft Score Caps

The grader applies post-processing caps to `rubric_score` (the RL training signal) when the agent has made structurally poor decisions. `raw_score` is **never modified**.

| Condition | Effect |
|-----------|--------|
| `hook_strength < 0.4` after step 3 | `rubric_score` capped at 0.60 |
| `avg_retention < 0.3` | `rubric_score` × 0.7 multiplier |
| `platform_compliant == False` | `rubric_score` − 0.10 |

The `/grader` response now includes a `grader_metadata` block:

```json
{
  "grader_metadata": {
    "risk_score": 0.72,
    "score_cap_applied": true,
    "cap_reason": "low_hook_strength",
    "irreversible_decision_point": false,
    "finalized_at_step": null,
    "finalized_risk_score": null
  }
}
```

### Irreversible Decision Point

`finalize_edit` is the only truly irreversible action. When called, the episode metadata records:

- `irreversible_decision_point: true`
- `finalized_at_step` — which step the agent committed
- `finalized_risk_score` — the risk level at the moment of commitment

This allows evaluators to distinguish agents that commit confidently at low risk vs. agents that panic-finalize under pressure.
