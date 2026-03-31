---
title: AIVideoOptimizer
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# AI Short-Form Video Optimization Environment

An OpenEnv-compatible reinforcement learning environment that teaches agents to edit short-form videos for maximum viral engagement.

---

## Overview

Short-form video is the dominant content format on the internet. Platforms like Instagram Reels, YouTube Shorts, and TikTok reward creators who can produce tightly edited, high-retention videos — but optimizing a video across engagement, pacing, hook strength, and platform compliance simultaneously is a non-trivial multi-objective problem.

This environment simulates a real-world video editing workflow where an RL agent acts as an AI video editor. The agent receives a raw video represented as a sequence of scenes, each with measurable quality attributes, and must apply a series of editing actions to maximize a composite viral performance score. Every decision the agent makes — cutting a scene, boosting the hook, syncing audio — produces a meaningful reward signal grounded in real creator best practices.

---

## Problem Statement

Over 500 hours of video are uploaded to YouTube every minute. On short-form platforms, the average viewer decides whether to keep watching within the first three seconds. Creators face a complex optimization problem:

- **Engagement** drops sharply when low-quality scenes are left in the timeline
- **Retention** is sensitive to pacing — jarring cuts and poor transitions cause viewers to scroll away
- **Platform compliance** (duration limits, subtitle presence) directly affects algorithmic distribution
- **Hook strength** in the opening seconds determines whether a video goes viral or gets buried

Human editors rely on intuition built over years of experience. This environment provides a structured, reproducible testbed for training AI agents to learn those same editorial instincts through interaction.

---

## Environment Description

The environment models a video as a list of scenes. Each scene has a type (hook, highlight, content, filler, transition, cta), a duration, an engagement score, and three production quality metrics: transition quality, cut smoothness, and audio sync score.

An agent interacts with the environment through the standard OpenEnv loop:

1. `POST /reset` — initialize a new episode with a randomly generated video (reproducible via seed)
2. `POST /step` — apply one editing action and receive the updated state, reward, done flag, and info
3. `GET /state` — read the current state at any time without advancing the episode

This mirrors a real editing session: the agent opens a project, reviews the timeline, makes edits one at a time, and receives feedback on how each edit affects the video's performance metrics.

Episodes end when `max_steps` (15) is reached or the engagement score crosses a high-performance threshold.

---

## Action Space

| Action | Parameters | Effect |
|---|---|---|
| `cut_scene` | `{"scene_id": str}` | Removes a scene from the timeline. Penalizes removal of high-engagement scenes. |
| `add_subtitles` | `{}` | Enables subtitles. Lifts engagement on all scenes by +0.06. Idempotent — penalizes repeat use. |
| `add_music` | `{"genre": str}` | Adds background music. Boosts hook/highlight/content scenes by +0.12, others by +0.05. |
| `enhance_pacing` | `{}` | Smooths engagement transitions between scenes using neighbour averaging. Applies a global micro-lift. |
| `boost_hook` | `{}` | Strengthens the first scene's hook. Raises hook_strength by +0.30, engagement by +0.20, and carries +0.10 energy to the second scene. |
| `improve_transition` | `{}` | Raises transition_quality on all scenes by +0.15 and engagement by +0.03. |
| `smooth_cut` | `{}` | Raises cut_smoothness on all scenes by +0.15 and engagement by +0.02. |
| `sync_audio` | `{}` | Raises audio_sync_score on all scenes by +0.15 and engagement by +0.03. |
| `trim_duration` | `{"target_seconds": float}` | Automatically removes lowest-engagement non-hook scenes until duration is within target. Applies a +0.03 engagement lift to remaining scenes. |
| `reorder_scenes` | `{"order": [str, ...]}` | Reorders all scenes. If a hook/highlight scene moves to first position, applies a +0.02 engagement lift across all scenes. |

All one-time actions (add_music, boost_hook, enhance_pacing, improve_transition, smooth_cut, sync_audio) are idempotent-guarded — repeating them returns a `-0.2` penalty and marks the action invalid.

---

## Observation Space

Each call to `/step` or `/state` returns a full `Observation` object:

```json
{
  "scenes": [
    {
      "id": "scene_0",
      "duration": 5.5,
      "engagement_score": 0.834,
      "scene_type": "hook",
      "has_hook": true,
      "hook_strength": 1.0,
      "transition_quality": 0.741,
      "cut_smoothness": 0.724,
      "audio_sync_score": 0.738
    }
  ],
  "total_duration": 12.3,
  "subtitles_present": true,
  "music_added": true,
  "platform": "reels",
  "current_engagement_score": 0.834,
  "hook_first": true,
  "platform_compliant": true,
  "retention_curve": [0.97, 0.94, 0.91],
  "avg_retention": 0.970,
  "watch_time": 11.9,
  "hook_strength": 1.0,
  "pacing_score": 0.674,
  "avg_transition_quality": 0.741,
  "avg_cut_smoothness": 0.724,
  "avg_audio_sync_score": 0.738
}
```

| Field | Description |
|---|---|
| `scenes` | Full list of scenes with per-scene quality metrics |
| `current_engagement_score` | Mean engagement across all scenes (0.0–1.0) |
| `avg_retention` | Mean of the per-scene viewer retention curve (0.0–1.0) |
| `retention_curve` | Per-scene retention values simulating viewer drop-off |
| `watch_time` | Estimated watch time in seconds (duration × retention) |
| `hook_strength` | Opening hook strength of the first scene (0.0–1.0) |
| `pacing_score` | Smoothness of engagement transitions between scenes |
| `avg_transition_quality` | Mean transition quality across all scenes |
| `avg_cut_smoothness` | Mean cut smoothness across all scenes |
| `avg_audio_sync_score` | Mean audio sync quality across all scenes |
| `total_duration` | Total video length in seconds |
| `platform_compliant` | True if duration is within the platform limit |
| `hook_first` | True if the first scene is a hook/highlight with has_hook=True |

---

## Reward Function

The reward function is dense and step-based — the agent receives a signal after every action, not just at episode end.

**Positive signals:**

| Signal | Weight |
|---|---|
| Engagement improvement | delta × 2.5 |
| Retention improvement | delta × 2.0 |
| Hook strength improvement | delta × 1.5 |
| Pacing improvement | delta × 1.0 |
| Transition quality (continuous) | value × 0.05 |
| Cut smoothness (continuous) | value × 0.05 |
| Audio sync (continuous) | value × 0.05 |
| Platform compliance (transition) | +0.20 |
| Platform compliance (persistent) | +0.05 |
| Subtitles present | +0.10 |
| Hook-first structure (transition) | +0.15 |
| Hook-first structure (persistent) | +0.03 |
| Engagement crosses 0.80 milestone | +0.20 |
| Hook strength crosses 0.70 milestone | +0.15 |

**Penalties:**

| Condition | Penalty |
|---|---|
| Invalid scene_id in cut_scene | -0.30 |
| Removing a high-engagement scene (>= 0.65) | -0.20 |
| Repeating an idempotent action | -0.20 |
| Fewer than 3 scenes remaining | -0.10 |
| Action produces no measurable change | -0.05 |

---

## Tasks

All tasks use the same environment (seed=42, platform=reels) and are scored on a 0.0–1.0 scale by the `/grader` endpoint.

### Task 1 — Easy (target score: 0.65)

Remove low-engagement scenes to raise the average engagement score above 0.60. The video starts with 5 scenes including at least one filler and one transition scene with engagement below 0.30.

**Expected strategy:** Identify scenes with `engagement_score < 0.30` and remove them using `cut_scene`. Follow up with `add_subtitles` and `add_music`.

### Task 2 — Medium (target score: 0.78)

Optimize for platform compliance and viewer retention. The video must fit within the 30-second Reels limit, subtitles must be present, and `avg_retention` must reach 0.55 or above.

**Expected strategy:** Trim duration, add subtitles and music, call `enhance_pacing` to smooth retention drop-off. Use `improve_transition`, `smooth_cut`, and `sync_audio` to lift production quality.

### Task 3 — Hard (target score: 0.875)

Full viral optimization. The agent must achieve `engagement >= 0.80`, `avg_retention >= 0.90`, `hook_strength >= 0.70`, `avg_transition_quality >= 0.70`, platform compliance, subtitles, and hook-first ordering — all within 15 steps.

**Expected strategy:** The correct action sequence matters. Reorder first, then boost the hook, cut dead-weight scenes, enhance pacing, improve all three production quality metrics, add subtitles, then add music. Executing these out of order yields lower reward.

---

## Grader System

The `/grader` endpoint computes a deterministic, weighted score from the current state:

```
score = engagement           * 0.35
      + retention            * 0.15
      + platform_compliance  * 0.20
      + subtitles            * 0.10
      + hook_strength        * 0.10
      + pacing               * 0.05
      + avg_transition_quality * 0.03
      + avg_cut_smoothness     * 0.01
      + avg_audio_sync_score   * 0.01
```

The response includes a full breakdown by component:

```json
{
  "score": 0.9082,
  "breakdown": {
    "engagement": 0.2920,
    "retention": 0.1456,
    "platform_compliance": 0.2000,
    "subtitles": 0.1000,
    "hook_strength": 0.1000,
    "pacing": 0.0337,
    "transition_quality": 0.0222,
    "cut_smoothness": 0.0072,
    "audio_sync": 0.0074,
    "final_score": 0.9082
  },
  "passed": true
}
```

The grader is fully deterministic — the same state always produces the same score, making evaluation reproducible across agents and runs.

---

## Baseline Agent

The `/baseline` endpoint runs a deterministic heuristic agent across all three tasks using seed=42. The strategy is:

1. Reorder scenes — place the highest hook_strength scene first
2. Boost hook — strengthen the opening
3. Cut scenes — remove filler/transition scenes with engagement below 0.30
4. Trim duration — if total duration exceeds the platform limit
5. Enhance pacing — smooth engagement transitions
6. Improve transition — if avg_transition_quality < 0.70
7. Smooth cut — if avg_cut_smoothness < 0.70
8. Sync audio — if avg_audio_sync_score < 0.70
9. Add subtitles
10. Add music

Baseline results (seed=42, platform=reels):

| Task | Score | Engagement | Retention | Steps |
|---|---|---|---|---|
| task_1 | 0.9082 | 0.834 | 0.970 | 9 |
| task_2 | 0.9082 | 0.834 | 0.970 | 9 |
| task_3 | 0.9082 | 0.834 | 0.970 | 9 |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode. Accepts `platform` and `seed` query params. |
| POST | `/step` | Apply one action. Returns state, reward, done, info. |
| GET | `/state` | Read current state without advancing the episode. |
| GET | `/tasks` | Return all three task definitions with target scores. |
| GET | `/grader` | Score the current state with full breakdown. |
| GET | `/baseline` | Run the deterministic baseline agent on all tasks. |
| GET | `/feedback` | Return AI-generated coaching tips based on current state. |

Interactive API documentation is available at `http://localhost:8000/docs`.

---

## Setup

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Start the server:**

```bash
python -m uvicorn app:app --reload
```

**Run the baseline agent client:**

```bash
python client.py --platform reels --seed 42
```

**Optional flags:**

```bash
python client.py --host http://localhost:8000 --platform shorts --seed 123
```

---

## Docker

**Build the image:**

```bash
docker build -t video-opt-env .
```

**Run the container:**

```bash
docker run -p 8000:8000 video-opt-env
```

The API will be available at `http://localhost:8000`.

---

## Hugging Face Deployment

This environment can be deployed as a Hugging Face Space using the Docker SDK. Once deployed, the Space acts as a persistent API endpoint — any agent can call `/reset`, `/step`, and `/grader` over HTTP without any local setup.

To deploy, push the repository to a Hugging Face Space configured with the Docker runtime. The `Dockerfile` is production-ready and requires no modification.

---

## Example Output

```
══════════════════════════════════════════════════════════════════════
  AI Short-Form Video Optimization  ·  v4.0
══════════════════════════════════════════════════════════════════════
  episode  : f55fd1dd-4035-4b2b-94e1-ab0d9222deae
  platform : reels   seed: 42
  scenes   : 5   duration: 18.83s
  engagement   : 0.3510
  retention    : 0.6395   watch_time: 12.2s
  hook_strength: 0.7720   pacing: 0.0635
  transition_q : 0.4202   cut_smooth: 0.4452   audio_sync: 0.4790

  step 01  | boost_hook           | r=+0.672 | eng=0.411 | ret=0.655
  step 02  | cut_scene            | r=+0.739 | eng=0.485 | ret=0.767
  step 03  | cut_scene            | r=+0.825 | eng=0.582 | ret=0.908
  step 04  | enhance_pacing       | r=+0.576 | eng=0.637 | ret=0.925
  step 05  | improve_transition   | r=+0.267 | eng=0.667 | tq=0.741
  step 06  | smooth_cut           | r=+0.247 | eng=0.687 | cs=0.724
  step 07  | sync_audio           | r=+0.276 | eng=0.717 | as=0.738
  step 08  | add_subtitles        | r=+0.462 | eng=0.777 | ret=0.953
  step 09  | add_music            | r=+0.729 | eng=0.834 | ret=0.970

  GRADER SCORE : 0.9082  PASSED
    engagement             : 0.2920
    retention              : 0.1456
    platform_compliance    : 0.2000
    subtitles              : 0.1000
    hook_strength          : 0.1000
    pacing                 : 0.0337
    transition_quality     : 0.0222
    cut_smoothness         : 0.0072
    audio_sync             : 0.0074

  AI FEEDBACK
  Excellent — your video is optimized for viral performance.
```

---

## Key Features

- **Real-world simulation** — models the actual decisions a video editor makes: cutting scenes, adjusting pacing, strengthening hooks, syncing audio
- **Multi-objective optimization** — agents must balance engagement, retention, production quality, and platform compliance simultaneously
- **Continuous reward shaping** — dense per-step rewards guide the agent toward better edits without requiring episode completion
- **Deterministic evaluation** — the grader produces identical scores for identical states, enabling fair comparison across agents
- **Platform-aware** — supports Reels (30s), Shorts (60s), and TikTok (60s) with platform-specific compliance rules
- **AI feedback system** — the `/feedback` endpoint generates human-readable coaching tips based on the current state
- **Extensible design** — new scene types, actions, platforms, and reward components can be added without breaking the OpenEnv interface

---

## Project Structure

```
.
├── app.py            # FastAPI server — all endpoints
├── environment.py    # Core RL environment logic
├── models.py         # Pydantic models (Action, Scene, Observation, State)
├── client.py         # Baseline agent client
├── requirements.txt  # Python dependencies
└── Dockerfile        # Container definition
```

---

## Future Improvements

- **Real video dataset integration** — replace synthetic scene generation with features extracted from actual short-form videos using computer vision models
- **Learning-based agents** — train PPO or DQN agents against this environment and compare against the heuristic baseline
- **Multi-platform optimization** — extend the reward function to jointly optimize for Reels, Shorts, and TikTok in a single episode
- **Temporal modeling** — add scene-level timestamps and model viewer attention as a time series rather than a static retention curve
- **Creator persona profiles** — parameterize the environment with audience demographics to simulate platform-specific viewer behavior

---

## Requirements

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
```

---

## License

MIT
