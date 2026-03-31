

# AI Short-Form Video Optimization Environment

An OpenEnv-compatible reinforcement learning environment for training agents to edit short-form videos for maximum viral engagement across Instagram Reels, YouTube Shorts, and TikTok.

---

## Why This Environment Matters

Short-form video is the dominant content format on the internet — over 500 hours of video are uploaded to YouTube every minute. On platforms like Reels, Shorts, and TikTok, the average viewer decides whether to keep watching within the first three seconds.

Creators face a genuinely hard multi-objective optimization problem:

- Cutting the wrong scene tanks engagement
- Poor pacing causes viewers to scroll away
- Weak hooks bury videos in the algorithm
- Duration violations reduce platform distribution

This environment gives RL agents a structured, reproducible testbed to learn the same editorial instincts that experienced human editors develop over years — through interaction, reward signals, and feedback.

---

## Key Features

- Multi-factor reward system covering engagement, retention, hook strength, pacing, transitions, cut smoothness, and audio sync
- Deterministic episodes via seed control — same seed always produces the same video
- Progressive difficulty with three benchmark tasks (Easy → Medium → Hard)
- OpenEnv-compatible API — standard reset / step / grader loop
- Built-in grader with weighted scoring and full breakdown
- AI feedback endpoint with actionable coaching tips
- Dockerized and fully reproducible
- Supports three platforms: Reels (30s), Shorts (60s), TikTok (60s)

---

## Live Demo

API: `https://saravanabalajisara-ai-video-optimizer-env.hf.space`

Swagger UI: `https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs`

ReDoc: `https://saravanabalajisara-ai-video-optimizer-env.hf.space/redoc`

Space: `https://huggingface.co/spaces/saravanabalajisara/ai-video-optimizer-env`

GitHub: `https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV`

---

## Baseline Performance

| Task   | Level  | Score  | Passed | Steps |
|--------|--------|--------|--------|-------|
| task_1 | Easy   | 0.9082 | ✅     | 9     |
| task_2 | Medium | 0.9082 | ✅     | 9     |
| task_3 | Hard   | 0.9082 | ✅     | 9     |

All three tasks pass with a single deterministic strategy in 9 steps — well within the 15-step limit.

---

## Architecture

```
FastAPI (app.py)
    │
    ├── VideoOptimizationEnv (environment.py)
    │       ├── Scene generation (seed-controlled)
    │       ├── Action handlers (10 actions)
    │       ├── Reward computation (dense, per-step)
    │       └── Observation builder (retention curve, pacing, quality metrics)
    │
    ├── Pydantic models (models.py)
    │       ├── Action — flexible input (action / action_type / type)
    │       ├── Scene, Observation, State
    │       └── Response shapes (StepResponse, GraderResponse, AIFeedback)
    │
    └── Docker (port 7860, non-root user, HF Spaces compatible)
```

---

## API Example

**Reset the environment:**

```json
POST /reset
{
  "platform": "reels",
  "seed": 42
}
```

**Apply an action:**

```json
POST /step
{
  "action": "boost_hook"
}
```

**With parameters:**

```json
POST /step
{
  "action": "cut_scene",
  "parameters": {"scene_id": "scene_2"}
}
```

**Get score:**

```
GET /grader
```

---

## API Endpoints

| Method | Endpoint    | Description                                      |
|--------|-------------|--------------------------------------------------|
| POST   | `/reset`    | Start a new episode. Accepts platform and seed.  |
| POST   | `/step`     | Apply one action. Returns state, reward, done.   |
| GET    | `/state`    | Read current state without advancing episode.    |
| GET    | `/tasks`    | Return all three task definitions.               |
| GET    | `/grader`   | Score the current state with full breakdown.     |
| GET    | `/baseline` | Run the deterministic baseline on all tasks.     |
| GET    | `/feedback` | AI coaching tips based on current state.         |
| GET    | `/docs`     | Interactive Swagger UI.                          |
| GET    | `/redoc`    | ReDoc API documentation.                         |

---

## Action Space

| Action               | Parameters                        | Effect                                                    |
|----------------------|-----------------------------------|-----------------------------------------------------------|
| `cut_scene`          | `{"scene_id": str}`               | Removes a scene. Penalizes removal of high-engagement scenes. |
| `reorder_scenes`     | `{"order": [str, ...]}`           | Reorders all scenes. Hook-first gives +0.02 global lift.  |
| `add_subtitles`      | `{}`                              | Enables subtitles. Lifts all scenes by +0.06.             |
| `add_music`          | `{"genre": str}`                  | Boosts hook/content scenes by +0.12, others by +0.05.     |
| `boost_hook`         | `{}`                              | hook_strength +0.30, engagement +0.20.                    |
| `trim_duration`      | `{"target_seconds": float}`       | Removes lowest-engagement scenes to hit duration target.  |
| `enhance_pacing`     | `{}`                              | Smooths engagement transitions. Global +0.02 lift.        |
| `improve_transition` | `{}`                              | transition_quality +0.15, engagement +0.03.               |
| `smooth_cut`         | `{}`                              | cut_smoothness +0.15, engagement +0.02.                   |
| `sync_audio`         | `{}`                              | audio_sync_score +0.15, engagement +0.03.                 |

All one-time actions are idempotent-guarded — repeating them returns a `-0.2` penalty.

---

## Observation Space

| Field                    | Description                                              |
|--------------------------|----------------------------------------------------------|
| `scenes`                 | Full list of scenes with per-scene quality metrics       |
| `current_engagement_score` | Mean engagement across all scenes (0.0–1.0)            |
| `avg_retention`          | Mean viewer retention across the episode (0.0–1.0)       |
| `retention_curve`        | Per-scene retention values simulating viewer drop-off    |
| `watch_time`             | Estimated watch time in seconds                          |
| `hook_strength`          | Opening hook strength of the first scene (0.0–1.0)       |
| `pacing_score`           | Smoothness of engagement transitions between scenes      |
| `avg_transition_quality` | Mean transition quality across all scenes                |
| `avg_cut_smoothness`     | Mean cut smoothness across all scenes                    |
| `avg_audio_sync_score`   | Mean audio sync quality across all scenes                |
| `total_duration`         | Total video length in seconds                            |
| `platform_compliant`     | True if duration is within the platform limit            |
| `hook_first`             | True if first scene is a hook/highlight with has_hook    |

---

## Reward Function

**Positive signals:**

| Signal                        | Weight                        |
|-------------------------------|-------------------------------|
| Engagement improvement        | delta × 2.5                   |
| Retention improvement         | delta × 2.0                   |
| Hook strength improvement     | delta × 1.5                   |
| Pacing improvement            | delta × 1.0                   |
| Transition quality            | value × 0.05 (continuous)     |
| Cut smoothness                | value × 0.05 (continuous)     |
| Audio sync                    | value × 0.05 (continuous)     |
| Platform compliance           | +0.20 on transition, +0.05 persistent |
| Subtitles present             | +0.10 persistent              |
| Hook-first structure          | +0.15 on transition, +0.03 persistent |
| Engagement crosses 0.80       | +0.20 milestone               |
| Hook strength crosses 0.70    | +0.15 milestone               |

**Penalties:**

| Condition                          | Penalty |
|------------------------------------|---------|
| Invalid scene_id in cut_scene      | -0.30   |
| Removing a high-engagement scene   | -0.20   |
| Repeating an idempotent action     | -0.20   |
| Fewer than 3 scenes remaining      | -0.10   |
| Action produces no measurable change | -0.05 |

---

## Tasks

| Task   | Level  | Target | Description                                                  |
|--------|--------|--------|--------------------------------------------------------------|
| task_1 | Easy   | 0.65   | Remove low-engagement scenes to raise avg engagement > 0.60  |
| task_2 | Medium | 0.78   | Platform compliance + retention ≥ 0.55 + production quality  |
| task_3 | Hard   | 0.875  | Full viral optimization — all metrics, correct action order  |

---

## Grader Formula

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

---

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

```bash
# Docker
docker build -t video-opt-env .
docker run -p 7860:7860 video-opt-env
```

---

## Project Structure

```
.
├── app.py            # FastAPI server — all endpoints
├── environment.py    # Core RL environment logic
├── models.py         # Pydantic models
├── client.py         # Baseline agent client
├── inference.py      # Inference runner
├── requirements.txt  # Dependencies
└── Dockerfile        # Container definition
```

---

## License

MIT
