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

An OpenEnv-compatible reinforcement learning environment where agents learn to edit short-form videos for maximum viral engagement across Instagram Reels, YouTube Shorts, and TikTok.

---

## Why This Matters

Every day, over 2 billion short-form videos are watched on Reels, Shorts, and TikTok. The difference between a video that goes viral and one that gets buried comes down to a handful of editorial decisions made in the first edit — scene order, hook strength, pacing, duration compliance, and production quality.

This environment models that exact problem as a **sequential decision task with real-world consequences**:

- Cutting the wrong scene is **irreversible** — it permanently changes the retention curve
- Hook placement determines **everything** — the algorithm scores the first 3 seconds
- Action order matters — boosting a hook before reordering wastes the boost permanently
- Over-editing is penalized — efficiency is rewarded just like in real production
- Each platform (Reels 30s / Shorts 60s / TikTok 60s) has hard duration constraints

Agents must learn **editorial judgment** — the same skill human editors develop over years — through dense reward signals, structured feedback, and multi-scenario evaluation across 5 diverse video scenarios.

---

## Tasks

| Task   | Level  | Target | Description |
|--------|--------|--------|-------------|
| task_1 | Easy   | 0.65   | Remove low-engagement filler scenes, raise avg engagement > 0.60 |
| task_2 | Medium | 0.78   | Platform compliance + retention >= 0.55 + production quality |
| task_3 | Hard   | 0.875  | Full viral optimization — all metrics, correct action order, efficiency bonus |

**5 Scenario seeds** (via `/scenarios`): viral hook, retention crisis, duration overrun, weak production, TikTok challenge — across all 3 platforms.

---

## Reward & Grader

**Dense per-step reward (agent gets signal after every action):**

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

**Penalties:** invalid actions (-0.30), removing high-engagement scenes (-0.20), repeated idempotent actions (-0.20), over-editing (-0.10).

**Grader formula (deterministic, weights sum to 1.0):**

```
score = engagement * 0.35 + retention * 0.15 + platform_compliance * 0.20
      + subtitles * 0.10 + hook_strength * 0.10 + pacing * 0.05
      + transition_quality * 0.03 + cut_smoothness * 0.01 + audio_sync * 0.01
      + efficiency_bonus (up to +0.02 for solving in <= 7 steps)
```

**Efficiency tiers:** Elite (>= 0.875 in <= 7 steps) → Optimal → Good → Acceptable.

---

## Baseline Performance

| Task   | Level  | Score  | Passed | Steps |
|--------|--------|--------|--------|-------|
| task_1 | Easy   | 0.9182 | ✅     | 9     |
| task_2 | Medium | 0.9182 | ✅     | 9     |
| task_3 | Hard   | 0.9182 | ✅     | 9     |

---

## API Endpoints

| Method | Endpoint      | Description |
|--------|---------------|-------------|
| POST   | `/reset`      | Start new episode. Accepts platform and seed. |
| POST   | `/step`       | Apply one action. Returns state, reward, done, info. |
| GET    | `/state`      | Read current state without advancing episode. |
| GET    | `/tasks`      | Return all three task definitions. |
| GET    | `/grader`     | Score current state with full breakdown. |
| GET    | `/baseline`   | Run deterministic baseline on all tasks. |
| GET    | `/feedback`   | AI coaching tips based on current state. |
| GET    | `/hint`       | Best next action suggestion with reasoning. |
| GET    | `/scenarios`  | 5 diverse scenario seeds across platforms. |
| GET    | `/trajectory` | Full episode action log with per-step metrics. |
| GET    | `/efficiency` | Step efficiency rating and score-per-step analysis. |
| GET    | `/docs`       | Interactive Swagger UI. |

---

## Action Space

| Action | Parameters | Effect |
|--------|-----------|--------|
| `cut_scene` | `{"scene_id": str}` | Remove scene. Penalizes high-engagement cuts. |
| `reorder_scenes` | `{"order": [str]}` | Reorder scenes. Hook-first gives +0.02 lift. |
| `add_subtitles` | `{}` | Enable subtitles. +0.06 engagement lift. |
| `add_music` | `{}` | Add music. +0.12 on hook/content scenes. |
| `boost_hook` | `{}` | hook_strength +0.30, engagement +0.20. |
| `trim_duration` | `{"target_seconds": float}` | Trim to platform limit. |
| `enhance_pacing` | `{}` | Smooth engagement transitions. |
| `improve_transition` | `{}` | transition_quality +0.15. |
| `smooth_cut` | `{}` | cut_smoothness +0.15. |
| `sync_audio` | `{}` | audio_sync_score +0.15. |

---

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

```bash
# Docker
docker build -t video-opt-env .
docker run -p 7860:7860 video-opt-env
```

```bash
# Inference
ENV_URL=http://localhost:7860 python inference.py
```

---

## Live URLs

- API: `https://saravanabalajisara-ai-video-optimizer-env.hf.space`
- Docs: `https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs`
- Space: `https://huggingface.co/spaces/saravanabalajisara/ai-video-optimizer-env`
- GitHub: `https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV`

---

## Project Structure

```
.
├── app.py            # FastAPI server — all endpoints
├── environment.py    # Core RL environment logic
├── models.py         # Pydantic models
├── inference.py      # LLM-powered inference script
├── server/app.py     # OpenEnv entry point
├── requirements.txt  # Pinned dependencies
├── openenv.yaml      # OpenEnv spec
└── Dockerfile        # Container (python:3.11-slim-bullseye, port 7860)
```

---

## License

MIT
