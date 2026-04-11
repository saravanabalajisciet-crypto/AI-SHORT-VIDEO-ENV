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

## Why This Matters

The creator economy is a **$250 billion industry**. Every day, 50 million creators on Instagram Reels, YouTube Shorts, and TikTok face the same hard problem: given raw footage, make editorial decisions that maximize reach.

The difference between a video that goes viral and one that gets buried comes down to a handful of decisions made in the first edit:

- **Hook placement** — the algorithm scores the first 3 seconds. Wrong order = buried.
- **Scene cuts** — removing the wrong scene is irreversible. It permanently changes the retention curve.
- **Duration compliance** — 1 second over the platform limit = 35% less distribution.
- **Pacing** — uneven engagement transitions cause viewers to scroll away mid-video.
- **Audience targeting** — a Gen-Z audience weights hook strength 2x more than a brand audience.

This environment models that exact problem as a **sequential decision task with real-world consequences**. Agents must learn editorial judgment — the same skill human editors develop over years — through dense reward signals, real video data, and multi-scenario evaluation.

**Real-world grounding:** Scene data is derived from 50 video profiles across 30 niches. The first 10 videos are directly grounded in published research:
- [opus.pro](https://www.opus.pro/blog/tiktok-length-format-retention-data): Analysis of 500 TikTok videos — 70% retention = 4.3x more impressions than 40% retention
- [creatorsjet.com](https://www.creatorsjet.com/blog/best-instagram-reel-length-for-engagement-based-on-500-viral-videos): 500 viral Instagram Reels engagement analysis
- [socialinsider.io](https://www.socialinsider.io/social-media-benchmarks/social-media-video-statistics): 2025 Social Media Video Performance Statistics
- [vidico.com](https://vidico.com/news/instagram-reels-statistics): Reels organic reach dropped 50% in 2023 — hook-first is now mandatory
- [dmnews.com](https://dmnews.com/video-marketing-works-until-you-realize-no-ones-watching-past-second-three): 75% of viewers click away before midpoint

Key findings encoded in the environment:
- Hook window: first **3 seconds** determine viewer retention
- Optimal TikTok length: **21-34s** for completion rate
- 70% retention → **4.3x more impressions** than 40% retention
- Pattern interrupts every **3-5 seconds** maintain attention
- Poor audio sync reduces watch time by **23%**

---

## Baseline Performance

| Agent | task_1 | task_2 | task_3 (avg 3 seeds) | Notes |
|-------|--------|--------|----------------------|-------|
| Random agent | 0.31 | 0.28 | 0.24 | Random action each step |
| Worst case (all fillers kept) | 0.18 | 0.15 | 0.12 | No cuts, no hook boost |
| Greedy (always boost_hook only) | 0.52 | 0.48 | 0.41 | Single action repeated |
| Partial (hook + subtitles only) | 0.71 | 0.68 | 0.62 | 3-step partial strategy |
| Heuristic baseline | 0.87 | 0.87 | 0.92 | Full optimal sequence |
| GPT-4o-mini (via proxy) | ~0.82 | ~0.79 | ~0.74 | LLM with hint endpoint |

The environment has a **4x difficulty range** — random agents score ~0.28, optimal heuristic scores ~0.92. task_3 requires correct action sequencing across 3 diverse seeds to pass, making it genuinely hard for frontier models without reasoning.

---

## Tasks

| Task | Level | Target | Description |
|------|-------|--------|-------------|
| task_1 | Easy | 0.65 | Remove low-engagement filler scenes, raise avg engagement > 0.60 |
| task_2 | Medium | 0.78 | Platform compliance + retention >= 0.55 + production quality |
| task_3 | Hard | 0.92 | Full viral optimization across 3 seeds (42, 7, 13) — must generalize |

task_3 is evaluated as the **average score across seeds 42, 7, and 13**. A simple heuristic that memorizes seed=42 will fail on seeds 7 and 13. Agents must generalize.

---

## Novel Mechanics

**Audience Personas** — each episode is assigned an audience (gen_z / millennial / brand) based on seed. Each persona weights hook_strength, retention, engagement, and compliance differently. The `finalize_edit` action triggers persona-weighted final scoring.

**Irreversible Actions** — `cut_scene` permanently removes a scene. `finalize_edit` locks the episode and ends it immediately. Agents must plan ahead, not just react.

**Real Video Dataset** — 50 video profiles across 30 niches and 3 platforms. Each seed maps to a specific real-world-inspired video structure. Call `/dataset` to inspect.

**Step Efficiency Scoring** — solving in <= 7 steps earns a +0.02 bonus. Over-editing (> 13 steps) is penalized. Efficiency is rewarded just like in real production.

---

## Reward & Grader

**Dense per-step reward (12 signal components):**

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

**Grader formula:**
```
score = engagement * 0.35 + retention * 0.15 + platform_compliance * 0.20
      + subtitles * 0.10 + hook_strength * 0.10 + pacing * 0.05
      + transition_quality * 0.03 + cut_smoothness * 0.01 + audio_sync * 0.01
      + efficiency_bonus (up to +0.02 for <= 7 steps)
```

---

## Action Space (11 actions)

| Action | Parameters | Effect | Reversible? |
|--------|-----------|--------|-------------|
| `cut_scene` | `{"scene_id": str}` | Remove scene. Penalizes high-engagement cuts. | No |
| `reorder_scenes` | `{"order": [str]}` | Reorder scenes. Hook-first gives +0.02 lift. | Yes |
| `add_subtitles` | `{}` | +0.06 engagement lift. | Yes (one-time) |
| `add_music` | `{}` | +0.12 on hook/content scenes. | Yes (one-time) |
| `boost_hook` | `{}` | hook_strength +0.30, engagement +0.20. | Yes (one-time) |
| `trim_duration` | `{"target_seconds": float}` | Trim to platform limit. | No |
| `enhance_pacing` | `{}` | Smooth engagement transitions. | Yes (one-time) |
| `improve_transition` | `{}` | transition_quality +0.15. | Yes (one-time) |
| `smooth_cut` | `{}` | cut_smoothness +0.15. | Yes (one-time) |
| `sync_audio` | `{}` | audio_sync_score +0.15. | Yes (one-time) |
| `finalize_edit` | `{}` | **IRREVERSIBLE.** Locks episode, triggers persona-weighted scoring. | No |

---

## API Endpoints (14 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start new episode. Accepts platform, seed. |
| POST | `/step` | Apply one action. Returns state, reward, done, info. |
| GET | `/state` | Read current state. |
| GET | `/tasks` | All 3 task definitions. |
| GET | `/grader` | Score current state. Records to leaderboard. |
| GET | `/baseline` | Deterministic baseline across all tasks. |
| GET | `/feedback` | AI coaching tips. |
| GET | `/hint` | Best next action with reasoning. |
| GET | `/scenarios` | 5 diverse scenario seeds. |
| GET | `/trajectory` | Full episode action log. |
| GET | `/efficiency` | Step efficiency rating. |
| GET | `/dataset` | Real video dataset metadata (50 videos, 30 niches). |
| GET | `/persona` | Current episode audience persona + scoring weights. |
| GET | `/leaderboard` | Top scores across all graded episodes. |

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
# Inference (validator format)
ENV_URL=http://localhost:7860 \
API_BASE_URL=https://api.openai.com/v1 \
API_KEY=your-key \
python inference.py
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
├── app.py              # FastAPI server — 14 endpoints
├── environment.py      # Core RL environment logic
├── models.py           # Pydantic models (typed)
├── inference.py        # LLM-powered inference script
├── video_dataset.json  # 50 real-world video profiles, 30 niches
├── server/app.py       # OpenEnv entry point
├── requirements.txt    # Pinned dependencies
├── openenv.yaml        # OpenEnv spec v6
└── Dockerfile          # python:3.11-slim-bullseye, port 7860
```

---

## License

MIT
