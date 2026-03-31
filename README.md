---
title: ai-video-optimizer-env
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# AI Short-Form Video Optimization Environment

An OpenEnv-compatible reinforcement learning environment for optimizing short-form videos (Reels / Shorts / TikTok).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start a new episode |
| POST | `/step` | Apply one action |
| GET | `/state` | Read current state |
| GET | `/tasks` | Get benchmark tasks |
| GET | `/grader` | Score current state |
| GET | `/baseline` | Run baseline agent |
| GET | `/feedback` | Get AI coaching tips |
| GET | `/docs` | Interactive API docs |

## Quick Start

```bash
# Reset environment
curl -X POST "https://saravanabalajisara-ai-video-optimizer-env.hf.space/reset?platform=reels&seed=42"

# Apply action
curl -X POST "https://saravanabalajisara-ai-video-optimizer-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "boost_hook", "parameters": {}}'

# Get score
curl "https://saravanabalajisara-ai-video-optimizer-env.hf.space/grader"
```

## API Example

```json
POST /reset
{
  "platform": "reels",
  "seed": 42
}
```

```json
POST /step
{
  "action": "boost_hook"
}
```

## Action Space

| Action | Parameters |
|---|---|
| `cut_scene` | `{"scene_id": str}` |
| `reorder_scenes` | `{"order": [str, ...]}` |
| `add_subtitles` | `{}` |
| `add_music` | `{"genre": str}` |
| `boost_hook` | `{}` |
| `trim_duration` | `{"target_seconds": float}` |
| `enhance_pacing` | `{}` |
| `improve_transition` | `{}` |
| `smooth_cut` | `{}` |
| `sync_audio` | `{}` |

## Tasks

| Task | Level | Target Score |
|---|---|---|
| task_1 | Easy | 0.65 |
| task_2 | Medium | 0.78 |
| task_3 | Hard | 0.875 |

## Grader Formula

```
score = engagement * 0.35 + retention * 0.15 + platform_compliance * 0.20
      + subtitles * 0.10 + hook_strength * 0.10 + pacing * 0.05
      + avg_transition_quality * 0.03 + avg_cut_smoothness * 0.01
      + avg_audio_sync_score * 0.01
```

## License

MIT
