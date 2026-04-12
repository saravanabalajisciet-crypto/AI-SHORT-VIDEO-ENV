---
title: AI Video Optimizer Env
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

<div align="center">

# 🎬 AI Short-Form Video Optimization
## A Decision-Constrained RL Environment for OpenEnv

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HF_Space-blue)](https://saravanabalajisara-ai-video-optimizer-env.hf.space/ui)
[![API Docs](https://img.shields.io/badge/📖_API_Docs-Swagger-green)](https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs)
[![GitHub](https://img.shields.io/badge/💻_GitHub-Source-black)](https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV)
[![OpenEnv](https://img.shields.io/badge/🤖_OpenEnv-Compatible-orange)](https://github.com/meta-pytorch/OpenEnv)

**The only OpenEnv environment where agents face irreversible decisions, commitment pressure, and real-world editorial consequences.**

</div>

---

## 🧠 What Makes This Different

Most RL environments are reversible. You can undo a move, retry a step, reset cleanly.

**This environment is not.**

```
cut_scene     → permanently removes content. Retention curve changes forever.
boost_hook    → one-time. You cannot un-boost.
finalize_edit → ends the episode immediately. No more steps.
```

The agent must learn **when to act, when to wait, and when to commit** — not just what to do. This is the core challenge that separates this environment from toy optimization tasks.

> **The hard RL problem:** Optimal policy under partial irreversibility, step budget constraints, and persona-specific scoring — where early mistakes compound and cannot be undone.

---

## 🎯 The Problem

Every day, 50 million creators on Instagram Reels, YouTube Shorts, and TikTok face the same hard problem: given raw footage, make editorial decisions that maximize reach.

The difference between viral and buried comes down to a handful of irreversible decisions:

| Decision | Why It's Hard |
|----------|--------------|
| Hook placement | Wrong order = buried by algorithm. Reordering after boosting wastes the boost permanently. |
| Scene cuts | Removing the wrong scene permanently damages the retention curve. |
| Duration compliance | 1 second over the platform limit = 35% less distribution. |
| Audience targeting | Gen-Z weights hook 2× more than brand. Same video, different optimal strategy. |
| When to finalize | Commit too early = miss efficiency bonus. Too late = over-editing penalty. |

This environment models that exact problem as a **sequential decision task with irreversible consequences**.

---

## 🔬 Why This Is a Hard RL Problem

| Property | This Environment |
|----------|-----------------|
| **Reversibility** | Partially irreversible — `cut_scene`, `boost_hook`, `finalize_edit` are permanent |
| **Commitment pressure** | `finalize_edit` ends episode — timing of commitment is a learnable skill |
| **Risk signal** | `risk_score` (0.0–1.0) tracks proximity to unrecoverable states at every step |
| **Soft caps** | Poor early decisions permanently limit maximum achievable score |
| **Order sensitivity** | Action sequence is graded — wrong order penalized even if all actions are correct |
| **Generalization** | Task 3: 3 seeds. Task 4: 5 seeds × 3 platforms. No memorization possible. |
| **Persona variation** | gen_z / millennial / brand — different scoring weights per episode |
| **Dense reward** | 12-component reward signal with efficiency bonus and order penalty |

**Heuristic ceiling:** A perfect rule-based agent scores 0.91 on task_3 but **fails task_3 (target 0.93) and task_4 (target 0.95)**. RL is required to pass the hard tasks.

---

## 🚀 Quick Start

```bash
pip install requests
```

```python
from client import VideoOptimizationEnv, VideoAction

# Connect to live HF Space — no setup needed
env = VideoOptimizationEnv.from_hf_space()

# Start episode
result = env.reset(platform="reels", seed=42)
print(result.engagement)          # 0.536
print(result.steps_remaining)     # 15
print(result.observation["risk_score"])  # 0.42 — medium risk
print(result.state["metadata"]["audience_persona"])  # gen_z

# Take actions
result = env.step(VideoAction("reorder_scenes", {"order": ["scene_0", "scene_2", "scene_1"]}))
result = env.step(VideoAction("boost_hook"))      # one-time, irreversible
result = env.step(VideoAction("enhance_pacing"))
result = env.step(VideoAction("add_subtitles"))
result = env.step(VideoAction("add_music"))

# Grade
score = env.grade()
print(score["score"])           # 0.91
print(score["rubric_score"])    # 0.91  ← use this for RL training (GRPO/PPO)
print(score["raw_score"])       # 0.89  ← pure observation quality
print(score["explanation"])     # "Strong result — passes hard task threshold..."
print(score["grader_metadata"]["risk_score"])  # 0.08 — low risk

# Get strategy plan
import requests
plan = requests.get("https://saravanabalajisara-ai-video-optimizer-env.hf.space/strategy").json()
print(plan["immediate_action"])   # "boost_hook"
print(plan["strategy_mode"])      # "aggressive"
print(plan["point_of_no_recovery"])  # False

# Commit (irreversible — ends episode, triggers persona bonus)
result = env.step(VideoAction("finalize_edit"))
print(result.info["persona_score"])  # 0.94
```

---

## 🏗️ Architecture

```
ai-video-optimizer-env/
├── app.py              # FastAPI — 15 endpoints + Gradio UI mount
├── environment.py      # Core RL logic: reset, step, reward, risk
├── models.py           # Typed Pydantic models throughout
├── strategy_engine.py  # Risk-aware action planner + score explainer
├── gradio_ui.py        # Live interactive demo at /ui
├── client.py           # VideoOptimizationEnv client (context manager)
├── inference.py        # LLM agent (OpenAI proxy compatible)
├── evaluate.py         # Benchmark runner — mirrors validator execution
├── test_environment.py # 52 pytest tests
├── scenario_config.json # Task verifiers with explicit pass/fail criteria
├── video_dataset.json  # 50 real-world video profiles, 30 niches
├── openenv.yaml        # OpenEnv spec v6
├── server/app.py       # OpenEnv entry point
└── Dockerfile          # python:3.11-slim, port 7860
```

---

## 📋 Tasks

| Task | Level | Target | Seeds | What Makes It Hard |
|------|-------|--------|-------|-------------------|
| task_1 | Easy | 0.65 | 1 | Remove filler, raise engagement above 0.60 |
| task_2 | Medium | 0.78 | 1 | Compliance + retention + production quality |
| task_3 | Hard | **0.90** | 3 (avg) | Generalize across seeds, correct action order required |
| task_4 | Elite | **0.95** | 5 × 3 platforms | No memorization possible, ≤ 8 steps |

**Heuristic baseline results from `/baseline`:**

| Agent | task_1 | task_2 | task_3 | task_4 |
|-------|--------|--------|--------|--------|
| Random (initial state) | 0.54 | 0.54 | 0.54 | 0.54 |
| Heuristic (optimal sequence) | 0.87 | 0.87 | 0.91 | 0.92 |
| **Target** | **0.65** | **0.78** | **0.90** | **0.95** |
| **Heuristic passes?** | ✅ | ✅ | ❌ | ❌ |

Tasks 3 and 4 require a learning agent. The heuristic cannot pass them.

---

## ⚡ Reward System

### Per-Step Dense Reward (12 components)

```python
reward = (engagement_delta  * 2.5)   # primary signal
       + (retention_delta   * 2.0)   # viewer drop-off
       + (hook_delta        * 1.5)   # first-impression strength
       + (pacing_delta      * 1.0)   # structural smoothness
       + (transition_quality * 0.05) # continuous quality signal
       + (cut_smoothness    * 0.05)
       + (audio_sync        * 0.05)
       + compliance_bonus           # +0.20 on transition, +0.05 persistent
       + subtitles_bonus            # +0.10 persistent
       + hook_first_bonus           # +0.15 on transition
       + milestone_bonuses          # +0.20 at eng>0.80, +0.15 at hook>0.70
       + persona_finalize_bonus     # up to +0.50 on finalize_edit
```

### Grader Formula (deterministic, weights sum to 1.0)

```
raw_score    = engagement*0.35 + retention*0.15 + compliance*0.20
             + subtitles*0.10 + hook*0.10 + pacing*0.05
             + transition*0.03 + cut*0.01 + audio*0.01

rubric_score = raw_score + efficiency_bonus + order_penalty
```

**Use `rubric_score` for RL training** — it's efficiency-adjusted and order-sensitive.

### Soft Score Caps (Decision Pressure)

Poor early decisions permanently limit maximum achievable score:

| Condition | Effect on rubric_score |
|-----------|----------------------|
| `hook_strength < 0.4` after step 3 | Capped at 0.60 |
| `avg_retention < 0.3` | × 0.7 multiplier |
| `platform_compliant == False` | − 0.10 |

`raw_score` is **never modified** — caps apply only to the RL training signal.

---

## 🎭 Audience Personas

Each episode is assigned a persona based on seed. Persona weights influence `finalize_edit` bonus:

| Persona | Hook Weight | Retention Weight | Compliance Weight | Strategy |
|---------|-------------|-----------------|-------------------|----------|
| `gen_z` | **0.45** | 0.20 | 0.10 | Prioritize hook first |
| `millennial` | 0.25 | **0.35** | 0.15 | Prioritize retention |
| `brand` | 0.20 | 0.20 | **0.40** | Prioritize compliance |

---

## 🔍 Risk-Aware Evaluation

Every observation includes `risk_score` — an advisory signal tracking proximity to irreversible failure:

```python
obs["risk_score"]  # 0.0 = safe, 1.0 = point of no recovery
```

| Risk Level | Range | Agent Posture |
|------------|-------|---------------|
| Low | 0.0 – 0.4 | Safe exploration — full sequence recommended |
| Medium | 0.4 – 0.7 | Careful optimization — avoid risky cuts |
| High | > 0.7 | Limited recovery — only safe multipliers |

`risk_score` does NOT affect reward directly. It is an advisory signal for decision-making.

### Irreversible Decision Point

When `finalize_edit` is called, the episode metadata records:
- `irreversible_decision_point: true`
- `finalized_at_step` — which step the agent committed
- `finalized_risk_score` — risk level at moment of commitment

This lets evaluators distinguish agents that commit confidently at low risk vs. agents that panic-finalize under pressure.

---

## 🛠️ API Reference (15 endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset` | Start episode. Accepts `platform`, `seed`, `simulate`, `session_id`. |
| POST | `/step` | Apply action. Returns `state`, `reward`, `done`, `info`. |
| GET | `/state` | Read current state without advancing. |
| GET | `/tasks` | All 4 task definitions (easy → elite). |
| GET | `/grader` | Score with `raw_score`, `rubric_score`, `explanation`, `grader_metadata`. |
| GET | `/baseline` | Heuristic + random agent scores across all tasks. |
| GET | `/hint` | Best next action with reasoning. |
| GET | `/strategy` | Full risk-aware action plan with persona focus. |
| GET | `/feedback` | AI coaching tips for current state. |
| GET | `/trajectory` | Full action log with `decision_quality` per step. |
| GET | `/efficiency` | Step efficiency rating (elite/optimal/good). |
| GET | `/scenarios` | 5 diverse scenario seeds across platforms. |
| GET | `/persona` | Current audience persona + scoring weights. |
| GET | `/dataset` | Real video dataset metadata + research citations. |
| GET | `/analyze_url` | **Live YouTube oEmbed API** — fetch real video metadata + scene breakdown. |
| GET | `/leaderboard` | Top scores across all graded episodes. |
| GET | `/ui` | **Live Gradio demo** — interactive environment explorer. |

---

## 🎮 Live Interactive Demo

Try the environment live at:

**[https://saravanabalajisara-ai-video-optimizer-env.hf.space/ui](https://saravanabalajisara-ai-video-optimizer-env.hf.space/ui)**

- Reset episode, choose platform and seed
- Take actions and watch engagement/risk/retention change in real time
- Grade the current state with full score breakdown and explanation
- Get the strategy engine's risk-aware action plan

---

## 🔧 Setup

**Local (no Docker):**
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker:**
```bash
docker build -t video-env:latest .
docker run -p 7860:7860 video-env:latest
```

**Run inference against HF Space:**
```bash
OPENENV_BASE_URL=https://saravanabalajisara-ai-video-optimizer-env.hf.space \
API_BASE_URL=https://api.openai.com/v1 \
API_KEY=your-key \
python inference.py
```

---

## 🧪 Testing

```bash
pip install pytest
# Start server first
pytest test_environment.py -v
```

52 tests covering: reset, step, all actions, observation fields, grader, tasks, hint, strategy, trajectory, session isolation, simulate mode, environment logic, strategy engine unit tests.

---

## 📊 Evaluation

`evaluate.py` is a standalone benchmark runner that mirrors validator execution:

```bash
# Against local server
python evaluate.py

# Against live HF Space
python evaluate.py --env-url https://saravanabalajisara-ai-video-optimizer-env.hf.space

# With LLM agent (fires one LLM step per episode)
python evaluate.py --use-llm --api-base-url <url> --api-key <key> --model gpt-4o-mini

# Single task
python evaluate.py --task task_3

# Save structured results
python evaluate.py --output results.json
```

**Exit codes:** `0` = all passed · `1` = some failed · `2` = env unreachable

Sample output:
```
════════════════════════════════════════════════════════════════════
  EVALUATION REPORT
────────────────────────────────────────────────────────────────────
  [PASS ✓] task_1    score=0.8821  target=0.65  level=easy     steps=9
  [PASS ✓] task_2    score=0.8821  target=0.78  level=medium   steps=9
  [PASS ✓] task_3    score=0.9012  target=0.90  level=hard     steps=9  seeds={42: 0.91, 7: 0.89, 13: 0.91}
────────────────────────────────────────────────────────────────────
  ALL TASKS PASSED ✓
  Average score: 0.8885  |  Elapsed: 18.3s
════════════════════════════════════════════════════════════════════
```

---

## 📊 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENENV_BASE_URL` | RL environment server URL (injected by validator) |
| `API_BASE_URL` | LLM proxy base URL (injected by validator) |
| `API_KEY` | LLM API key (injected by validator) |
| `MODEL_NAME` | LLM model identifier (default: `gpt-4o-mini`) |
| `HF_TOKEN` | HuggingFace token (fallback for API_KEY) |

---

## 📚 Real-World Grounding

Scene data is derived from 50 video profiles across 30 niches. The first 10 videos are grounded in published research:

- [opus.pro](https://www.opus.pro/blog/tiktok-length-format-retention-data) — 500 TikTok videos: 70% retention = 4.3× more impressions
- [socialinsider.io](https://www.socialinsider.io/social-media-benchmarks/social-media-video-statistics) — 2025 Social Media Video Performance Statistics
- [vidico.com](https://vidico.com/news/instagram-reels-statistics) — Reels organic reach dropped 50% in 2023; hook-first is mandatory
- [dmnews.com](https://dmnews.com/video-marketing-works-until-you-realize-no-ones-watching-past-second-three) — 75% of viewers click away before midpoint

---

## 🔗 Links

| | |
|--|--|
| 🚀 Live API | https://saravanabalajisara-ai-video-optimizer-env.hf.space |
| 📖 Swagger Docs | https://saravanabalajisara-ai-video-optimizer-env.hf.space/docs |
| 🎮 Interactive Demo | https://saravanabalajisara-ai-video-optimizer-env.hf.space/ui |
| 🤗 HF Space | https://huggingface.co/spaces/saravanabalajisara/ai-video-optimizer-env |
| 💻 GitHub | https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV |

---

## 📄 License

MIT

---

## 📖 Citation

```bibtex
@misc{ai-video-optimizer-env-2026,
  author  = {SaravanaBalaji},
  title   = {AI Short-Form Video Optimization: A Decision-Constrained RL Environment},
  year    = {2026},
  url     = {https://github.com/saravanabalajisciet-crypto/AI-SHORT-VIDEO-ENV},
  note    = {OpenEnv-compatible environment for training agents under partial
             irreversibility, commitment pressure, and persona-specific scoring.}
}
```
