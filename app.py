from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
import time
import uuid
import json
import os

from models import (
    Action, State, StepResponse, GraderResponse,
    TaskDefinition, BaselineResult, AIFeedback, ActionType,
)
from environment import VideoOptimizationEnv, PLATFORM_LIMITS, generate_feedback

app = FastAPI(
    title="AI Short-Form Video Optimization Environment",
    description=(
        "OpenEnv-compatible RL environment for optimizing Reels / Shorts / TikTok. "
        "Features retention curves, hook strength, pacing, transition quality, "
        "cut smoothness, audio sync, AI feedback, scenario-based tasks, "
        "action budgets, step efficiency scoring, and trajectory logging."
    ),
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── FIX 2: Per-session environment isolation ──────────────────────────────────
# Each session gets its own env instance keyed by episode_id.
# Prevents concurrent request state corruption.
_sessions: Dict[str, Dict[str, Any]] = {}
_SESSION_TTL = 3600  # 1 hour

# Fallback single env for clients that don't use session_id
_default_env = VideoOptimizationEnv(platform="reels", seed=42)
_default_trajectory: List[Dict[str, Any]] = []


def _get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session."""
    now = time.time()
    # Evict stale sessions
    stale = [k for k, v in _sessions.items() if now - v["created"] > _SESSION_TTL]
    for k in stale:
        del _sessions[k]
    if session_id not in _sessions:
        _sessions[session_id] = {
            "env": VideoOptimizationEnv(platform="reels", seed=42),
            "trajectory": [],
            "created": now,
        }
    return _sessions[session_id]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "Invalid request payload."
    return JSONResponse(status_code=400, content={"detail": msg})


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "AI Short-Form Video Optimization Environment",
        "version": "6.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/grader", "/baseline",
            "/feedback", "/hint", "/scenarios", "/trajectory", "/efficiency",
            "/dataset", "/persona", "/leaderboard",
        ],
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ── /reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset", response_model=State, tags=["Environment"])
async def reset(
    request: Request,
    platform: Literal["reels", "shorts", "tiktok"] = Query("reels"),
    seed: int = Query(42, description="RNG seed for reproducibility"),
):
    """
    Start a new episode. Returns the initial state.
    Accepts query params OR JSON body.
    Optionally pass session_id in body for isolated multi-agent sessions.
    """
    global _default_trajectory
    session_id = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            platform = body.get("platform", platform)
            seed = int(body.get("seed", seed))
            session_id = body.get("session_id")
    except Exception:
        pass

    if platform not in ("reels", "shorts", "tiktok"):
        raise HTTPException(status_code=400, detail=f"Invalid platform '{platform}'. Choose: reels, shorts, tiktok")

    if session_id:
        sess = _get_session(session_id)
        sess["env"].platform = platform
        sess["env"].seed = seed
        state = sess["env"].reset()
        sess["trajectory"] = []
    else:
        _default_env.platform = platform
        _default_env.seed = seed
        state = _default_env.reset()
        _default_trajectory = []

    return state


# ── /step ──────────────────────────────────────────────────────────────────────
@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: Request):
    """
    Apply one action. Returns (state, reward, done, info).
    Accepts: {"action_type": "boost_hook"} or {"action": "boost_hook"}
    """
    global _default_trajectory
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    session_id = body.pop("session_id", None) if isinstance(body, dict) else None

    try:
        action = Action(**body)
    except ValidationError as e:
        errors = e.errors()
        msg = errors[0]["msg"] if errors else "Invalid action payload."
        raise HTTPException(status_code=400, detail=msg)

    if session_id:
        sess = _get_session(session_id)
        env = sess["env"]
        traj = sess["trajectory"]
    else:
        env = _default_env
        traj = _default_trajectory

    try:
        state, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    traj.append({
        "step": state.step_count,
        "action": action.action_type.value,
        "parameters": action.parameters,
        "reward": reward,
        "valid": info.get("valid", True),
        "engagement": state.observation.current_engagement_score,
        "retention": state.observation.avg_retention,
        "hook_strength": state.observation.hook_strength,
        "done": done,
    })

    return StepResponse(state=state, reward=reward, done=done, info=info)


# ── /state ─────────────────────────────────────────────────────────────────────
@app.get("/state", response_model=State, tags=["Environment"])
def get_state():
    try:
        return _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks", response_model=List[TaskDefinition], tags=["Benchmark"])
def get_tasks():
    return [
        TaskDefinition(
            id="task_1",
            level="easy",
            description=(
                "Remove low-engagement scenes (score < 0.30) to raise average engagement "
                "above 0.60. Filler and transition scenes drag the average down."
            ),
            expected_behavior=(
                "Agent cuts scenes with engagement_score < 0.30, "
                "then adds subtitles and music to lift remaining scores."
            ),
            evaluation_criteria=(
                "current_engagement_score >= 0.60 AND platform_compliant == True. "
                "Grader score >= 0.65."
            ),
            target_score=0.65,
        ),
        TaskDefinition(
            id="task_2",
            level="medium",
            description=(
                "Optimize for platform compliance and viewer retention: "
                "total_duration <= 30s, subtitles present, avg_retention >= 0.55. "
                "Also improve production quality (transition, cut, audio)."
            ),
            expected_behavior=(
                "Agent trims duration, adds subtitles and music, calls enhance_pacing, "
                "and uses improve_transition / smooth_cut / sync_audio."
            ),
            evaluation_criteria=(
                "platform_compliant == True AND subtitles_present == True AND "
                "avg_retention >= 0.55 AND current_engagement_score >= 0.65. "
                "Grader score >= 0.78."
            ),
            target_score=0.78,
        ),
        TaskDefinition(
            id="task_3",
            level="hard",
            description=(
                "Full viral optimization across 3 diverse seeds (42, 7, 13). "
                "Must achieve engagement >= 0.82, avg_retention >= 0.92, "
                "hook_strength >= 0.75, all production metrics >= 0.72, "
                "platform compliance, subtitles, hook-first ordering — "
                "AND solve efficiently in <= 10 steps per seed. "
                "Requires correct action sequencing AND generalisation across seeds."
            ),
            expected_behavior=(
                "Agent must: reorder_scenes -> boost_hook -> cut_scene(s) -> "
                "enhance_pacing -> improve_transition -> smooth_cut -> sync_audio -> "
                "add_subtitles -> add_music. Order matters. Must work across seeds."
            ),
            evaluation_criteria=(
                "current_engagement_score >= 0.82 AND avg_retention >= 0.92 AND "
                "hook_strength >= 0.75 AND avg_transition_quality >= 0.72 AND "
                "platform_compliant == True AND subtitles_present == True AND "
                "steps <= 10. Grader score >= 0.90 averaged across seeds 42, 7, 13."
            ),
            target_score=0.90,
        ),
    ]


# ── /grader ────────────────────────────────────────────────────────────────────
@app.get("/grader", response_model=GraderResponse, tags=["Benchmark"])
def grader():
    """
    Deterministic multi-factor scoring. Weights sum to 1.0.
    Includes step efficiency bonus. Records to leaderboard.
    """
    try:
        state = _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    order_score = _default_env.order_score()
    score, breakdown, raw_score, rubric_score = _compute_score(obs, state.step_count, order_score)

    # NEW ADDITION: apply soft caps post-processing (raw_score untouched)
    from environment import _compute_risk_score
    risk = getattr(obs, "risk_score", None) or _compute_risk_score(
        obs.hook_strength, obs.avg_retention, obs.pacing_score
    )
    capped_score, cap_applied, cap_reason = _apply_soft_caps(score, obs, state.step_count)
    score = capped_score
    breakdown["final_score"] = score

    grader_metadata = {
        "risk_score": risk,
        "score_cap_applied": cap_applied,
        "cap_reason": cap_reason,
        "irreversible_decision_point": state.metadata.get("irreversible_decision_point", False),
        "finalized_at_step": state.metadata.get("finalized_at_step", None),
        "finalized_risk_score": state.metadata.get("finalized_risk_score", None),
    }

    # Record to leaderboard
    _leaderboard.append({
        "score": score,
        "raw_score": raw_score,
        "rubric_score": rubric_score,
        "steps": state.step_count,
        "seed": state.metadata.get("seed", 42),
        "persona": state.metadata.get("audience_persona", "unknown"),
        "niche": state.metadata.get("niche", "unknown"),
        "engagement": round(obs.current_engagement_score, 4),
        "retention": round(obs.avg_retention, 4),
        "order_score": round(order_score, 4),
        "timestamp": time.time(),
    })
    # Keep only last 100 for memory
    if len(_leaderboard) > 100:
        _leaderboard.pop(0)

    # Probe vs Trainable (inspired by CARLA's ethical scenario distinction)
    # Probe: always score 1.0 for RL, but track agent choice as metric
    # Trainable: reward depends on performance
    task_type = "trainable"

    return GraderResponse(
        score=score,
        breakdown=breakdown,
        passed=score >= 0.875,
        raw_score=raw_score,
        rubric_score=rubric_score,
        task_type=task_type,
        grader_metadata=grader_metadata,
    )


def _compute_score(obs, step_count: int = 0, order_score: float = 1.0) -> tuple:
    eng        = obs.current_engagement_score
    retention  = obs.avg_retention
    compliance = 1.0 if obs.platform_compliant else 0.0
    subtitles  = 1.0 if obs.subtitles_present  else 0.0
    hook       = obs.hook_strength
    pacing     = obs.pacing_score
    tq         = obs.avg_transition_quality
    cs         = obs.avg_cut_smoothness
    asy        = obs.avg_audio_sync_score

    raw = (
        eng        * 0.35 +
        retention  * 0.15 +
        compliance * 0.20 +
        subtitles  * 0.10 +
        hook       * 0.10 +
        pacing     * 0.05 +
        tq         * 0.03 +
        cs         * 0.01 +
        asy        * 0.01
    )

    efficiency_bonus = 0.0
    if step_count > 0:
        if step_count <= 7:
            efficiency_bonus = 0.02
        elif step_count <= 10:
            efficiency_bonus = 0.01
        elif step_count > 13:
            efficiency_bonus = -0.01

    order_penalty = 0.0
    if order_score < 0.5:
        order_penalty = -0.08
    elif order_score < 0.8:
        order_penalty = -0.03

    # rubric_score = RL training signal (efficiency + order aware)
    rubric_score = round(min(raw + efficiency_bonus + order_penalty, 1.0), 4)
    # raw_score = pure observation quality (no step bonuses)
    raw_score = round(min(raw, 1.0), 4)
    # final score = rubric_score (what we report)
    score = rubric_score

    breakdown = {
        "engagement":          round(eng        * 0.35, 4),
        "retention":           round(retention  * 0.15, 4),
        "platform_compliance": round(compliance * 0.20, 4),
        "subtitles":           round(subtitles  * 0.10, 4),
        "hook_strength":       round(hook       * 0.10, 4),
        "pacing":              round(pacing     * 0.05, 4),
        "transition_quality":  round(tq         * 0.03, 4),
        "cut_smoothness":      round(cs         * 0.01, 4),
        "audio_sync":          round(asy        * 0.01, 4),
        "efficiency_bonus":    round(efficiency_bonus, 4),
        "order_score":         round(order_score, 4),
        "order_penalty":       round(order_penalty, 4),
        "raw_score":           raw_score,
        "rubric_score":        rubric_score,
        "final_score":         score,
    }
    return score, breakdown, raw_score, rubric_score


# ── NEW ADDITION: soft score caps (post-processing only, raw_score untouched) ──
def _apply_soft_caps(rubric_score: float, obs, step_count: int) -> tuple:
    """
    Apply decision-pressure soft caps to rubric_score only.
    raw_score is NEVER modified. Returns (capped_score, cap_applied, cap_reason).
    """
    score = rubric_score
    cap_applied = False
    cap_reason = None

    # Cap 1: weak hook after step 3 → ceiling 0.60
    if step_count > 3 and obs.hook_strength < 0.4:
        if score > 0.60:
            score = 0.60
            cap_applied = True
            cap_reason = "low_hook_strength"

    # Cap 2: critically low retention → 0.7x multiplier
    if obs.avg_retention < 0.3:
        score = round(score * 0.7, 4)
        cap_applied = True
        cap_reason = cap_reason or "low_retention"

    # Cap 3: non-compliant at finalize → flat penalty
    if not obs.platform_compliant:
        score = round(max(0.0, score - 0.10), 4)
        cap_applied = True
        cap_reason = cap_reason or "platform_non_compliant"

    return round(score, 4), cap_applied, cap_reason


# ── /feedback ──────────────────────────────────────────────────────────────────
@app.get("/feedback", response_model=AIFeedback, tags=["Tools"])
def feedback():
    try:
        state = _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    obs = state.observation
    score, _, _, _ = _compute_score(obs, state.step_count)
    return generate_feedback(obs, score)


# ── /hint ──────────────────────────────────────────────────────────────────────
@app.get("/hint", tags=["Tools"])
def hint():
    """Best next action suggestion with reasoning. Helps LLM agents plan."""
    try:
        state = _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    suggestions = []

    if not obs.hook_first or obs.hook_strength < 0.5:
        suggestions.append({"action": "reorder_scenes", "priority": 10,
            "reason": "No hook-first scene. Reorder to put highest hook_strength scene first."})
    if obs.hook_strength < 0.75:
        suggestions.append({"action": "boost_hook", "priority": 9,
            "reason": f"hook_strength={obs.hook_strength:.2f} < 0.75. Boost for +0.30 lift."})
    if not obs.platform_compliant:
        suggestions.append({"action": "trim_duration", "priority": 8,
            "reason": f"Duration {obs.total_duration:.1f}s exceeds limit. Trim to 30s."})

    low_scenes = [s for s in obs.scenes
                  if s.engagement_score < 0.30 and s.scene_type in ("filler", "transition")]
    if low_scenes and len(obs.scenes) > 3:
        worst = min(low_scenes, key=lambda s: s.engagement_score)
        suggestions.append({"action": "cut_scene", "priority": 8,
            "parameters": {"scene_id": worst.id},
            "reason": f"Scene {worst.id} engagement={worst.engagement_score:.2f}. Cut it."})

    if obs.avg_retention < 0.6:
        suggestions.append({"action": "enhance_pacing", "priority": 7,
            "reason": f"avg_retention={obs.avg_retention:.2f} < 0.60."})
    if obs.avg_transition_quality < 0.72:
        suggestions.append({"action": "improve_transition", "priority": 6,
            "reason": f"avg_transition_quality={obs.avg_transition_quality:.2f} < 0.72."})
    if obs.avg_cut_smoothness < 0.72:
        suggestions.append({"action": "smooth_cut", "priority": 5,
            "reason": f"avg_cut_smoothness={obs.avg_cut_smoothness:.2f} < 0.72."})
    if obs.avg_audio_sync_score < 0.72:
        suggestions.append({"action": "sync_audio", "priority": 4,
            "reason": f"avg_audio_sync_score={obs.avg_audio_sync_score:.2f} < 0.72."})
    if not obs.subtitles_present:
        suggestions.append({"action": "add_subtitles", "priority": 3,
            "reason": "Subtitles absent. +0.10 grader score."})
    if not obs.music_added:
        suggestions.append({"action": "add_music", "priority": 2,
            "reason": "Music absent. +0.12 on hook/content scenes."})

    if not suggestions:
        return {"best_action": None, "reason": "Well optimized. Call /grader.",
                "all_suggestions": [], "steps_used": state.step_count,
                "steps_remaining": state.max_steps - state.step_count}

    suggestions.sort(key=lambda x: x["priority"], reverse=True)
    best = suggestions[0]
    return {
        "best_action": best["action"],
        "parameters": best.get("parameters", {}),
        "reason": best["reason"],
        "all_suggestions": suggestions,
        "steps_used": state.step_count,
        "steps_remaining": state.max_steps - state.step_count,
    }


# ── /leaderboard (NEW) ─────────────────────────────────────────────────────────
_leaderboard: List[Dict[str, Any]] = []
_leaderboard_max = 20


@app.get("/leaderboard", tags=["Benchmark"])
def leaderboard():
    """
    Global leaderboard of best scores seen across all episodes.
    Automatically updated on every /grader call.
    Shows top 20 scores with seed, steps, persona, and timestamp.
    Designed for benchmarking — judges can see score distribution.
    """
    if not _leaderboard:
        return {
            "top_scores": [],
            "total_episodes_graded": 0,
            "best_score": None,
            "avg_score": None,
            "message": "No episodes graded yet. Run /reset then /step then /grader.",
        }
    sorted_lb = sorted(_leaderboard, key=lambda x: x["score"], reverse=True)
    return {
        "top_scores": sorted_lb[:20],
        "total_episodes_graded": len(_leaderboard),
        "best_score": sorted_lb[0]["score"],
        "avg_score": round(sum(x["score"] for x in _leaderboard) / len(_leaderboard), 4),
        "score_distribution": {
            "elite_0.95+":   sum(1 for x in _leaderboard if x["score"] >= 0.95),
            "good_0.85+":    sum(1 for x in _leaderboard if 0.85 <= x["score"] < 0.95),
            "pass_0.65+":    sum(1 for x in _leaderboard if 0.65 <= x["score"] < 0.85),
            "fail_below_0.65": sum(1 for x in _leaderboard if x["score"] < 0.65),
        },
    }



@app.get("/dataset", tags=["Tools"])
def dataset_info():
    """
    Returns metadata about the real-world video dataset used for scene generation.
    Demonstrates real-world grounding of the environment.
    """
    from environment import _REAL_VIDEOS, _AUDIENCE_PERSONAS, _ENGAGEMENT_PATTERNS
    d = json.load(open(os.path.join(os.path.dirname(__file__), "video_dataset.json"))) if True else {}
    return {
        "total_videos": len(_REAL_VIDEOS),
        "real_research_grounded": d.get("real_video_count", 0),
        "platforms": list(set(v["platform"] for v in _REAL_VIDEOS)),
        "niches": list(set(v["niche"] for v in _REAL_VIDEOS)),
        "viral_count": sum(1 for v in _REAL_VIDEOS if v.get("viral")),
        "non_viral_count": sum(1 for v in _REAL_VIDEOS if not v.get("viral")),
        "engagement_patterns": _ENGAGEMENT_PATTERNS,
        "audience_personas": list(_AUDIENCE_PERSONAS.keys()),
        "data_sources": d.get("data_sources", []),
        "research_findings": d.get("research_findings", {}),
        "data_source": "10 videos grounded in published research (opus.pro, socialinsider, vidico, dmnews). 40 videos derived from real engagement patterns.",
        "usage": "Scenes are initialized from this dataset. Each seed maps to a specific video.",
    }


# ── /persona (NEW) ─────────────────────────────────────────────────────────────
@app.get("/persona", tags=["Tools"])
def persona():
    """
    Returns the current episode's audience persona and its scoring weights.
    Use finalize_edit to trigger persona-weighted final scoring.
    """
    try:
        state = _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from environment import _AUDIENCE_PERSONAS
    persona_name = state.metadata.get("audience_persona", "gen_z")
    persona_data = _AUDIENCE_PERSONAS.get(persona_name, {})

    return {
        "persona": persona_name,
        "description": persona_data.get("description", ""),
        "scoring_weights": {
            "hook_strength": persona_data.get("hook_weight", 0.25),
            "retention": persona_data.get("retention_weight", 0.25),
            "engagement": persona_data.get("engagement_weight", 0.25),
            "compliance": persona_data.get("compliance_weight", 0.25),
        },
        "max_attention_seconds": persona_data.get("max_attention_seconds", 15),
        "tip": f"Optimize for {persona_name} by prioritizing "
               + ("hook_strength and engagement" if persona_name == "gen_z"
                  else "retention and watch_time" if persona_name == "millennial"
                  else "platform_compliance and reach"),
        "finalize_edit": "Call finalize_edit action to lock your edit and receive persona-weighted bonus.",
    }



@app.get("/scenarios", tags=["Tools"])
def scenarios():
    """5 diverse scenario seeds across platforms and difficulty levels."""
    return {
        "scenarios": [
            {"id": "scenario_viral_hook",      "seed": 42,  "platform": "reels",
             "description": "Standard viral optimization. Strong hook, needs sequencing.",
             "difficulty": "medium", "target_score": 0.875},
            {"id": "scenario_retention_crisis", "seed": 7,   "platform": "reels",
             "description": "High filler content causing retention drop. Cut aggressively.",
             "difficulty": "hard",   "target_score": 0.78},
            {"id": "scenario_duration_overrun", "seed": 13,  "platform": "reels",
             "description": "Video exceeds 30s limit. Trim + quality fix required.",
             "difficulty": "medium", "target_score": 0.75},
            {"id": "scenario_weak_production",  "seed": 99,  "platform": "shorts",
             "description": "Good engagement but poor transitions, cuts, audio sync.",
             "difficulty": "medium", "target_score": 0.80},
            {"id": "scenario_tiktok_challenge", "seed": 21,  "platform": "tiktok",
             "description": "TikTok format. Hook-first ordering critical.",
             "difficulty": "hard",   "target_score": 0.85},
        ],
        "usage": "POST /reset?platform=<platform>&seed=<seed> to start a scenario.",
        "task_3_seeds": [42, 7, 13],
        "task_3_note": "task_3 is evaluated as average score across seeds 42, 7, 13.",
    }


# ── /trajectory ────────────────────────────────────────────────────────────────
@app.get("/trajectory", tags=["Tools"])
def trajectory():
    """Full action trajectory of current episode with per-step metrics."""
    if not _default_trajectory:
        return {"episode_steps": 0, "trajectory": [], "total_reward": 0.0,
                "message": "No steps taken yet."}
    total_reward = round(sum(t["reward"] for t in _default_trajectory), 4)
    valid_steps = sum(1 for t in _default_trajectory if t["valid"])
    return {
        "episode_steps": len(_default_trajectory),
        "valid_steps": valid_steps,
        "invalid_steps": len(_default_trajectory) - valid_steps,
        "total_reward": total_reward,
        "avg_reward_per_step": round(total_reward / len(_default_trajectory), 4),
        "trajectory": _default_trajectory,
        "action_sequence": _default_env.action_history,
        "order_score": _default_env.order_score(),
        "engagement_progression": [t["engagement"] for t in _default_trajectory],
        "retention_progression":  [t["retention"]  for t in _default_trajectory],
    }


# ── /efficiency ────────────────────────────────────────────────────────────────
@app.get("/efficiency", tags=["Tools"])
def efficiency():
    """Step efficiency rating: elite/optimal/good/acceptable."""
    try:
        state = _default_env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    score, breakdown, raw_score, rubric_score = _compute_score(obs, state.step_count)
    steps_used = state.step_count
    invalid_count = sum(1 for t in _default_trajectory if not t.get("valid", True))

    if steps_used == 0:       rating = "not_started"
    elif score >= 0.92 and steps_used <= 7:  rating = "elite"
    elif score >= 0.875 and steps_used <= 10: rating = "optimal"
    elif score >= 0.78 and steps_used <= 10:  rating = "good"
    elif score >= 0.65:       rating = "acceptable"
    else:                     rating = "needs_improvement"

    return {
        "score": score, "steps_used": steps_used,
        "steps_remaining": state.max_steps - steps_used,
        "valid_actions": steps_used - invalid_count,
        "invalid_actions": invalid_count,
        "efficiency_rating": rating,
        "efficiency_bonus": breakdown.get("efficiency_bonus", 0.0),
        "score_per_step": round(score / max(steps_used, 1), 4),
        "recommendation": (
            "Excellent efficiency!" if rating in ("elite", "optimal")
            else "Achieve same score in fewer steps for efficiency bonus."
        ),
    }


# ── /baseline ──────────────────────────────────────────────────────────────────
@app.get("/baseline", response_model=List[BaselineResult], tags=["Benchmark"])
def baseline():
    """Deterministic baseline agent across all three tasks."""
    results = []
    task_seeds = {
        "task_1": [42],
        "task_2": [42],
        "task_3": [42, 7, 13],  # FIX 1: task_3 evaluated across 3 seeds
    }

    for task_id in ["task_1", "task_2", "task_3"]:
        seeds = task_seeds[task_id]
        seed_scores = []

        for seed in seeds:
            b_env = VideoOptimizationEnv(platform="reels", seed=seed)
            state = b_env.reset()
            steps = 0

            def do(action_type: str, params: dict = None):
                nonlocal state, steps
                if state.done:
                    return
                state, _, _, _ = b_env.step(
                    Action(action_type=action_type, parameters=params or {})
                )
                steps += 1

            scenes = state.observation.scenes
            hooks = [s for s in scenes if s.scene_type in ("hook", "highlight") and s.has_hook]
            if hooks:
                best_hook = max(hooks, key=lambda s: s.hook_strength)
                if scenes[0].id != best_hook.id:
                    new_order = [best_hook.id] + [s.id for s in scenes if s.id != best_hook.id]
                    do("reorder_scenes", {"order": new_order})

            do("boost_hook")

            for scene in sorted(
                [s for s in state.observation.scenes
                 if s.engagement_score < 0.30 and s.scene_type in ("filler", "transition")],
                key=lambda s: s.engagement_score,
            ):
                if len(state.observation.scenes) <= 3:
                    break
                do("cut_scene", {"scene_id": scene.id})

            if state.observation.total_duration > PLATFORM_LIMITS["reels"]:
                do("trim_duration", {"target_seconds": PLATFORM_LIMITS["reels"]})

            do("enhance_pacing")
            if state.observation.avg_transition_quality < 0.72:
                do("improve_transition")
            if state.observation.avg_cut_smoothness < 0.72:
                do("smooth_cut")
            if state.observation.avg_audio_sync_score < 0.72:
                do("sync_audio")
            if not state.observation.subtitles_present:
                do("add_subtitles")
            do("add_music")

            seed_scores.append(_compute_score(state.observation, steps)[0])

        avg_score = round(sum(seed_scores) / len(seed_scores), 4)
        target = {"task_1": 0.65, "task_2": 0.78, "task_3": 0.90}[task_id]

        # Use seed=42 state for result details
        b_env2 = VideoOptimizationEnv(platform="reels", seed=42)
        state2 = b_env2.reset()
        steps2 = 0

        def do2(action_type: str, params: dict = None):
            nonlocal state2, steps2
            if state2.done:
                return
            state2, _, _, _ = b_env2.step(
                Action(action_type=action_type, parameters=params or {})
            )
            steps2 += 1

        scenes2 = state2.observation.scenes
        hooks2 = [s for s in scenes2 if s.scene_type in ("hook", "highlight") and s.has_hook]
        if hooks2:
            best2 = max(hooks2, key=lambda s: s.hook_strength)
            if scenes2[0].id != best2.id:
                do2("reorder_scenes", {"order": [best2.id] + [s.id for s in scenes2 if s.id != best2.id]})
        do2("boost_hook")
        for scene in sorted([s for s in state2.observation.scenes
                              if s.engagement_score < 0.30 and s.scene_type in ("filler", "transition")],
                             key=lambda s: s.engagement_score):
            if len(state2.observation.scenes) <= 3:
                break
            do2("cut_scene", {"scene_id": scene.id})
        if state2.observation.total_duration > PLATFORM_LIMITS["reels"]:
            do2("trim_duration", {"target_seconds": PLATFORM_LIMITS["reels"]})
        do2("enhance_pacing")
        if state2.observation.avg_transition_quality < 0.72:
            do2("improve_transition")
        if state2.observation.avg_cut_smoothness < 0.72:
            do2("smooth_cut")
        if state2.observation.avg_audio_sync_score < 0.72:
            do2("sync_audio")
        if not state2.observation.subtitles_present:
            do2("add_subtitles")
        do2("add_music")

        obs2 = state2.observation
        fb = generate_feedback(obs2, avg_score)

        results.append(BaselineResult(
            task_id=task_id,
            score=avg_score,
            passed=avg_score >= target,
            steps=steps2,
            final_engagement=obs2.current_engagement_score,
            final_retention=obs2.avg_retention,
            final_duration=obs2.total_duration,
            subtitles=obs2.subtitles_present,
            hook_first=obs2.hook_first,
            hook_strength=obs2.hook_strength,
            pacing_score=obs2.pacing_score,
            avg_transition_quality=obs2.avg_transition_quality,
            avg_cut_smoothness=obs2.avg_cut_smoothness,
            avg_audio_sync_score=obs2.avg_audio_sync_score,
            platform_compliant=obs2.platform_compliant,
            feedback=fb,
        ))

    return results


# ── /ws WebSocket ──────────────────────────────────────────────────────────────
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_env = VideoOptimizationEnv(platform="reels", seed=42)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type") or msg.get("action_type")

            if msg_type == "reset":
                ws_env.platform = msg.get("platform", "reels")
                ws_env.seed = int(msg.get("seed", 42))
                state = ws_env.reset()
                await websocket.send_json({"type": "reset", "state": state.model_dump()})

            elif msg_type == "step":
                action_str = msg.get("action") or msg.get("action_type") or msg.get("type")
                try:
                    action = Action(action_type=action_str, parameters=msg.get("parameters", {}))
                    state, reward, done, info = ws_env.step(action)
                    await websocket.send_json({"type": "step", "state": state.model_dump(),
                                               "reward": reward, "done": done, "info": info})
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            elif msg_type == "state":
                try:
                    await websocket.send_json({"type": "state", "state": ws_env.state.model_dump()})
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})

            elif msg_type == "grader":
                try:
                    state = ws_env.state
                    score, breakdown, raw_score, rubric_score = _compute_score(state.observation, state.step_count)
                    await websocket.send_json({"type": "grader", "score": score,
                                               "breakdown": breakdown, "passed": score >= 0.875})
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})
            else:
                await websocket.send_json({"error": f"Unknown type '{msg_type}'."})

    except WebSocketDisconnect:
        pass
