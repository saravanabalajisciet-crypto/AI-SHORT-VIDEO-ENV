from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel
import time
import random

from models import (
    Action, State, StepResponse, GraderResponse,
    TaskDefinition, BaselineResult, AIFeedback, ActionType,
)
from environment import VideoOptimizationEnv, PLATFORM_LIMITS, generate_feedback
from pydantic import ValidationError

app = FastAPI(
    title="AI Short-Form Video Optimization Environment",
    description=(
        "OpenEnv-compatible RL environment for optimizing Reels / Shorts / TikTok. "
        "Features retention curves, hook strength, pacing, transition quality, "
        "cut smoothness, audio sync, AI feedback, scenario-based tasks, "
        "action budgets, step efficiency scoring, and trajectory logging."
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

env = VideoOptimizationEnv(platform="reels", seed=42)

# ── Episode trajectory log (ADD-ONLY) ─────────────────────────────────────────
_trajectory: List[Dict[str, Any]] = []
_episode_start_time: float = 0.0
_action_budget: int = 10  # max efficient actions before penalty kicks in


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0]["msg"] if errors else "Invalid request payload."
    return JSONResponse(status_code=400, content={"detail": msg})


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "AI Short-Form Video Optimization Environment",
        "version": "5.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/grader", "/baseline",
            "/feedback", "/hint", "/scenarios", "/trajectory", "/efficiency",
        ],
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ── /reset ─────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    platform: str = "reels"
    seed: int = 42


@app.post("/reset", response_model=State, tags=["Environment"])
async def reset(
    request: Request,
    platform: Literal["reels", "shorts", "tiktok"] = Query("reels"),
    seed: int = Query(42, description="RNG seed for reproducibility"),
):
    """
    Start a new episode. Returns the initial state.

    Accepts query params OR JSON body:
    - /reset?platform=reels&seed=42
    - Body: {"platform": "reels", "seed": 42}
    """
    global _trajectory, _episode_start_time
    try:
        body = await request.json()
        if isinstance(body, dict):
            platform = body.get("platform", platform)
            seed = int(body.get("seed", seed))
    except Exception:
        pass

    if platform not in ("reels", "shorts", "tiktok"):
        raise HTTPException(status_code=400, detail=f"Invalid platform '{platform}'. Choose: reels, shorts, tiktok")

    env.platform = platform
    env.seed = seed
    state = env.reset()

    # Reset trajectory
    _trajectory = []
    _episode_start_time = time.time()

    return state


# ── /step ──────────────────────────────────────────────────────────────────────
@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: Request):
    """
    Apply one action. Returns (state, reward, done, info).

    Accepts any of these JSON formats:
    - {"action_type": "boost_hook"}
    - {"action": "boost_hook"}
    - {"type": "boost_hook"}

    With optional parameters:
    - {"action_type": "cut_scene", "parameters": {"scene_id": "scene_1"}}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    print(f"[/step] received: {body}")

    try:
        action = Action(**body)
    except ValidationError as e:
        errors = e.errors()
        msg = errors[0]["msg"] if errors else "Invalid action payload."
        raise HTTPException(status_code=400, detail=msg)

    print(f"[/step] resolved action_type: {action.action_type}")

    try:
        state, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log to trajectory (ADD-ONLY)
    _trajectory.append({
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
        return env.state
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
                "total_duration <= 30 s, subtitles present, avg_retention >= 0.55. "
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
                "Full viral optimization: engagement >= 0.80, avg_retention >= 0.90, "
                "hook_strength >= 0.70, avg_transition_quality >= 0.70, "
                "platform compliance, subtitles, and hook-first ordering. "
                "Requires all 10 actions in the correct sequence."
            ),
            expected_behavior=(
                "Agent must execute in order: reorder_scenes -> boost_hook -> cut_scene(s) -> "
                "enhance_pacing -> improve_transition -> smooth_cut -> sync_audio -> "
                "add_subtitles -> add_music. Order matters for maximum reward."
            ),
            evaluation_criteria=(
                "current_engagement_score >= 0.80 AND avg_retention >= 0.90 AND "
                "hook_strength >= 0.70 AND avg_transition_quality >= 0.70 AND "
                "platform_compliant == True AND subtitles_present == True. "
                "Grader score >= 0.875."
            ),
            target_score=0.875,
        ),
    ]


# ── /grader ────────────────────────────────────────────────────────────────────
@app.get("/grader", response_model=GraderResponse, tags=["Benchmark"])
def grader():
    """
    Deterministic multi-factor scoring (weights sum to 1.0):

    score = engagement           * 0.35
          + retention            * 0.15
          + platform_compliance  * 0.20
          + subtitles            * 0.10
          + hook_strength        * 0.10
          + pacing               * 0.05
          + avg_transition_quality * 0.03
          + avg_cut_smoothness     * 0.01
          + avg_audio_sync_score   * 0.01

    Extended: efficiency bonus/penalty based on steps used.
    """
    try:
        state = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    score, breakdown = _compute_score(obs, state.step_count)
    return GraderResponse(score=score, breakdown=breakdown, passed=score >= 0.875)


def _compute_score(obs, step_count: int = 0) -> tuple:
    eng        = obs.current_engagement_score
    retention  = obs.avg_retention
    compliance = 1.0 if obs.platform_compliant else 0.0
    subtitles  = 1.0 if obs.subtitles_present  else 0.0
    hook       = obs.hook_strength
    pacing     = obs.pacing_score
    tq         = obs.avg_transition_quality
    cs         = obs.avg_cut_smoothness
    asy        = obs.avg_audio_sync_score

    weighted = (
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

    # ADD-ONLY: step efficiency bonus (reward fewer steps for same quality)
    efficiency_bonus = 0.0
    if step_count > 0:
        if step_count <= 7:
            efficiency_bonus = 0.02   # solved efficiently
        elif step_count <= 10:
            efficiency_bonus = 0.01
        elif step_count > 13:
            efficiency_bonus = -0.01  # over-edited

    score = round(min(weighted + efficiency_bonus, 1.0), 4)
    breakdown = {
        "engagement":            round(eng        * 0.35, 4),
        "retention":             round(retention  * 0.15, 4),
        "platform_compliance":   round(compliance * 0.20, 4),
        "subtitles":             round(subtitles  * 0.10, 4),
        "hook_strength":         round(hook       * 0.10, 4),
        "pacing":                round(pacing     * 0.05, 4),
        "transition_quality":    round(tq         * 0.03, 4),
        "cut_smoothness":        round(cs         * 0.01, 4),
        "audio_sync":            round(asy        * 0.01, 4),
        "efficiency_bonus":      round(efficiency_bonus, 4),
        "final_score":           score,
    }
    return score, breakdown


# ── /feedback ──────────────────────────────────────────────────────────────────
@app.get("/feedback", response_model=AIFeedback, tags=["WOW"])
def feedback():
    """AI coaching — actionable tips based on the current state."""
    try:
        state = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    obs = state.observation
    score, _ = _compute_score(obs, state.step_count)
    return generate_feedback(obs, score)


# ── /hint (NEW tool-style endpoint) ───────────────────────────────────────────
@app.get("/hint", tags=["Tools"])
def hint():
    """
    Tool-style helper: returns the single best next action to take
    based on current observation gaps. Helps LLM agents reason about
    what to do next without guessing.
    """
    try:
        state = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    suggestions = []

    if not obs.hook_first or obs.hook_strength < 0.5:
        suggestions.append({
            "action": "reorder_scenes",
            "reason": "No hook-first scene. Reorder to put highest hook_strength scene first.",
            "priority": 10,
        })
    if obs.hook_strength < 0.7:
        suggestions.append({
            "action": "boost_hook",
            "reason": f"hook_strength={obs.hook_strength:.2f} < 0.70. Boost it for +0.30 lift.",
            "priority": 9,
        })
    if not obs.platform_compliant:
        suggestions.append({
            "action": "trim_duration",
            "reason": f"Duration {obs.total_duration:.1f}s exceeds platform limit. Trim to 30s.",
            "priority": 8,
        })
    if obs.avg_retention < 0.6:
        suggestions.append({
            "action": "enhance_pacing",
            "reason": f"avg_retention={obs.avg_retention:.2f} < 0.60. Pacing fix helps retention.",
            "priority": 7,
        })
    if obs.avg_transition_quality < 0.7:
        suggestions.append({
            "action": "improve_transition",
            "reason": f"avg_transition_quality={obs.avg_transition_quality:.2f} < 0.70.",
            "priority": 6,
        })
    if obs.avg_cut_smoothness < 0.7:
        suggestions.append({
            "action": "smooth_cut",
            "reason": f"avg_cut_smoothness={obs.avg_cut_smoothness:.2f} < 0.70.",
            "priority": 5,
        })
    if obs.avg_audio_sync_score < 0.7:
        suggestions.append({
            "action": "sync_audio",
            "reason": f"avg_audio_sync_score={obs.avg_audio_sync_score:.2f} < 0.70.",
            "priority": 4,
        })
    if not obs.subtitles_present:
        suggestions.append({
            "action": "add_subtitles",
            "reason": "Subtitles not present. Adds +0.10 to grader score.",
            "priority": 3,
        })
    if not obs.music_added:
        suggestions.append({
            "action": "add_music",
            "reason": "Music not added. Boosts hook/content engagement by +0.12.",
            "priority": 2,
        })

    # Check for low-engagement scenes to cut
    low_scenes = [
        s for s in obs.scenes
        if s.engagement_score < 0.30 and s.scene_type in ("filler", "transition")
    ]
    if low_scenes and len(obs.scenes) > 3:
        worst = min(low_scenes, key=lambda s: s.engagement_score)
        suggestions.append({
            "action": "cut_scene",
            "parameters": {"scene_id": worst.id},
            "reason": f"Scene {worst.id} has engagement={worst.engagement_score:.2f}. Cut it.",
            "priority": 8,
        })

    if not suggestions:
        return {
            "best_action": None,
            "reason": "Environment looks well-optimized. Call /grader to check score.",
            "all_suggestions": [],
            "steps_used": state.step_count,
            "steps_remaining": state.max_steps - state.step_count,
        }

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


# ── /scenarios (NEW) ──────────────────────────────────────────────────────────
@app.get("/scenarios", tags=["Tools"])
def scenarios():
    """
    Returns a set of diverse scenario seeds with descriptions.
    Agents can use these seeds with /reset to test across varied starting conditions.
    Adds scenario diversity beyond the default seed=42.
    """
    return {
        "scenarios": [
            {
                "id": "scenario_viral_hook",
                "seed": 42,
                "platform": "reels",
                "description": "Standard viral optimization. Strong hook available, needs sequencing.",
                "difficulty": "medium",
                "target_score": 0.875,
            },
            {
                "id": "scenario_retention_crisis",
                "seed": 7,
                "platform": "reels",
                "description": "High filler content causing retention drop. Agent must cut aggressively.",
                "difficulty": "hard",
                "target_score": 0.78,
            },
            {
                "id": "scenario_duration_overrun",
                "seed": 13,
                "platform": "reels",
                "description": "Video exceeds 30s limit. Trim + quality fix required.",
                "difficulty": "medium",
                "target_score": 0.75,
            },
            {
                "id": "scenario_weak_production",
                "seed": 99,
                "platform": "shorts",
                "description": "Good engagement but poor transitions, cuts, and audio sync.",
                "difficulty": "medium",
                "target_score": 0.80,
            },
            {
                "id": "scenario_tiktok_challenge",
                "seed": 21,
                "platform": "tiktok",
                "description": "TikTok format. Hook-first ordering critical for algorithm boost.",
                "difficulty": "hard",
                "target_score": 0.85,
            },
        ],
        "usage": "POST /reset?platform=reels&seed=<seed> to start a scenario episode.",
    }


# ── /trajectory (NEW) ─────────────────────────────────────────────────────────
@app.get("/trajectory", tags=["Tools"])
def trajectory():
    """
    Returns the full action trajectory of the current episode.
    Useful for debugging agent reasoning and analyzing decision quality.
    """
    if not _trajectory:
        return {
            "episode_steps": 0,
            "trajectory": [],
            "total_reward": 0.0,
            "message": "No steps taken yet. Call /reset then /step.",
        }

    total_reward = round(sum(t["reward"] for t in _trajectory), 4)
    valid_steps = sum(1 for t in _trajectory if t["valid"])
    invalid_steps = len(_trajectory) - valid_steps

    return {
        "episode_steps": len(_trajectory),
        "valid_steps": valid_steps,
        "invalid_steps": invalid_steps,
        "total_reward": total_reward,
        "avg_reward_per_step": round(total_reward / len(_trajectory), 4),
        "trajectory": _trajectory,
        "engagement_progression": [t["engagement"] for t in _trajectory],
        "retention_progression": [t["retention"] for t in _trajectory],
    }


# ── /efficiency (NEW) ─────────────────────────────────────────────────────────
@app.get("/efficiency", tags=["Tools"])
def efficiency():
    """
    Returns step efficiency analysis for the current episode.
    Rewards agents that achieve high scores in fewer steps.
    Penalizes wasted/invalid actions.
    """
    try:
        state = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    score, breakdown = _compute_score(obs, state.step_count)

    steps_used = state.step_count
    max_steps = state.max_steps
    invalid_count = sum(1 for t in _trajectory if not t.get("valid", True))
    valid_count = steps_used - invalid_count

    # Efficiency rating
    if steps_used == 0:
        rating = "not_started"
    elif score >= 0.875 and steps_used <= 7:
        rating = "elite"
    elif score >= 0.875 and steps_used <= 10:
        rating = "optimal"
    elif score >= 0.78 and steps_used <= 10:
        rating = "good"
    elif score >= 0.65:
        rating = "acceptable"
    else:
        rating = "needs_improvement"

    return {
        "score": score,
        "steps_used": steps_used,
        "steps_remaining": max_steps - steps_used,
        "valid_actions": valid_count,
        "invalid_actions": invalid_count,
        "efficiency_rating": rating,
        "efficiency_bonus": breakdown.get("efficiency_bonus", 0.0),
        "score_per_step": round(score / max(steps_used, 1), 4),
        "recommendation": (
            "Excellent efficiency!" if rating in ("elite", "optimal")
            else "Try to achieve the same score in fewer steps."
        ),
    }


# ── /baseline ──────────────────────────────────────────────────────────────────
@app.get("/baseline", response_model=List[BaselineResult], tags=["Benchmark"])
def baseline():
    """
    Deterministic baseline agent across all three tasks (seed=42).
    """
    results = []

    for task_id in ["task_1", "task_2", "task_3"]:
        b_env = VideoOptimizationEnv(platform="reels", seed=42)
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
        hook_candidates = [s for s in scenes if s.scene_type in ("hook", "highlight") and s.has_hook]
        if hook_candidates:
            best_hook = max(hook_candidates, key=lambda s: s.hook_strength)
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

        limit = PLATFORM_LIMITS["reels"]
        if state.observation.total_duration > limit:
            do("trim_duration", {"target_seconds": limit})

        do("enhance_pacing")

        if state.observation.avg_transition_quality < 0.70:
            do("improve_transition")
        if state.observation.avg_cut_smoothness < 0.70:
            do("smooth_cut")
        if state.observation.avg_audio_sync_score < 0.70:
            do("sync_audio")
        if not state.observation.subtitles_present:
            do("add_subtitles")

        do("add_music")

        obs = state.observation
        score, _ = _compute_score(obs, steps)
        fb = generate_feedback(obs, score)

        results.append(BaselineResult(
            task_id=task_id,
            score=score,
            passed=score >= {"task_1": 0.65, "task_2": 0.78, "task_3": 0.875}[task_id],
            steps=steps,
            final_engagement=obs.current_engagement_score,
            final_retention=obs.avg_retention,
            final_duration=obs.total_duration,
            subtitles=obs.subtitles_present,
            hook_first=obs.hook_first,
            hook_strength=obs.hook_strength,
            pacing_score=obs.pacing_score,
            avg_transition_quality=obs.avg_transition_quality,
            avg_cut_smoothness=obs.avg_cut_smoothness,
            avg_audio_sync_score=obs.avg_audio_sync_score,
            platform_compliant=obs.platform_compliant,
            feedback=fb,
        ))

    return results


# ── /ws WebSocket ──────────────────────────────────────────────────────────────
from fastapi import WebSocket, WebSocketDisconnect
import json


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent multi-mode sessions.
    """
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
                platform = msg.get("platform", "reels")
                seed = int(msg.get("seed", 42))
                ws_env.platform = platform
                ws_env.seed = seed
                state = ws_env.reset()
                await websocket.send_json({"type": "reset", "state": state.model_dump()})

            elif msg_type == "step":
                action_str = msg.get("action") or msg.get("action_type") or msg.get("type")
                parameters = msg.get("parameters", {})
                try:
                    action = Action(action_type=action_str, parameters=parameters)
                    state, reward, done, info = ws_env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "state": state.model_dump(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    })
                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            elif msg_type == "state":
                try:
                    state = ws_env.state
                    await websocket.send_json({"type": "state", "state": state.model_dump()})
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})

            elif msg_type == "grader":
                try:
                    state = ws_env.state
                    score, breakdown = _compute_score(state.observation, state.step_count)
                    await websocket.send_json({
                        "type": "grader",
                        "score": score,
                        "breakdown": breakdown,
                        "passed": score >= 0.875,
                    })
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})

            else:
                await websocket.send_json({
                    "error": f"Unknown type '{msg_type}'. Use: reset, step, state, grader"
                })

    except WebSocketDisconnect:
        pass
