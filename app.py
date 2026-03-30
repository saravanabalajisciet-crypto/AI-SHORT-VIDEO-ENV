from fastapi import FastAPI, HTTPException, Query
from typing import List, Literal

from models import (
    Action, State, StepResponse, GraderResponse,
    TaskDefinition, BaselineResult, AIFeedback,
)
from environment import VideoOptimizationEnv, PLATFORM_LIMITS, generate_feedback

app = FastAPI(
    title="AI Short-Form Video Optimization Environment",
    description=(
        "OpenEnv-compatible RL environment for optimizing Reels / Shorts / TikTok. "
        "Features retention curves, hook strength, pacing, transition quality, "
        "cut smoothness, audio sync, and AI feedback."
    ),
    version="4.0.0",
)

env = VideoOptimizationEnv(platform="reels", seed=42)


# ── /reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset", response_model=State, tags=["Environment"])
def reset(
    platform: Literal["reels", "shorts", "tiktok"] = Query("reels"),
    seed: int = Query(42, description="RNG seed for reproducibility"),
):
    """Start a new episode. Returns the initial state."""
    env.platform = platform
    env.seed = seed
    return env.reset()


# ── /step ──────────────────────────────────────────────────────────────────────
@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(action: Action):
    """
    Apply one action. Returns (state, reward, done, info).

    Actions:
    - cut_scene:           {"scene_id": str}
    - reorder_scenes:      {"order": [str, ...]}
    - add_subtitles:       {}
    - add_music:           {"genre": str}
    - boost_hook:          {}
    - trim_duration:       {"target_seconds": float}
    - enhance_pacing:      {}
    - improve_transition:  {}
    - smooth_cut:          {}
    - sync_audio:          {}
    """
    try:
        state, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(state=state, reward=reward, done=done, info=info)


# ── /state ─────────────────────────────────────────────────────────────────────
@app.get("/state", response_model=State, tags=["Environment"])
def get_state():
    """Read the current state without advancing the episode."""
    try:
        return env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── /tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks", response_model=List[TaskDefinition], tags=["Benchmark"])
def get_tasks():
    """Return the three benchmark task definitions."""
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
                "Agent must execute in order: reorder_scenes → boost_hook → cut_scene(s) → "
                "enhance_pacing → improve_transition → smooth_cut → sync_audio → "
                "add_subtitles → add_music. Order matters for maximum reward."
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
    """
    try:
        state = env.state
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    obs = state.observation
    score, breakdown = _compute_score(obs)
    return GraderResponse(score=score, breakdown=breakdown, passed=score >= 0.875)


def _compute_score(obs) -> tuple:
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
    score = round(min(weighted, 1.0), 4)
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
    score, _ = _compute_score(obs)
    return generate_feedback(obs, score)


# ── /baseline ──────────────────────────────────────────────────────────────────
@app.get("/baseline", response_model=List[BaselineResult], tags=["Benchmark"])
def baseline():
    """
    Deterministic baseline agent across all three tasks (seed=42).

    Strategy:
      1.  reorder_scenes    → hook first
      2.  boost_hook
      3.  cut_scene(s)      → remove filler/transition < 0.30
      4.  trim_duration     → if over platform limit
      5.  enhance_pacing
      6.  improve_transition → if avg_transition_quality < 0.70
      7.  smooth_cut         → if avg_cut_smoothness < 0.70
      8.  sync_audio         → if avg_audio_sync_score < 0.70
      9.  add_subtitles
      10. add_music
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

        # 1. Reorder: best hook scene first
        scenes = state.observation.scenes
        hook_candidates = [s for s in scenes if s.scene_type in ("hook", "highlight") and s.has_hook]
        if hook_candidates:
            best_hook = max(hook_candidates, key=lambda s: s.hook_strength)
            if scenes[0].id != best_hook.id:
                new_order = [best_hook.id] + [s.id for s in scenes if s.id != best_hook.id]
                do("reorder_scenes", {"order": new_order})

        # 2. Boost hook
        do("boost_hook")

        # 3. Cut dead-weight scenes (filler/transition < 0.30), keep >= 3 scenes
        for scene in sorted(
            [s for s in state.observation.scenes
             if s.engagement_score < 0.30 and s.scene_type in ("filler", "transition")],
            key=lambda s: s.engagement_score,
        ):
            if len(state.observation.scenes) <= 3:
                break
            do("cut_scene", {"scene_id": scene.id})

        # 4. Trim duration if needed
        limit = PLATFORM_LIMITS["reels"]
        if state.observation.total_duration > limit:
            do("trim_duration", {"target_seconds": limit})

        # 5. Enhance pacing
        do("enhance_pacing")

        # 6. Improve transition quality if below threshold
        if state.observation.avg_transition_quality < 0.70:
            do("improve_transition")

        # 7. Smooth cuts if below threshold
        if state.observation.avg_cut_smoothness < 0.70:
            do("smooth_cut")

        # 8. Sync audio if below threshold
        if state.observation.avg_audio_sync_score < 0.70:
            do("sync_audio")

        # 9. Add subtitles
        if not state.observation.subtitles_present:
            do("add_subtitles")

        # 10. Add music
        do("add_music")

        obs = state.observation
        score, _ = _compute_score(obs)
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
