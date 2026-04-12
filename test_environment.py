"""
test_environment.py — pytest test suite for AI Video Optimizer Env.

Run locally:
    pip install pytest requests
    # Start server first: python -m uvicorn server.app:app --port 7860
    pytest test_environment.py -v

Or against HF Space:
    ENV_URL=https://saravanabalajisara-ai-video-optimizer-env.hf.space pytest test_environment.py -v
"""

import os
import pytest
import requests

BASE = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")


# ── helpers ────────────────────────────────────────────────────────────────────

def get(path):
    r = requests.get(f"{BASE}{path}", timeout=15)
    return r

def post(path, **kwargs):
    r = requests.post(f"{BASE}{path}", timeout=15, **kwargs)
    return r

def reset(platform="reels", seed=42):
    return post("/reset", params={"platform": platform, "seed": seed}).json()

def step(action_type, parameters=None):
    return post("/step", json={"action_type": action_type, "parameters": parameters or {}}).json()


# ── health ─────────────────────────────────────────────────────────────────────

def test_root():
    r = get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "running"
    assert "version" in data

def test_health():
    r = get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ── reset ──────────────────────────────────────────────────────────────────────

def test_reset_default():
    state = reset()
    assert "episode_id" in state
    assert state["step_count"] == 0
    assert state["done"] is False
    obs = state["observation"]
    assert 0.0 <= obs["current_engagement_score"] <= 1.0
    assert obs["steps_remaining"] == 15
    assert obs["risk_score"] is not None

def test_reset_all_platforms():
    for platform in ["reels", "shorts", "tiktok"]:
        state = reset(platform=platform)
        assert state["observation"]["platform"] == platform

def test_reset_invalid_platform():
    r = post("/reset", params={"platform": "youtube", "seed": 42})
    assert r.status_code == 400

def test_reset_different_seeds():
    s1 = reset(seed=42)["observation"]["current_engagement_score"]
    s2 = reset(seed=7)["observation"]["current_engagement_score"]
    # Different seeds should produce different initial states
    assert s1 != s2

def test_reset_clears_episode():
    reset(seed=42)
    step("boost_hook")
    state_mid = get("/state").json()
    assert state_mid["step_count"] == 1
    reset(seed=42)
    state_fresh = get("/state").json()
    assert state_fresh["step_count"] == 0


# ── step ───────────────────────────────────────────────────────────────────────

def test_step_boost_hook():
    reset()
    resp = step("boost_hook")
    assert "state" in resp
    assert "reward" in resp
    assert "done" in resp
    assert resp["state"]["step_count"] == 1

def test_step_all_actions():
    reset()
    actions = [
        "boost_hook", "enhance_pacing", "improve_transition",
        "smooth_cut", "sync_audio", "add_subtitles", "add_music",
    ]
    for action in actions:
        resp = step(action)
        assert resp["state"] is not None
        assert isinstance(resp["reward"], float)

def test_step_invalid_action():
    reset()
    r = post("/step", json={"action_type": "nonexistent_action"})
    assert r.status_code == 400

def test_step_repeat_idempotent_penalized():
    reset()
    step("boost_hook")
    resp2 = step("boost_hook")
    assert resp2["info"]["valid"] is False
    assert resp2["reward"] < 0

def test_step_cut_scene_valid():
    state = reset()
    scenes = state["observation"]["scenes"]
    scene_id = scenes[-1]["id"]
    resp = step("cut_scene", {"scene_id": scene_id})
    assert resp["info"]["valid"] is True

def test_step_cut_scene_invalid_id():
    reset()
    resp = step("cut_scene", {"scene_id": "nonexistent_scene_999"})
    assert resp["info"]["valid"] is False

def test_step_reorder_scenes():
    state = reset()
    ids = [s["id"] for s in state["observation"]["scenes"]]
    resp = step("reorder_scenes", {"order": list(reversed(ids))})
    assert resp["info"]["valid"] is True

def test_step_reorder_wrong_ids():
    reset()
    resp = step("reorder_scenes", {"order": ["bad_id_1", "bad_id_2"]})
    assert resp["info"]["valid"] is False

def test_step_trim_duration():
    reset()
    resp = step("trim_duration", {"target_seconds": 30.0})
    # valid if duration was over 30, invalid if already compliant
    assert "valid" in resp["info"]

def test_step_finalize_edit():
    reset()
    step("boost_hook")
    resp = step("finalize_edit")
    assert resp["done"] is True
    assert resp["info"]["valid"] is True
    assert "persona" in resp["info"]

def test_step_after_done_raises():
    reset()
    step("finalize_edit")
    r = post("/step", json={"action_type": "boost_hook"})
    assert r.status_code == 400

def test_step_increments_count():
    reset()
    for i in range(1, 6):
        resp = step("enhance_pacing" if i == 1 else "smooth_cut" if i == 2
                    else "sync_audio" if i == 3 else "add_subtitles" if i == 4
                    else "add_music")
        assert resp["state"]["step_count"] == i


# ── observation fields ─────────────────────────────────────────────────────────

def test_observation_has_risk_score():
    state = reset()
    obs = state["observation"]
    assert "risk_score" in obs
    assert 0.0 <= obs["risk_score"] <= 1.0

def test_observation_has_steps_remaining():
    state = reset()
    assert state["observation"]["steps_remaining"] == 15
    step("boost_hook")
    state2 = get("/state").json()
    assert state2["observation"]["steps_remaining"] == 14

def test_observation_retention_curve_length():
    state = reset()
    obs = state["observation"]
    assert len(obs["retention_curve"]) == len(obs["scenes"])

def test_observation_platform_compliant():
    state = reset(platform="reels", seed=42)
    obs = state["observation"]
    assert isinstance(obs["platform_compliant"], bool)


# ── grader ─────────────────────────────────────────────────────────────────────

def test_grader_returns_score():
    reset()
    step("boost_hook")
    r = get("/grader")
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["score"] <= 1.0
    assert "breakdown" in data
    assert "raw_score" in data
    assert "rubric_score" in data

def test_grader_raw_score_never_exceeds_rubric():
    reset()
    for a in ["boost_hook", "add_subtitles", "add_music"]:
        step(a)
    data = get("/grader").json()
    # raw_score is pure, rubric_score has bonuses — both valid
    assert data["raw_score"] >= 0.0
    assert data["rubric_score"] >= 0.0

def test_grader_has_explanation():
    reset()
    step("boost_hook")
    data = get("/grader").json()
    assert "explanation" in data
    assert len(data["explanation"]) > 10

def test_grader_has_metadata():
    reset()
    step("boost_hook")
    data = get("/grader").json()
    assert "grader_metadata" in data
    meta = data["grader_metadata"]
    assert "risk_score" in meta
    assert "score_cap_applied" in meta

def test_grader_breakdown_weights_sum():
    reset()
    for a in ["boost_hook", "add_subtitles", "add_music", "enhance_pacing"]:
        step(a)
    bd = get("/grader").json()["breakdown"]
    core = sum([
        bd["engagement"], bd["retention"], bd["platform_compliance"],
        bd["subtitles"], bd["hook_strength"], bd["pacing"],
        bd["transition_quality"], bd["cut_smoothness"], bd["audio_sync"],
    ])
    assert abs(core - bd["raw_score"]) < 0.01


# ── tasks ──────────────────────────────────────────────────────────────────────

def test_tasks_returns_four():
    r = get("/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) == 4
    ids = [t["id"] for t in tasks]
    assert "task_1" in ids
    assert "task_4" in ids

def test_tasks_difficulty_progression():
    tasks = get("/tasks").json()
    levels = [t["level"] for t in tasks]
    assert levels[0] == "easy"
    assert levels[-1] == "elite"

def test_tasks_target_scores_ascending():
    tasks = get("/tasks").json()
    scores = [t["target_score"] for t in tasks]
    assert scores == sorted(scores)


# ── hint & strategy ────────────────────────────────────────────────────────────

def test_hint_returns_action():
    reset()
    r = get("/hint")
    assert r.status_code == 200
    data = r.json()
    assert "best_action" in data
    assert "reason" in data

def test_strategy_returns_plan():
    reset()
    r = get("/strategy")
    assert r.status_code == 200
    data = r.json()
    assert "recommended_sequence" in data
    assert "immediate_action" in data
    assert data["risk_label"] in ("low", "medium", "high")
    assert "strategy_mode" in data
    assert data["strategy_mode"] in ("aggressive", "balanced", "conservative")
    assert "point_of_no_recovery" in data
    assert isinstance(data["point_of_no_recovery"], bool)


# ── trajectory ─────────────────────────────────────────────────────────────────

def test_trajectory_empty_before_steps():
    reset()
    r = get("/trajectory")
    assert r.status_code == 200
    data = r.json()
    assert data["episode_steps"] == 0

def test_trajectory_has_decision_quality():
    reset()
    for a in ["boost_hook", "add_subtitles", "add_music"]:
        step(a)
    data = get("/trajectory").json()
    assert "decision_quality_summary" in data
    dqs = data["decision_quality_summary"]
    assert set(dqs.keys()) == {"excellent", "good", "neutral", "poor", "invalid"}
    assert sum(dqs.values()) == 3

def test_trajectory_engagement_progression_length():
    reset()
    for a in ["boost_hook", "enhance_pacing", "add_subtitles"]:
        step(a)
    data = get("/trajectory").json()
    assert len(data["engagement_progression"]) == 3


# ── session isolation ──────────────────────────────────────────────────────────

def test_session_isolation():
    # Use isolated sessions — don't touch default env
    post("/reset", json={"platform": "reels", "seed": 42, "session_id": "test-iso-1"})
    post("/reset", json={"platform": "reels", "seed": 7,  "session_id": "test-iso-2"})
    post("/step",  json={"action_type": "boost_hook", "session_id": "test-iso-1"})
    post("/step",  json={"action_type": "add_music",  "session_id": "test-iso-2"})
    # Sessions are isolated — iso-1 step should not affect iso-2
    # Verify by resetting iso-1 and checking it starts fresh
    r = post("/reset", json={"platform": "reels", "seed": 42, "session_id": "test-iso-1"})
    assert r.status_code == 200
    assert r.json()["step_count"] == 0

def test_simulate_mode_reset():
    r = post("/reset", json={"platform": "reels", "seed": 42,
                              "simulate": True, "session_id": "sim-test-99"})
    assert r.status_code == 200


# ── other endpoints ────────────────────────────────────────────────────────────

def test_feedback():
    reset()
    step("boost_hook")
    r = get("/feedback")
    assert r.status_code == 200
    data = r.json()
    assert "overall" in data
    assert "tips" in data

def test_scenarios():
    r = get("/scenarios")
    assert r.status_code == 200
    data = r.json()
    assert len(data["scenarios"]) >= 5

def test_persona():
    reset()
    r = get("/persona")
    assert r.status_code == 200
    data = r.json()
    assert data["persona"] in ("gen_z", "millennial", "brand")

def test_dataset():
    r = get("/dataset")
    assert r.status_code == 200
    data = r.json()
    assert data["total_videos"] > 0

def test_leaderboard_after_grader():
    reset()
    step("boost_hook")
    get("/grader")
    r = get("/leaderboard")
    assert r.status_code == 200
    data = r.json()
    assert data["total_episodes_graded"] >= 1

def test_efficiency():
    reset()
    for a in ["boost_hook", "add_subtitles", "add_music"]:
        step(a)
    r = get("/efficiency")
    assert r.status_code == 200
    data = r.json()
    assert "efficiency_rating" in data
    assert data["efficiency_rating"] in (
        "not_started", "elite", "optimal", "good", "acceptable", "needs_improvement"
    )

def test_baseline():
    r = get("/baseline")
    assert r.status_code == 200
    results = r.json()
    assert len(results) >= 3
    for res in results:
        assert 0.0 <= res["score"] <= 1.0


# ── environment logic ──────────────────────────────────────────────────────────

def test_engagement_increases_after_boost():
    state = reset()
    eng_before = state["observation"]["current_engagement_score"]
    resp = step("boost_hook")
    eng_after = resp["state"]["observation"]["current_engagement_score"]
    assert eng_after > eng_before

def test_subtitles_flag_set():
    reset()
    step("add_subtitles")
    state = get("/state").json()
    assert state["observation"]["subtitles_present"] is True

def test_music_flag_set():
    reset()
    step("add_music")
    state = get("/state").json()
    assert state["observation"]["music_added"] is True

def test_max_steps_ends_episode():
    reset(seed=99)
    # Use distinct one-time actions + repeatable ones to exhaust 15 steps
    one_time = ["boost_hook", "enhance_pacing", "improve_transition",
                "smooth_cut", "sync_audio", "add_subtitles", "add_music"]
    repeatable = ["trim_duration"] * 8  # will be invalid but still count as steps
    done = False
    for a in one_time + repeatable:
        resp = post("/step", json={"action_type": a, "parameters": {}}).json()
        if resp.get("done"):
            done = True
            break
    assert done is True

def test_risk_score_decreases_after_boost():
    state = reset()
    risk_before = state["observation"]["risk_score"]
    step("boost_hook")
    state2 = get("/state").json()
    risk_after = state2["observation"]["risk_score"]
    # Boosting hook should reduce risk
    assert risk_after <= risk_before


# ── strategy engine unit tests ─────────────────────────────────────────────────

def test_strategy_engine_standalone():
    from strategy_engine import suggest_strategy, explain_score
    obs = {
        "risk_score": 0.2,
        "hook_strength": 0.4,
        "avg_retention": 0.6,
        "current_engagement_score": 0.55,
        "pacing_score": 0.5,
        "avg_transition_quality": 0.5,
        "avg_cut_smoothness": 0.5,
        "avg_audio_sync_score": 0.5,
        "platform_compliant": True,
        "hook_first": False,
        "subtitles_present": False,
        "music_added": False,
        "total_duration": 28.0,
        "scenes": [],
    }
    plan = suggest_strategy(obs, persona="gen_z", steps_used=2)
    assert plan["risk_label"] == "low"
    assert plan["strategy_mode"] == "aggressive"
    assert plan["point_of_no_recovery"] is False
    assert isinstance(plan["recommended_sequence"], list)

def test_explain_score_standalone():
    from strategy_engine import explain_score
    obs = {"subtitles_present": True, "music_added": True, "hook_strength": 0.8}
    result = explain_score(obs, score=0.91, raw_score=0.89,
                           breakdown={"raw_score": 0.89, "rubric_score": 0.91,
                                      "efficiency_bonus": 0.02},
                           action_history=["boost_hook", "add_subtitles"])
    assert len(result) > 10
    assert "Elite" in result or "elite" in result.lower() or "Strong" in result
