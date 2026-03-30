"""
inference.py — Baseline agent that runs a full episode against the environment.

Reads from environment variables:
  API_BASE_URL  : base URL of the running environment (default: http://localhost:8000)
  MODEL_NAME    : optional label for logging (default: baseline-heuristic)
  HF_TOKEN      : optional Hugging Face token for authenticated spaces

Usage:
  python inference.py
  API_BASE_URL=http://localhost:8000 python inference.py
"""

import os
import sys
import requests

# ── config ─────────────────────────────────────────────────────────────────────
BASE_URL   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME",   "baseline-heuristic")
HF_TOKEN   = os.getenv("HF_TOKEN",     "")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PLATFORM = "reels"
SEED     = 42

TASK_TARGETS = {"task_1": 0.65, "task_2": 0.78, "task_3": 0.875}

W = 68


# ── helpers ────────────────────────────────────────────────────────────────────

def post(path: str, json: dict = None, params: dict = None) -> dict:
    r = requests.post(f"{BASE_URL}{path}", json=json, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


# ── baseline strategy ──────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    """Run one full episode using the deterministic baseline strategy."""

    print(f"\n{'─'*W}")
    print(f"  Task: {task_id.upper()}  |  Model: {MODEL_NAME}  |  Seed: {SEED}")
    print(f"{'─'*W}")

    # 1. Reset
    state = post("/reset", params={"platform": PLATFORM, "seed": SEED})
    obs   = state["observation"]
    print(f"  Initial  eng={obs['current_engagement_score']:.3f}  "
          f"ret={obs['avg_retention']:.3f}  "
          f"dur={obs['total_duration']:.1f}s  "
          f"scenes={len(obs['scenes'])}")

    done     = False
    step_num = 0
    total_reward = 0.0

    def step(action_type: str, parameters: dict = None) -> None:
        nonlocal state, done, step_num, total_reward
        if done:
            return
        resp = post("/step", json={"action_type": action_type, "parameters": parameters or {}})
        state        = resp["state"]
        done         = resp["done"]
        reward       = resp["reward"]
        valid        = resp["info"].get("valid", True)
        total_reward += reward
        step_num     += 1
        o   = state["observation"]
        tag = "✓" if valid else "✗"
        print(f"  step {step_num:02d} {tag} | {action_type:<20s} | "
              f"r={reward:+.3f} | eng={o['current_engagement_score']:.3f} | "
              f"ret={o['avg_retention']:.3f} | hook={o['hook_strength']:.3f}")

    # 2. Reorder: best hook scene first
    scenes = obs["scenes"]
    hooks  = [s for s in scenes if s["scene_type"] in ("hook", "highlight") and s["has_hook"]]
    if hooks:
        best = max(hooks, key=lambda s: s["hook_strength"])
        if scenes[0]["id"] != best["id"]:
            order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
            step("reorder_scenes", {"order": order})

    # 3. Boost hook
    step("boost_hook")

    # 4. Cut low-engagement filler/transition scenes
    for scene in sorted(
        [s for s in state["observation"]["scenes"]
         if s["engagement_score"] < 0.30 and s["scene_type"] in ("filler", "transition")],
        key=lambda s: s["engagement_score"],
    ):
        if len(state["observation"]["scenes"]) <= 3:
            break
        step("cut_scene", {"scene_id": scene["id"]})

    # 5. Trim duration if over platform limit
    if state["observation"]["total_duration"] > 30.0:
        step("trim_duration", {"target_seconds": 30.0})

    # 6. Enhance pacing
    step("enhance_pacing")

    # 7. Production quality actions (conditional)
    if state["observation"]["avg_transition_quality"] < 0.70:
        step("improve_transition")
    if state["observation"]["avg_cut_smoothness"] < 0.70:
        step("smooth_cut")
    if state["observation"]["avg_audio_sync_score"] < 0.70:
        step("sync_audio")

    # 8. Add subtitles
    if not state["observation"]["subtitles_present"]:
        step("add_subtitles")

    # 9. Add music
    step("add_music")

    # 10. Score
    grader = get("/grader")
    score  = grader["score"]
    passed = score >= TASK_TARGETS[task_id]

    print(f"\n  GRADER SCORE : {score:.4f}  {'PASSED' if passed else 'FAILED'}")
    print(f"  Total reward : {total_reward:.4f}  |  Steps: {step_num}")
    for k, v in grader["breakdown"].items():
        if k != "final_score":
            print(f"    {k:<25s}: {v:.4f}")

    return {
        "task_id":    task_id,
        "score":      score,
        "passed":     passed,
        "steps":      step_num,
        "reward":     round(total_reward, 4),
        "engagement": state["observation"]["current_engagement_score"],
        "retention":  state["observation"]["avg_retention"],
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*W}")
    print(f"  AI Short-Form Video Optimization — Inference")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API     : {BASE_URL}")
    print(f"  Platform: {PLATFORM}  Seed: {SEED}")
    print(f"{'═'*W}")

    # Health check
    try:
        get("/state")
    except Exception:
        try:
            post("/reset", params={"platform": PLATFORM, "seed": SEED})
        except Exception as e:
            print(f"\n  ERROR: Cannot reach {BASE_URL}")
            print(f"  {e}")
            sys.exit(1)

    results = []
    for task_id in ["task_1", "task_2", "task_3"]:
        result = run_episode(task_id)
        results.append(result)

    # Summary
    print(f"\n{'═'*W}")
    print(f"  FINAL RESULTS")
    print(f"{'─'*W}")
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        target = TASK_TARGETS[r["task_id"]]
        print(f"  [{status}] {r['task_id']:<8} score={r['score']:.4f}  "
              f"target={target}  eng={r['engagement']:.3f}  "
              f"ret={r['retention']:.3f}  steps={r['steps']}")
        if not r["passed"]:
            all_passed = False

    print(f"{'─'*W}")
    print(f"  Overall: {'ALL TASKS PASSED' if all_passed else 'SOME TASKS FAILED'}")
    print(f"{'═'*W}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
