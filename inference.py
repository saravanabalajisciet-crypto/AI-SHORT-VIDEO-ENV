"""
inference.py -- Fault-tolerant agent that runs a full episode against the environment.

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
import time
import requests

# -- config ---------------------------------------------------------------------
BASE_URL   = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME",   "baseline-heuristic")
HF_TOKEN   = os.getenv("HF_TOKEN",     "")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PLATFORM     = "reels"
SEED         = 42
TIMEOUT      = 30       # seconds per request
MAX_RETRIES  = 5        # retries before giving up
MAX_STEPS    = 15       # hard cap -- never exceed

TASK_TARGETS = {"task_1": 0.65, "task_2": 0.78, "task_3": 0.875}

W = 68


# -- safe HTTP helpers ----------------------------------------------------------

def _request(method: str, path: str, **kwargs) -> dict:
    """
    Execute an HTTP request with retry logic and safe fallback.
    Returns parsed JSON dict, or {} on failure.
    """
    url = f"{BASE_URL}{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method, url,
                headers=HEADERS,
                timeout=TIMEOUT,
                **kwargs,
            )
            if resp.status_code != 200:
                print(f"  [WARN] {method} {path} → HTTP {resp.status_code} "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                return {}
            try:
                return resp.json()
            except Exception as json_err:
                print(f"  [WARN] JSON parse error on {path}: {json_err}")
                return {}
        except requests.exceptions.Timeout:
            print(f"  [WARN] Timeout on {path} (attempt {attempt}/{MAX_RETRIES})")
        except requests.exceptions.ConnectionError as e:
            print(f"  [WARN] Connection error on {path}: {e} (attempt {attempt}/{MAX_RETRIES})")
        except Exception as e:
            print(f"  [WARN] Unexpected error on {path}: {e} (attempt {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            time.sleep(1)
    print(f"  [ERROR] {method} {path} failed after {MAX_RETRIES} attempts -- using fallback.")
    return {}


def post(path: str, json: dict = None, params: dict = None) -> dict:
    return _request("POST", path, json=json, params=params)


def get(path: str) -> dict:
    return _request("GET", path)


# -- safe observation accessors -------------------------------------------------

def _obs(state: dict) -> dict:
    """Safely extract observation from state dict."""
    try:
        return state.get("observation", {}) or {}
    except Exception:
        return {}


def _get(d: dict, key: str, default=0.0):
    """Safe dict getter with default."""
    try:
        val = d.get(key, default)
        return val if val is not None else default
    except Exception:
        return default


# -- baseline strategy ----------------------------------------------------------

def run_episode(task_id: str) -> dict:
    """Run one full episode. Never raises -- always returns a result dict."""

    default_result = {
        "task_id":    task_id,
        "score":      0.0,
        "passed":     False,
        "steps":      0,
        "reward":     0.0,
        "engagement": 0.0,
        "retention":  0.0,
    }

    try:
        print(f"\n{'-'*W}")
        print(f"  Task: {task_id.upper()}  |  Model: {MODEL_NAME}  |  Seed: {SEED}")
        print(f"{'-'*W}")

        # 1. Reset
        state = post("/reset", params={"platform": PLATFORM, "seed": SEED})
        if not state:
            print(f"  [ERROR] /reset returned empty -- skipping {task_id}")
            print(f"  score=0.0")
            return default_result

        obs = _obs(state)
        print(f"  Initial  eng={_get(obs,'current_engagement_score'):.3f}  "
              f"ret={_get(obs,'avg_retention'):.3f}  "
              f"dur={_get(obs,'total_duration'):.1f}s  "
              f"scenes={len(_get(obs,'scenes',[]))}")

        done         = bool(state.get("done", False))
        step_num     = 0
        total_reward = 0.0

        def step(action_type: str, parameters: dict = None) -> None:
            nonlocal state, done, step_num, total_reward
            if done or step_num >= MAX_STEPS:
                return
            try:
                resp = post("/step", json={"action_type": action_type,
                                           "parameters": parameters or {}})
                if not resp:
                    print(f"  [WARN] /step {action_type} returned empty -- skipping")
                    return
                state        = resp.get("state", state)
                done         = bool(resp.get("done", done))
                reward       = float(resp.get("reward", 0.0))
                info         = resp.get("info", {}) or {}
                valid        = info.get("valid", True)
                total_reward += reward
                step_num     += 1
                o   = _obs(state)
                tag = "OK" if valid else "!!"
                print(f"  step {step_num:02d} {tag} | {action_type:<20s} | "
                      f"r={reward:+.3f} | eng={_get(o,'current_engagement_score'):.3f} | "
                      f"ret={_get(o,'avg_retention'):.3f} | "
                      f"hook={_get(o,'hook_strength'):.3f}")
            except Exception as e:
                print(f"  [WARN] step({action_type}) error: {e}")

        # 2. Reorder: best hook scene first
        try:
            scenes = list(_get(obs, "scenes", []))
            hooks  = [s for s in scenes
                      if isinstance(s, dict)
                      and s.get("scene_type") in ("hook", "highlight")
                      and s.get("has_hook")]
            if hooks:
                best = max(hooks, key=lambda s: float(s.get("hook_strength", 0.0)))
                if scenes and scenes[0].get("id") != best.get("id"):
                    order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
                    step("reorder_scenes", {"order": order})
        except Exception as e:
            print(f"  [WARN] reorder logic error: {e}")

        # 3. Boost hook
        step("boost_hook")

        # 4. Cut low-engagement filler/transition scenes
        try:
            current_scenes = list(_get(_obs(state), "scenes", []))
            candidates = sorted(
                [s for s in current_scenes
                 if isinstance(s, dict)
                 and float(s.get("engagement_score", 1.0)) < 0.30
                 and s.get("scene_type") in ("filler", "transition")],
                key=lambda s: float(s.get("engagement_score", 0.0)),
            )
            for scene in candidates:
                if len(list(_get(_obs(state), "scenes", []))) <= 3:
                    break
                if step_num >= MAX_STEPS:
                    break
                step("cut_scene", {"scene_id": scene["id"]})
        except Exception as e:
            print(f"  [WARN] cut_scene logic error: {e}")

        # 5. Trim duration if over platform limit
        try:
            if float(_get(_obs(state), "total_duration", 0.0)) > 30.0:
                step("trim_duration", {"target_seconds": 30.0})
        except Exception as e:
            print(f"  [WARN] trim_duration logic error: {e}")

        # 6. Enhance pacing
        step("enhance_pacing")

        # 7. Production quality actions (conditional)
        try:
            o = _obs(state)
            if float(_get(o, "avg_transition_quality", 1.0)) < 0.70:
                step("improve_transition")
            if float(_get(o, "avg_cut_smoothness", 1.0)) < 0.70:
                step("smooth_cut")
            if float(_get(o, "avg_audio_sync_score", 1.0)) < 0.70:
                step("sync_audio")
        except Exception as e:
            print(f"  [WARN] quality actions error: {e}")

        # 8. Add subtitles
        try:
            if not _get(_obs(state), "subtitles_present", False):
                step("add_subtitles")
        except Exception as e:
            print(f"  [WARN] add_subtitles logic error: {e}")

        # 9. Add music
        step("add_music")

        # 10. Grade
        score = 0.0
        breakdown = {}
        try:
            grader = get("/grader")
            if grader:
                score     = float(grader.get("score", 0.0))
                breakdown = grader.get("breakdown", {}) or {}
            else:
                print(f"  [WARN] /grader returned empty -- defaulting score to 0.0")
        except Exception as e:
            print(f"  [WARN] grader error: {e} -- defaulting score to 0.0")

        passed = score >= TASK_TARGETS.get(task_id, 1.0)
        final_obs = _obs(state)

        print(f"\n  GRADER SCORE : {score:.4f}  {'PASSED' if passed else 'FAILED'}")
        print(f"  Total reward : {total_reward:.4f}  |  Steps: {step_num}")
        for k, v in breakdown.items():
            if k != "final_score":
                try:
                    print(f"    {k:<25s}: {float(v):.4f}")
                except Exception:
                    pass

        return {
            "task_id":    task_id,
            "score":      score,
            "passed":     passed,
            "steps":      step_num,
            "reward":     round(total_reward, 4),
            "engagement": float(_get(final_obs, "current_engagement_score", 0.0)),
            "retention":  float(_get(final_obs, "avg_retention", 0.0)),
        }

    except Exception as e:
        print(f"  [ERROR] run_episode({task_id}) crashed: {e}")
        print(f"  score=0.0")
        return default_result


# -- main -----------------------------------------------------------------------

def main():
    try:
        print(f"\n{'='*W}")
        print(f"  AI Short-Form Video Optimization -- Inference")
        print(f"  Model   : {MODEL_NAME}")
        print(f"  API     : {BASE_URL}")
        print(f"  Platform: {PLATFORM}  Seed: {SEED}")
        print(f"{'='*W}")

        # Health check -- non-fatal
        try:
            health = get("/health")
            if health:
                print(f"  Health  : {health.get('status', 'unknown')}")
            else:
                # Try a reset as fallback health check
                probe = post("/reset", params={"platform": PLATFORM, "seed": SEED})
                if not probe:
                    print(f"  [WARN] Cannot confirm API health at {BASE_URL} -- proceeding anyway")
        except Exception as e:
            print(f"  [WARN] Health check failed: {e} -- proceeding anyway")

        results = []
        for task_id in ["task_1", "task_2", "task_3"]:
            try:
                result = run_episode(task_id)
            except Exception as e:
                print(f"  [ERROR] Unhandled error in run_episode({task_id}): {e}")
                result = {
                    "task_id":    task_id,
                    "score":      0.0,
                    "passed":     False,
                    "steps":      0,
                    "reward":     0.0,
                    "engagement": 0.0,
                    "retention":  0.0,
                }
                print(f"  score=0.0")
            results.append(result)

        # Summary
        print(f"\n{'='*W}")
        print(f"  FINAL RESULTS")
        print(f"{'-'*W}")
        all_passed = True
        for r in results:
            try:
                status = "PASS" if r.get("passed") else "FAIL"
                target = TASK_TARGETS.get(r.get("task_id", ""), 1.0)
                print(f"  [{status}] {r.get('task_id','?'):<8} "
                      f"score={float(r.get('score',0.0)):.4f}  "
                      f"target={target}  "
                      f"eng={float(r.get('engagement',0.0)):.3f}  "
                      f"ret={float(r.get('retention',0.0)):.3f}  "
                      f"steps={r.get('steps',0)}")
                if not r.get("passed"):
                    all_passed = False
            except Exception as e:
                print(f"  [WARN] Could not print result row: {e}")
                all_passed = False

        print(f"{'-'*W}")
        print(f"  Overall: {'ALL TASKS PASSED' if all_passed else 'SOME TASKS FAILED'}")
        print(f"{'='*W}\n")

    except Exception as e:
        print(f"\n[FATAL] Unhandled exception in main(): {e}")
        print("score=0.0")

    # Always exit 0
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] Top-level crash: {e}")
        print("score=0.0")
        sys.exit(0)
