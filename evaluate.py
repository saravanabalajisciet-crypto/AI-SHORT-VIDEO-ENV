"""
evaluate.py — Benchmark runner for AI Short-Form Video Optimization Environment.

Runs all tasks (task_1 through task_3) against a live environment server,
measures grader scores, and produces a structured pass/fail report.

Usage:
    # Against local server (default)
    python evaluate.py

    # Against HF Space or remote server
    python evaluate.py --env-url https://saravanabalajisara-ai-video-optimizer-env.hf.space

    # With LLM agent (mirrors validator execution)
    python evaluate.py --use-llm --api-base-url <url> --api-key <key> --model gpt-4o-mini

    # Single task
    python evaluate.py --task task_2

    # Save results to JSON
    python evaluate.py --output results.json

Exit codes:
    0 — all tasks passed
    1 — one or more tasks failed
    2 — environment unreachable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI or env vars)
# ---------------------------------------------------------------------------
DEFAULT_ENV_URL = (
    os.getenv("OPENENV_BASE_URL")
    or os.getenv("ENV_URL")
    or os.getenv("ENVIRONMENT_URL")
    or "http://localhost:7860"
).rstrip("/")

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "").rstrip("/")
DEFAULT_API_KEY  = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or ""
DEFAULT_MODEL    = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Task definitions: id → (seeds, target_score, platform)
TASKS: Dict[str, Dict[str, Any]] = {
    "task_1": {"seeds": [42],       "target": 0.65, "platform": "reels",  "level": "easy"},
    "task_2": {"seeds": [42],       "target": 0.78, "platform": "reels",  "level": "medium"},
    "task_3": {"seeds": [42, 7, 13],"target": 0.90, "platform": "reels",  "level": "hard"},
}

# Optimal action sequence (order-sensitive for task_3)
OPTIMAL_SEQUENCE = [
    "reorder_scenes",
    "boost_hook",
    "cut_scene",
    "enhance_pacing",
    "improve_transition",
    "smooth_cut",
    "sync_audio",
    "add_subtitles",
    "add_music",
]

PLATFORM_LIMITS = {"reels": 30.0, "shorts": 60.0, "tiktok": 60.0}
MAX_STEPS = 15
TIMEOUT   = 30
W = 68  # print width


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _req(method: str, url: str, env_headers: dict, **kwargs) -> Optional[dict]:
    """Single HTTP request with timeout. Returns parsed JSON or None."""
    try:
        r = requests.request(method, url, headers=env_headers, timeout=TIMEOUT, **kwargs)
        if r.status_code == 200:
            return r.json()
        print(f"  [WARN] {method} {url} → HTTP {r.status_code}", flush=True)
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to {url}", flush=True)
        return None
    except Exception as e:
        print(f"  [ERROR] {method} {url}: {e}", flush=True)
        return None


def post(base: str, path: str, headers: dict, **kwargs) -> Optional[dict]:
    return _req("POST", f"{base}{path}", headers, **kwargs)


def get(base: str, path: str, headers: dict) -> Optional[dict]:
    return _req("GET", f"{base}{path}", headers)


# ---------------------------------------------------------------------------
# LLM decision (optional — mirrors inference.py)
# ---------------------------------------------------------------------------

def _llm_decide(
    obs: dict,
    step_num: int,
    task_id: str,
    api_base: str,
    api_key: str,
    model: str,
    hint_text: str = "",
) -> Tuple[Optional[str], dict, Optional[str]]:
    """
    Call LLM via OpenAI-compatible proxy. Returns (action, params, error).
    Falls back to (None, {}, error) on failure.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key)
        prompt = (
            f"You are an expert short-form video editor optimizing for {task_id}.\n"
            f"Step: {step_num}{(' | Hint: ' + hint_text) if hint_text else ''}\n\n"
            f"State:\n"
            f"  engagement={obs.get('current_engagement_score', 0):.3f}  "
            f"retention={obs.get('avg_retention', 0):.3f}  "
            f"hook_strength={obs.get('hook_strength', 0):.3f}\n"
            f"  duration={obs.get('total_duration', 0):.1f}s  "
            f"platform_compliant={obs.get('platform_compliant', False)}  "
            f"subtitles={obs.get('subtitles_present', False)}  "
            f"music={obs.get('music_added', False)}\n\n"
            f"Valid actions: {', '.join(OPTIMAL_SEQUENCE)}\n\n"
            f"Reply with JSON only, no markdown:\n"
            f'  {{"action": "action_name", "parameters": {{}}}}'
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        return parsed.get("action", "enhance_pacing"), parsed.get("parameters", {}), None
    except Exception as e:
        return None, {}, str(e)


# ---------------------------------------------------------------------------
# Heuristic action builder
# ---------------------------------------------------------------------------

def _build_heuristic(obs: dict, platform: str) -> List[Tuple[str, dict]]:
    """Build the optimal heuristic action sequence from current observation."""
    actions: List[Tuple[str, dict]] = []
    scenes = obs.get("scenes", [])

    # 1. Reorder — put best hook scene first
    hooks = [s for s in scenes
             if s.get("scene_type") in ("hook", "highlight") and s.get("has_hook")]
    if hooks:
        best = max(hooks, key=lambda s: float(s.get("hook_strength", 0)))
        if scenes and scenes[0].get("id") != best.get("id"):
            order = [best["id"]] + [s["id"] for s in scenes if s["id"] != best["id"]]
            actions.append(("reorder_scenes", {"order": order}))

    # 2. Boost hook
    actions.append(("boost_hook", {}))

    # 3. Cut low-engagement filler scenes
    candidates = sorted(
        [s for s in scenes
         if float(s.get("engagement_score", 1.0)) < 0.30
         and s.get("scene_type") in ("filler", "transition")],
        key=lambda s: float(s.get("engagement_score", 0)),
    )
    for scene in candidates:
        actions.append(("cut_scene", {"scene_id": scene["id"]}))

    # 4. Trim if over platform limit
    limit = PLATFORM_LIMITS.get(platform, 30.0)
    if float(obs.get("total_duration", 0)) > limit:
        actions.append(("trim_duration", {"target_seconds": limit}))

    # 5–9. Quality actions in optimal order
    actions.append(("enhance_pacing", {}))
    actions.append(("improve_transition", {}))
    actions.append(("smooth_cut", {}))
    actions.append(("sync_audio", {}))
    actions.append(("add_subtitles", {}))
    actions.append(("add_music", {}))

    return actions


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env_url: str,
    env_headers: dict,
    task_id: str,
    seed: int,
    platform: str,
    use_llm: bool = False,
    api_base: str = "",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one episode. Returns result dict with score, passed, steps, etc."""

    default = {
        "task_id": task_id, "seed": seed, "score": 0.0,
        "passed": False, "steps": 0, "reward": 0.0,
        "engagement": 0.0, "retention": 0.0, "hook_strength": 0.0,
        "order_score": 0.0, "breakdown": {},
    }

    # Reset
    state = post(env_url, "/reset", env_headers, params={"platform": platform, "seed": seed})
    if not state:
        print(f"  [ERROR] /reset failed for {task_id} seed={seed}", flush=True)
        return default

    obs = state.get("observation", {})
    if verbose:
        print(
            f"  reset  eng={obs.get('current_engagement_score', 0):.3f}  "
            f"ret={obs.get('avg_retention', 0):.3f}  "
            f"dur={obs.get('total_duration', 0):.1f}s  "
            f"scenes={len(obs.get('scenes', []))}",
            flush=True,
        )

    done         = bool(state.get("done", False))
    step_count   = 0
    total_reward = 0.0
    rewards      = []

    # Optionally fire one LLM step first
    llm_action_used: Optional[str] = None
    if use_llm and api_base and api_key:
        hint_resp = get(env_url, "/hint", env_headers)
        hint_text = ""
        if hint_resp and hint_resp.get("best_action"):
            hint_text = f"{hint_resp['best_action']}: {hint_resp.get('reason', '')}"

        llm_action, llm_params, llm_err = _llm_decide(
            obs, step_count + 1, task_id, api_base, api_key, model, hint_text
        )
        if llm_err:
            print(f"  [LLM] fallback ({llm_err})", flush=True)
        elif llm_action and not done and step_count < MAX_STEPS:
            resp = post(env_url, "/step", env_headers,
                        json={"action_type": llm_action, "parameters": llm_params or {}})
            if resp:
                state        = resp.get("state", state)
                obs          = state.get("observation", {})
                done         = bool(resp.get("done", done))
                reward       = float(resp.get("reward", 0.0))
                total_reward += reward
                rewards.append(reward)
                step_count   += 1
                llm_action_used = llm_action
                valid = resp.get("info", {}).get("valid", True)
                tag = "OK" if valid else "!!"
                if verbose:
                    print(
                        f"  {tag} LLM  {llm_action:<22} r={reward:+.3f}  "
                        f"eng={obs.get('current_engagement_score', 0):.3f}",
                        flush=True,
                    )

    # Heuristic sequence
    heuristic = _build_heuristic(obs, platform)
    for action_type, params in heuristic:
        if done or step_count >= MAX_STEPS:
            break
        # Skip if LLM already did this action
        if llm_action_used == action_type:
            continue
        # Don't cut if too few scenes remain
        if action_type == "cut_scene":
            cur_scenes = obs.get("scenes", [])
            if len(cur_scenes) <= 3:
                continue

        resp = post(env_url, "/step", env_headers,
                    json={"action_type": action_type, "parameters": params})
        if not resp:
            continue

        state        = resp.get("state", state)
        obs          = state.get("observation", {})
        done         = bool(resp.get("done", done))
        reward       = float(resp.get("reward", 0.0))
        total_reward += reward
        rewards.append(reward)
        step_count   += 1

        valid = resp.get("info", {}).get("valid", True)
        tag = "OK" if valid else "!!"
        if verbose:
            print(
                f"  {tag} [{step_count:02d}] {action_type:<22} r={reward:+.3f}  "
                f"eng={obs.get('current_engagement_score', 0):.3f}  "
                f"ret={obs.get('avg_retention', 0):.3f}",
                flush=True,
            )

    # Grade
    grader = get(env_url, "/grader", env_headers)
    if not grader:
        print(f"  [WARN] /grader returned empty — score=0.0", flush=True)
        return {**default, "steps": step_count, "reward": round(total_reward, 4)}

    score     = float(grader.get("score", 0.0))
    breakdown = grader.get("breakdown", {})
    passed    = score >= TASKS[task_id]["target"]

    return {
        "task_id":      task_id,
        "seed":         seed,
        "score":        round(score, 4),
        "passed":       passed,
        "steps":        step_count,
        "reward":       round(total_reward, 4),
        "engagement":   round(float(obs.get("current_engagement_score", 0)), 4),
        "retention":    round(float(obs.get("avg_retention", 0)), 4),
        "hook_strength":round(float(obs.get("hook_strength", 0)), 4),
        "order_score":  round(float(breakdown.get("order_score", 0)), 4),
        "breakdown":    breakdown,
    }


# ---------------------------------------------------------------------------
# Task runner (averages across seeds)
# ---------------------------------------------------------------------------

def run_task(
    env_url: str,
    env_headers: dict,
    task_id: str,
    use_llm: bool = False,
    api_base: str = "",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all seeds for a task and return averaged result."""
    cfg     = TASKS[task_id]
    seeds   = cfg["seeds"]
    target  = cfg["target"]
    platform = cfg["platform"]

    seed_results = []
    for seed in seeds:
        if verbose:
            print(f"\n  ── seed={seed} ──", flush=True)
        result = run_episode(
            env_url, env_headers, task_id, seed, platform,
            use_llm, api_base, api_key, model, verbose,
        )
        seed_results.append(result)
        time.sleep(1)  # brief pause between seeds

    avg_score = round(sum(r["score"] for r in seed_results) / len(seed_results), 4)
    best      = max(seed_results, key=lambda r: r["score"])

    return {
        "task_id":     task_id,
        "level":       cfg["level"],
        "target":      target,
        "avg_score":   avg_score,
        "passed":      avg_score >= target,
        "seeds_run":   seeds,
        "seed_scores": {r["seed"]: r["score"] for r in seed_results},
        "best_score":  best["score"],
        "steps":       best["steps"],
        "engagement":  best["engagement"],
        "retention":   best["retention"],
        "hook_strength": best["hook_strength"],
        "order_score": best["order_score"],
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(results: List[Dict[str, Any]], elapsed: float) -> None:
    print(f"\n{'='*W}", flush=True)
    print(f"  EVALUATION REPORT", flush=True)
    print(f"{'─'*W}", flush=True)

    all_passed = True
    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        seed_info = ""
        if len(r["seeds_run"]) > 1:
            seed_info = "  seeds=" + str(r["seed_scores"])
        print(
            f"  [{status}] {r['task_id']:<8}  "
            f"score={r['avg_score']:.4f}  target={r['target']}  "
            f"level={r['level']:<7}  steps={r['steps']}"
            f"{seed_info}",
            flush=True,
        )
        if not r["passed"]:
            all_passed = False
            gap = round(r["target"] - r["avg_score"], 4)
            print(f"           gap={gap:+.4f}  "
                  f"eng={r['engagement']:.3f}  "
                  f"ret={r['retention']:.3f}  "
                  f"hook={r['hook_strength']:.3f}  "
                  f"order={r['order_score']:.3f}", flush=True)

    print(f"{'─'*W}", flush=True)
    avg_all = round(sum(r["avg_score"] for r in results) / max(len(results), 1), 4)
    overall = "ALL TASKS PASSED ✓" if all_passed else "SOME TASKS FAILED ✗"
    print(f"  {overall}", flush=True)
    print(f"  Average score: {avg_all:.4f}  |  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'='*W}\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate AI Video Optimizer environment across all benchmark tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env-url", default=DEFAULT_ENV_URL,
        help=f"Environment server URL (default: {DEFAULT_ENV_URL})",
    )
    parser.add_argument(
        "--task", choices=list(TASKS.keys()), default=None,
        help="Run a single task instead of all (default: run all)",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Fire one LLM step per episode (requires --api-base-url and --api-key)",
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE,
                        help="LiteLLM proxy base URL for LLM calls")
    parser.add_argument("--api-key",      default=DEFAULT_API_KEY,
                        help="API key for LLM proxy")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--output",       default=None,
                        help="Save results to this JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    env_url     = args.env_url.rstrip("/")
    verbose     = not args.quiet
    env_headers = {}
    hf_token    = os.getenv("HF_TOKEN", "")
    if hf_token:
        env_headers["Authorization"] = f"Bearer {hf_token}"

    print(f"\n{'='*W}", flush=True)
    print(f"  AI Short-Form Video Optimization — Evaluation", flush=True)
    print(f"  Env URL : {env_url}", flush=True)
    print(f"  LLM     : {'enabled (' + args.model + ')' if args.use_llm else 'disabled (heuristic only)'}", flush=True)
    print(f"  Tasks   : {args.task or 'all (' + ', '.join(TASKS) + ')'}", flush=True)
    print(f"{'='*W}", flush=True)

    # Health check
    health = get(env_url, "/health", env_headers)
    if not health:
        print(f"\n[ERROR] Cannot reach environment at {env_url}", flush=True)
        print(f"  Start the server with:  uvicorn server.app:app --port 7860", flush=True)
        return 2

    print(f"  Health  : {health.get('status', 'unknown')}", flush=True)

    # Determine which tasks to run
    task_ids = [args.task] if args.task else list(TASKS.keys())

    start    = time.time()
    results  = []

    for task_id in task_ids:
        cfg = TASKS[task_id]
        print(f"\n{'─'*W}", flush=True)
        print(
            f"  Task: {task_id.upper()}  level={cfg['level']}  "
            f"target={cfg['target']}  seeds={cfg['seeds']}",
            flush=True,
        )
        print(f"{'─'*W}", flush=True)

        result = run_task(
            env_url, env_headers, task_id,
            use_llm=args.use_llm,
            api_base=args.api_base_url,
            api_key=args.api_key,
            model=args.model,
            verbose=verbose,
        )
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"\n  [{status}] {task_id}  avg_score={result['avg_score']:.4f}  "
            f"target={result['target']}",
            flush=True,
        )

    elapsed = round(time.time() - start, 1)
    _print_report(results, elapsed)

    # Save output
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "env_url":   env_url,
            "model":     args.model if args.use_llm else "heuristic",
            "elapsed_s": elapsed,
            "all_passed": all(r["passed"] for r in results),
            "avg_score":  round(sum(r["avg_score"] for r in results) / max(len(results), 1), 4),
            "results":   results,
        }
        try:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  Results saved to: {args.output}", flush=True)
        except Exception as e:
            print(f"  [WARN] Could not save output: {e}", flush=True)

    all_passed = all(r["passed"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
