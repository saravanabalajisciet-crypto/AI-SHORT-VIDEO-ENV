"""
Baseline agent client — runs against the live API server.

Usage:
    python client.py [--host http://localhost:8000] [--platform reels|shorts|tiktok] [--seed 42]
"""
import argparse
import requests


def run(host: str, platform: str, seed: int) -> None:
    base = host.rstrip("/")

    # ── reset ──────────────────────────────────────────────────────────────────
    state = requests.post(f"{base}/reset", params={"platform": platform, "seed": seed}).json()
    obs   = state["observation"]

    W = 70
    print(f"\n{'═'*W}")
    print(f"  AI Short-Form Video Optimization  ·  v4.0")
    print(f"{'═'*W}")
    print(f"  episode  : {state['episode_id']}")
    print(f"  platform : {obs['platform']}   seed: {seed}")
    print(f"  scenes   : {len(obs['scenes'])}   duration: {obs['total_duration']}s")
    print(f"  engagement   : {obs['current_engagement_score']:.4f}")
    print(f"  retention    : {obs['avg_retention']:.4f}   watch_time: {obs['watch_time']:.1f}s")
    print(f"  hook_strength: {obs['hook_strength']:.4f}   pacing: {obs['pacing_score']:.4f}")
    print(f"  transition_q : {obs['avg_transition_quality']:.4f}   cut_smooth: {obs['avg_cut_smoothness']:.4f}   audio_sync: {obs['avg_audio_sync_score']:.4f}")
    print(f"  hook_first   : {obs['hook_first']}   compliant: {obs['platform_compliant']}")
    print(f"{'─'*W}")

    done     = False
    step_num = 0

    def do_step(action_type: str, parameters: dict = None):
        nonlocal state, done, step_num
        if done:
            return
        payload = {"action_type": action_type, "parameters": parameters or {}}
        resp    = requests.post(f"{base}/step", json=payload).json()
        state   = resp["state"]
        done    = resp["done"]
        step_num += 1
        o   = state["observation"]
        tag = "✓" if resp["info"].get("valid", True) else "✗"
        print(
            f"  step {step_num:02d} {tag} | {action_type:<20s} | "
            f"r={resp['reward']:+.3f} | "
            f"eng={o['current_engagement_score']:.3f} | "
            f"ret={o['avg_retention']:.3f} | "
            f"tq={o['avg_transition_quality']:.3f} | "
            f"cs={o['avg_cut_smoothness']:.3f} | "
            f"as={o['avg_audio_sync_score']:.3f}"
        )
        if not resp["info"].get("valid", True):
            print(f"           ↳ {resp['info'].get('reason', '')}")

    obs = state["observation"]

    # ── baseline strategy ──────────────────────────────────────────────────────
    # 1. Reorder: best hook scene first
    hook_candidates = [s for s in obs["scenes"]
                       if s["scene_type"] in ("hook", "highlight") and s["has_hook"]]
    if hook_candidates:
        best_hook = max(hook_candidates, key=lambda s: s["hook_strength"])
        if obs["scenes"][0]["id"] != best_hook["id"]:
            new_order = [best_hook["id"]] + [s["id"] for s in obs["scenes"] if s["id"] != best_hook["id"]]
            do_step("reorder_scenes", {"order": new_order})

    # 2. Boost hook
    do_step("boost_hook")

    # 3. Cut filler/transition < 0.30, keep >= 3 scenes
    for scene in sorted(
        [s for s in state["observation"]["scenes"]
         if s["engagement_score"] < 0.30 and s["scene_type"] in ("filler", "transition")],
        key=lambda s: s["engagement_score"],
    ):
        if len(state["observation"]["scenes"]) <= 3:
            break
        do_step("cut_scene", {"scene_id": scene["id"]})

    # 4. Trim duration if needed
    limit = 30.0 if platform == "reels" else 60.0
    if state["observation"]["total_duration"] > limit:
        do_step("trim_duration", {"target_seconds": limit})

    # 5. Enhance pacing
    do_step("enhance_pacing")

    # 6. Improve transition quality if below 0.70
    if state["observation"]["avg_transition_quality"] < 0.70:
        do_step("improve_transition")

    # 7. Smooth cuts if below 0.70
    if state["observation"]["avg_cut_smoothness"] < 0.70:
        do_step("smooth_cut")

    # 8. Sync audio if below 0.70
    if state["observation"]["avg_audio_sync_score"] < 0.70:
        do_step("sync_audio")

    # 9. Add subtitles
    if not state["observation"]["subtitles_present"]:
        do_step("add_subtitles")

    # 10. Add music
    do_step("add_music")

    # ── grader ─────────────────────────────────────────────────────────────────
    grader = requests.get(f"{base}/grader").json()
    print(f"\n{'─'*W}")
    print(f"  GRADER SCORE : {grader['score']:.4f}  {'✅ PASSED' if grader['passed'] else '❌ NOT PASSED'}")
    for k, v in grader["breakdown"].items():
        bar = "█" * int(v * 40)
        print(f"    {k:<25s}: {v:.4f}  {bar}")

    # ── AI feedback ────────────────────────────────────────────────────────────
    fb = requests.get(f"{base}/feedback").json()
    print(f"\n{'─'*W}")
    print(f"  AI FEEDBACK")
    print(f"  {fb['overall']}")
    if fb["tips"]:
        for tip in fb["tips"]:
            print(f"  • {tip}")
    else:
        print("  • No further improvements needed.")

    # ── tasks ──────────────────────────────────────────────────────────────────
    tasks = requests.get(f"{base}/tasks").json()
    print(f"\n{'─'*W}")
    print(f"  TASKS")
    for t in tasks:
        status = "✅ PASS" if grader["score"] >= t["target_score"] else "❌ FAIL"
        print(f"  [{t['level'].upper():6s}] {t['id']}  target={t['target_score']}  {status}")

    # ── baseline endpoint ──────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  /baseline — all tasks (seed=42)")
    results = requests.get(f"{base}/baseline").json()
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(
            f"  {status} {r['task_id']} | score={r['score']:.4f} | "
            f"eng={r['final_engagement']:.3f} | ret={r['final_retention']:.3f} | "
            f"tq={r['avg_transition_quality']:.3f} | cs={r['avg_cut_smoothness']:.3f} | "
            f"as={r['avg_audio_sync_score']:.3f} | steps={r['steps']}"
        )
        print(f"       {r['feedback']['overall']}")

    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="http://localhost:8000")
    parser.add_argument("--platform", default="reels", choices=["reels", "shorts", "tiktok"])
    parser.add_argument("--seed",     default=42, type=int)
    args = parser.parse_args()
    run(args.host, args.platform, args.seed)
